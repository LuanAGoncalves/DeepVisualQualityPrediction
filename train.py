import torch
from torch.optim import Adam, SGD
import argparse
import os
import numpy as np
import shutil
import time

from GenDataset import GenDataset
from models import MultiscaleDQP, Default


def saveChekpoint(
    epoch,
    model,
    optimizer,
    train_loss,
    val_loss,
    PATH,
    run,
    isBest=False,
    network="Default",
):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "run": run,
        },
        PATH,
    )

    if isBest:
        shutil.copyfile(PATH, str(run) + "_" + network.lower() + "_model_best.pth.tar")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def sensitivity(psnr, dmos):
    a, b, c = [100.0, 0.0, 0.17544356]

    s = torch.log(((b - a) / (dmos - a)) - 1) / c + psnr

    return s


def PSNR(Pr, Pd):
    n = Pr.shape[0]
    MSE = torch.zeros(n, dtype=torch.float)

    for i in range(n):
        MSE[i] = ((Pr[i] - Pd[i]) ** 2).mean()

    MSE[MSE[:] == 0.0] = 0.0000001

    psnr = 10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / MSE)

    return psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        required=False,
        default="./databaserelease2/",
        help="path to trn dataset",
    )
    parser.add_argument("--model", required=False, default=None, help="Checkpoint")
    parser.add_argument(
        "--generate", required=False, default=False, help="Generate dataset"
    )
    parser.add_argument(
        "--network", required=False, default="Default", help="Default of MultiscaleDQP?"
    )
    parser.add_argument(
        "--networks",
        required=False,
        default="./networks/",
        help="path to store the generated models",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs to train the model"
    )
    parser.add_argument("--batchSize", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    opt = parser.parse_args()

    if os.path.isdir(opt.networks):
        pass
    else:
        print("Creating networks dir...")
        os.mkdir(opt.networks)
        print("Done!")

    if opt.network.lower() == "default":
        net = Default()
    elif opt.network.lower() == "multiscaledqp":
        net = MultiscaleDQP()
    net.apply(weights_init)

    criterion = torch.nn.L1Loss()
    optimizer = Adam(net.parameters(), opt.lr)

    start_run = 0

    if opt.model == None:
        start = 0
    else:
        print("Loading checkpoint...")
        checkpoint = torch.load(opt.model)
        net = net.cuda()
        net.load_state_dict(checkpoint["state_dict"])
        start = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_error = checkpoint["train_loss"]
        validation_error = checkpoint["val_loss"]
        start_run = checkpoint["run"]
        print("Done!")

    for n in range(start_run, 10):
        if opt.network.lower() == "default":
            dataloader = GenDataset(opt.dataroot, 32, n, opt.batchSize, generate=True)
        elif opt.network.lower() == "multiscaledqp":
            dataloader = GenDataset(opt.dataroot, 32, n, opt.batchSize, generate=False)
        train_error = []
        validation_error = []
        running_loss = []
        print("# Starting training...")
        for epoch in range(start, opt.epochs):
            for i, batch in enumerate(
                dataloader.iterate_minibatches(mode="train", shuffle=True)
            ):
                net = net.train()
                ref, _, s = batch

                ref = ref.cuda()
                net = net.cuda()
                s = s.cuda()

                optimizer.zero_grad()

                output = net(ref)

                criterion = criterion.cuda()
                ref = ref.cuda()
                output = output.cuda()

                error = criterion(output, s)
                error.backward()
                optimizer.step()

                running_loss.append(error.item())

                if i % 30 == 29:
                    running_val_loss = []
                    for batch in dataloader.iterate_minibatches(
                        mode="validation", shuffle=True
                    ):
                        net = net.eval()
                        ref, _, s = batch

                        ref = ref.cuda()
                        net = net.cuda()
                        s = s.cuda()

                        output = net(ref)

                        criterion = criterion.cuda()
                        ref = ref.cuda()
                        output = output.cuda()

                        error = criterion(output, s)
                        running_val_loss.append(error.item())

                    print(
                        "[%d][%d, %5d] Training loss: %.5f\tValidation loss: %.5f"
                        % (
                            n,
                            epoch + 1,
                            i + 1,
                            np.mean(running_loss),
                            np.mean(running_val_loss),
                        )
                    )

                    if len(validation_error) == 0:
                        best = True
                    elif np.mean(running_val_loss) < min(validation_error):
                        best = True
                    else:
                        best = False

                    train_error.append(np.mean(running_loss))
                    validation_error.append(np.mean(running_val_loss))

                    saveChekpoint(
                        epoch,
                        net,
                        optimizer,
                        train_error,
                        validation_error,
                        opt.networks + str(i) + ".pth.tar",
                        n,
                        best,
                        opt.network,
                    )

                    running_loss = []
        net.apply(weights_init)
        optimizer = Adam(net.parameters(), opt.lr)
    print("Done!")
