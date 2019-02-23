import torch
from torch.optim import Adam, SGD
import argparse
import os
import numpy as np
import shutil
import time

from GenDataset import GenDataset
from models import MultiscaleDSP, Default


def saveChekpoint(
    epoch, model, optimizer, train_loss, val_loss, PATH, isBest=False, network="Default"
):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        PATH,
    )

    if isBest:
        shutil.copyfile(PATH, network.lower() + "_model_best.pth.tar")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def sensitivity(psnr, dmos):
    a, b, c = [96.16552146, 0.4231659, 0.17716917]

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
        "--network", required=False, default="Default", help="Default of MultiscaleDSP?"
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
    parser.add_argument("--distType", type=int, default=None, help="Distortion type")
    opt = parser.parse_args()

    if os.path.isdir(opt.networks):
        pass
    else:
        print("Creating networks dir...")
        os.mkdir(opt.networks)
        print("Done!")

    dataloader = GenDataset(opt.dataroot, 32, opt.batchSize, generate=True)

    if opt.network.lower() == "default":
        net = Default()
    elif opt.network.lower() == "multiscaledsp":
        net = MultiscaleDSP()
    net.apply(weights_init)

    criterion = torch.nn.L1Loss()
    optimizer = Adam(net.parameters(), opt.lr)

    train_error = []
    validation_error = []

    running_loss = []

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
        print("Done!")

    print("# Starting training...")
    for epoch in range(start, opt.epochs):
        for i, batch in enumerate(
            dataloader.iterate_minibatches(
                mode="train", distortion=opt.distType, shuffle=True
            )
        ):
            net = net.train()
            ref, dist, s = batch

            psnr = torch.tensor(PSNR(ref, dist), dtype=torch.float)

            ref = ref.cuda()
            net = net.cuda()
            s = s.cuda()

            optimizer.zero_grad()

            output = net(ref)

            criterion = criterion.cuda()
            ref = ref.cuda()
            dist = dist.cuda()
            output = output.cuda()

            error = criterion(output, s)
            error.backward()
            optimizer.step()

            running_loss.append(error.item())

            if i % 30 == 29:
                running_val_loss = []
                for batch in dataloader.iterate_minibatches(
                    mode="validation", distortion=opt.distType, shuffle=True
                ):
                    net = net.eval()
                    ref, dist, s = batch

                    psnr = torch.tensor(PSNR(ref, dist), dtype=torch.float)

                    ref = ref.cuda()
                    net = net.cuda()
                    s = s.cuda()

                    output = net(ref)

                    criterion = criterion.cuda()
                    ref = ref.cuda()
                    dist = dist.cuda()
                    output = output.cuda()

                    error = criterion(output, s)
                    running_val_loss.append(error.item())

                print(
                    "[%d, %5d] Training loss: %.5f\tValidation loss: %.5f"
                    % (
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
                    best,
                    opt.network,
                )

                running_loss = []
    print("Done!")
