import torch
from torch.optim import Adam, SGD
import argparse
import os
import numpy as np
import shutil
import time

from GenDataset import GenDataset
from models import DenseDQP, Default, MultiscaleDQP
import Visualizations


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
        m.weight.data = torch.nn.init.kaiming_normal(m.weight.data)
    elif classname.find("Linear") != -1:
        m.weight.data = torch.nn.init.xavier_normal_(
            m.weight.data, torch.nn.init.calculate_gain("relu")
        )
    elif classname.find("BatchNorm") != -1:
        m.weight.data = torch.nn.init.normal_(m.weight.data, 0.0, 1.0)
        m.bias.data = torch.nn.init.constant_(m.bias.data, 0)


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
        "--input", required=False, default="reference", help="Reference or Distorted?"
    )
    parser.add_argument("--visdom", required=False, default=1, help="Use Visdom?")
    parser.add_argument(
        "--scale_factor", type=int, required=False, default=0, help="Use scale factor?"
    )
    parser.add_argument(
        "--generate", required=False, default=False, help="Generate dataset"
    )
    parser.add_argument(
        "--network", required=False, default="Default", help="Default of DenseDQP?"
    )
    parser.add_argument(
        "--networks",
        required=False,
        default="./networks/",
        help="path to store the generated models",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train the model"
    )
    parser.add_argument("--batchSize", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    opt = parser.parse_args()

    if os.path.isdir(opt.networks):
        pass
    else:
        print("Creating networks dir...")
        os.mkdir(opt.networks)
        print("Done!")

    criterion = torch.nn.L1Loss()

    start_run = 0
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start = 0

    X = []
    Y_train, Y_validation = [], []

    torch.manual_seed(100)

    for n in range(start_run, 30):
        count = 0
        X = []
        Y_train, Y_validation = [], []
        if opt.visdom == 1:
            plot = Visualizations.Plot("Model %d" % (n))
            plot.register_line("Loss", "Epoch", "Loss")
        if opt.network.lower() == "default":
            dataloader = GenDataset(opt.dataroot, 32, n, opt.batchSize, generate=True)
            net = Default(opt.scale_factor)
        elif opt.network.lower() == "densedqp":
            dataloader = GenDataset(opt.dataroot, 32, n, opt.batchSize, generate=False)
            net = DenseDQP(opt.scale_factor)
        elif opt.network.lower() == "multiscaledqp":
            dataloader = GenDataset(opt.dataroot, 32, n, opt.batchSize, generate=False)
            net = MultiscaleDQP(opt.scale_factor)
        train_error = []
        validation_error = []
        running_loss = []
        net.apply(weights_init)
        optimizer = Adam(net.parameters(), opt.lr)
        print("# Starting training...")
        for epoch in range(start, opt.epochs):
            for i, batch in enumerate(
                dataloader.iterate_minibatches(mode="train", shuffle=True)
            ):
                net = net.train()
                if opt.input.lower() == "reference":
                    ref, _, typeDist, s = batch
                elif opt.input.lower() == "distorted":
                    _, ref, typeDist, s = batch

                ref = ref.cuda()
                net = net.cuda()
                s = s.cuda()
                typeDist = typeDist.cuda()

                optimizer.zero_grad()

                output = net(ref, typeDist)
                output = output.view(-1)

                criterion = criterion.cuda()
                ref = ref.cuda()
                output = output.cuda()

                error = criterion(output, s)
                error.backward()
                optimizer.step()

                running_loss.append(error.item())

                if i % 30 == 29:
                    count += 1
                    X.append(count)
                    running_val_loss = []
                    for batch in dataloader.iterate_minibatches(
                        mode="validation", shuffle=True
                    ):
                        net = net.eval()
                        if opt.input.lower() == "reference":
                            ref, _, typeDist, s = batch
                        elif opt.input.lower() == "distorted":
                            _, ref, typeDist, s = batch

                        ref = ref.cuda()
                        net = net.cuda()
                        s = s.cuda()
                        typeDist = typeDist.cuda()

                        output = net(ref, typeDist)
                        output = output.view(-1)

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

                    Y_train.append(np.mean(running_loss))
                    Y_validation.append(np.mean(running_val_loss))

                    if opt.visdom == 1:
                        plot.update_line(
                            "Loss",
                            np.column_stack([X, X]),
                            np.column_stack([Y_train, Y_validation]),
                        )

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
