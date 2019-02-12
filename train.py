import torch
from torch.optim import Adam
import argparse
import os

from GenDataset import GenDataset
from criterion import paPSNR
from models import MultiscaleDSP, Default

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        required=False,
        default="./databaserelease2/",
        help="path to trn dataset",
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
    parser.add_argument("--lr", type=int, default=0.01, help="learning rate")
    opt = parser.parse_args()

    if os.path.isdir(opt.networks):
        pass
    else:
        print("Creating networks dir...")
        os.mkdir(opt.networks)
        print("Done!")

    dataloader = GenDataset(opt.dataroot, 32, opt.batchSize)

    net = Default()

    criterion = paPSNR()
    optimizer = Adam(net.parameters(), opt.lr)

    train_error = []
    validation_error = []

    for epoch in range(opt.epochs):
        for i, batch in enumerate(
            dataloader.iterate_minibatches(mode="train", shuffle=True)
        ):
            ref, dist, dmos = batch
            ref = ref.cuda()
            net = net.cuda()

            optimizer.zero_grad()

            output = net(ref)

            criterion = criterion.cuda()
            ref = ref.cuda()
            dist = dist.cuda()
            output = output.cuda()
            dmos = dmos.cuda()

            error = criterion(ref, dist, dmos, output)
            error.backward()
            optimizer.step()
            train_error.append(error.item())

            print(error)
