import torch
import argparse
import os
import numpy as np
import time

from GenDataset import GenDataset
from models import MultiscaleDSP, Default


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


# Implement image segmentation


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
        "--network", required=False, default="Default", help="Default of MultiscaleDSP?"
    )
    opt = parser.parse_args()

    dataloader = GenDataset(opt.dataroot, 32, opt.batchSize)

    if opt.network.lower() == "default":
        net = Default()
    elif opt.network.lower() == "multiscaledsp":
        net = MultiscaleDSP()

    net.load_state_dict(torch.load(opt.model))
    net = net.eval()

    print("# Starting training...")
    for i, batch in enumerate(dataloader.iterate_minibatches(mode="test")):
        ref, dist, dmos = batch
