import torch
import argparse
import os
from glob import glob
import numpy as np
import time
from scipy.stats import pearsonr, spearmanr

from GenDataset import GenDataset
from models import DenseDQP, Default, MultiscaleDQP, MultiscaleQP


def sigmoid(paPSNR):
    a, b, c = [100.0, 0.0, 0.21713241]
    return a + (b - a) / (1.0 + torch.exp(-c * paPSNR))


def paPSNR(Pr, Pd, s):
    n = Pr.shape[0]
    MSE = torch.zeros(n, 32, 32, dtype=torch.float)

    for i in range(n):
        MSE[i] = torch.pow(10.0, s[i] / 10.0) * (torch.pow((Pr[i] - Pd[i]), 2))

    mse = MSE.view(-1).mean()

    if mse == 0.0:
        mse = 1e-28

    psnr = 10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / mse)

    return psnr


def extractPatches(Pr, Pd):
    _, _, h, w = Pr.shape
    if h % 32 == 0 and w % 32 == 0:
        H = h
        W = w
    elif h % 32 != 0 and w % 32 == 0:
        H = h + (32 - h % 32)
        W = w
    elif h % 32 == 0 and w % 32 != 0:
        W = w + (32 - w % 32)
        H = h
    else:
        H = h + (32 - h % 32)
        W = w + (32 - w % 32)
    PrOut = torch.zeros(1, 1, H, W, dtype=torch.float)
    PrOut[:, :, :h, :w] = Pr
    PdOut = torch.zeros(1, 1, H, W, dtype=torch.float)
    PdOut[:, :, :h, :w] = Pd

    PrPatches, PdPatches = [], []

    for i in range(0, H, 32):
        for j in range(0, W, 32):
            PrPatches.append(PrOut[:, :, i : i + 32, j : j + 32].numpy())
            PdPatches.append(PdOut[:, :, i : i + 32, j : j + 32].numpy())

    return (
        torch.tensor(PrPatches, dtype=torch.float).view(-1, 1, 32, 32),
        torch.tensor(PdPatches, dtype=torch.float).view(-1, 1, 32, 32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        required=False,
        default="./databaserelease2/",
        help="path to trn dataset",
    )
    parser.add_argument(
        "--folder", type=str, required=False, default=None, help="Models folder"
    )
    parser.add_argument(
        "--input", required=False, default="reference", help="Reference or Distorted?"
    )
    parser.add_argument(
        "--network", required=False, default="Default", help="Default of DenseDQP?"
    )
    parser.add_argument(
        "--scale_factor", type=int, required=False, default=0, help="Use scale factor?"
    )
    parser.add_argument("--distType", type=int, default=None, help="Distortion type")
    opt = parser.parse_args()

    LCC = []
    SROCC = []

    count = 0

    for model in glob(opt.folder + "*.pth.tar"):
        if opt.network.lower() == "default":
            net = Default(opt.scale_factor)
        elif opt.network.lower() == "densedqp":
            net = DenseDQP(opt.scale_factor)
        elif opt.network.lower() == "multiscaledqp":
            net = MultiscaleDQP(opt.scale_factor)
        elif opt.network.lower() == "multiscaleqp":
            net = MultiscaleDQP(opt.scale_factor)

        dataloader = GenDataset(
            opt.dataroot, 32, int(model.split("/")[-1].split("_")[0])
        )

        net.load_state_dict(torch.load(model)["state_dict"])
        net = net.eval()

        QP, QPest = [], []

        print("# Testing model %d" % (count))
        for i, batch in enumerate(
            dataloader.iterate_minibatches(mode="test", distortion=opt.distType)
        ):
            ref, dist, typeDist, dmos = batch
            _, _, m, n = ref.shape
            PrPatches, PdPatches = extractPatches(ref, dist)
            s = torch.zeros(PrPatches.shape[0], dtype=torch.float)
            for j in range(PrPatches.shape[0]):
                net = net.cuda()
                if opt.input.lower() == "reference":
                    input = PrPatches[j].view(-1, 1, 32, 32)
                elif opt.input.lower() == "distorted":
                    input = PdPatches[j].view(-1, 1, 32, 32)
                input = input.cuda()
                s[j] = net(input, typeDist)
            papsnr = paPSNR(PrPatches, PdPatches, s)
            qpest = sigmoid(papsnr)
            QPest.append(qpest.detach().numpy())
            QP.append(dmos)

        QP = np.array(QP).tolist()
        QPest = np.array(QPest).tolist()

        lcc = pearsonr(QP, QPest)
        srocc = spearmanr(QP, QPest)

        LCC.append(lcc[0])
        SROCC.append(srocc[0])
        count += 1

    LCC = np.array(LCC)
    SROCC = np.array(SROCC)

    print("LCC = %f\tSROCC = %f" % (LCC.mean(), SROCC.mean()))
