import torch
import argparse
import os
import numpy as np
import time
from scipy.stats import pearsonr, spearmanr

from GenDataset import GenDataset
from models import MultiscaleDQP, Default


def sigmoid(paPSNR):
    a, b, c = [100.0, 0.0, 0.17544356]
    return a + (b - a) / (1.0 + torch.exp(-c * paPSNR))


def paPSNR(Pr, Pd, s):
    n = Pr.shape[0]
    MSE = torch.zeros(n, 32, 32, dtype=torch.float)

    for i in range(n):
        MSE[i] = 10 ** (s[i] / 10) * ((Pr[i] - Pd[i]) ** 2).mean()

    MSE[MSE[:] == 0.0] = 0.0000001

    psnr = 10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / MSE.mean())

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
        "--model", type=str, required=False, default=None, help="Checkpoint"
    )
    parser.add_argument(
        "--network", required=False, default="Default", help="Default of MultiscaleDSP?"
    )
    parser.add_argument("--distType", type=int, default=None, help="Distortion type")
    opt = parser.parse_args()

    dataloader = GenDataset(opt.dataroot, 32, int(opt.model.split("_")[0]))

    if opt.network.lower() == "default":
        net = Default()
    elif opt.network.lower() == "multiscaledqp":
        net = MultiscaleDQP()

    net.load_state_dict(torch.load(opt.model)["state_dict"])
    net = net.eval()

    QP, QPest = [], []

    print("# Starting testing...")
    for i, batch in enumerate(
        dataloader.iterate_minibatches(mode="test", distortion=opt.distType)
    ):
        print("Image %d" % (i))
        ref, dist, dmos = batch
        PrPatcehs, PdPatches = extractPatches(ref, dist)
        s = torch.zeros(PrPatcehs.shape[0], dtype=torch.float)
        for j in range(PrPatcehs.shape[0]):
            net = net.cuda()
            input = PrPatcehs[j].view(-1, 1, 32, 32)
            input = input.cuda()
            s[j] = net(input)
        # print(s)
        papsnr = paPSNR(PrPatcehs, PdPatches, s)
        qpest = sigmoid(papsnr)
        QPest.append(qpest.detach().numpy())
        QP.append(dmos)

    QP = np.array(QP).tolist()
    QPest = np.array(QPest).tolist()

    lcc = pearsonr(QP, QPest)
    srocc = spearmanr(QP, QPest)

    print(
        "LCC = %f\tpvalue = %f\nSROCC = %f\tpvalue = %f"
        % (lcc[0], lcc[1], srocc[0], srocc[1])
    )

    print("Done!")
