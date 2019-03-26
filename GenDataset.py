import torch
from torch import nn
import torchvision
import os, sys, tarfile
import pandas as pd
import numpy as np
from glob import glob
import argparse
import cv2
import scipy.io as io


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, groundTruth):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]

        groundTruth = groundTruth[top : top + new_h, left : left + new_w]

        return image, groundTruth


class GenDataset(nn.Module):
    def __init__(self, dataroot, outputSize, run, batchSize=32, generate=False):
        super(GenDataset, self).__init__()
        self.eps = 1e-20
        self.dataroot = dataroot
        self.folders = ["refimgs/", "jp2k/", "jpeg/", "wn/", "gblur/", "fastfading/"]
        self.refImgs = glob(self.dataroot + self.folders[0] + "*.bmp")
        self.run = run
        if generate == True:
            self.trainset, self.validationset, self.testset = self.genDataset()
        else:
            self.trainset, self.validationset, self.testset = (
                pd.read_pickle(self.dataroot + str(self.run) + "_trainset.pkl"),
                pd.read_pickle(self.dataroot + str(self.run) + "_validationset.pkl"),
                pd.read_pickle(self.dataroot + str(self.run) + "_testset.pkl"),
            )
        self.outputSize = outputSize
        self.batchSize = batchSize
        self.crop = RandomCrop(output_size=32)

    def calcPSNR(self, ref, dist):
        ref = torch.tensor(
            cv2.cvtColor(cv2.imread(ref), cv2.COLOR_BGR2GRAY), dtype=torch.float
        )
        dist = torch.tensor(
            cv2.cvtColor(cv2.imread(dist), cv2.COLOR_BGR2GRAY), dtype=torch.float
        )

        mse = ((ref - dist) ** 2).mean() + 1e-18

        return 10 * torch.log10(255 ** 2 / mse)

    def calcPatchPSNR(self, ref, dist):
        mse = ((ref - dist) ** 2).mean() + 1e-18

        return 10 * torch.log10(255 ** 2 / mse)

    def sensitivity(self, psnr, dmos):
        a, b, c = [100.0, 0.0, 0.17544356]

        s = torch.log(((b - a) / (dmos - a)) - 1) / c + psnr

        return s

    def genDataset(self):
        refs = []
        dists = []
        typeDist = []
        refnames_all = [
            x[0]
            for x in io.loadmat(self.dataroot + "refnames_all.mat")["refnames_all"][0]
        ]
        for i in range(len(refnames_all)):
            refs.append(self.dataroot + self.folders[0] + refnames_all[i])
            if i < 227:
                dists.append(
                    self.dataroot + self.folders[1] + "img" + str(i + 1) + ".bmp"
                )
                typeDist.append(self.folders[1].split("/")[0])
            elif i >= 227 and i < 460:
                dists.append(
                    self.dataroot + self.folders[2] + "img" + str(i - 227 + 1) + ".bmp"
                )
                typeDist.append(self.folders[2].split("/")[0])
            elif i >= 460 and i < 634:
                dists.append(
                    self.dataroot + self.folders[3] + "img" + str(i - 460 + 1) + ".bmp"
                )
                typeDist.append(self.folders[3].split("/")[0])
            elif i >= 634 and i < 808:
                dists.append(
                    self.dataroot + self.folders[4] + "img" + str(i - 634 + 1) + ".bmp"
                )
                typeDist.append(self.folders[4].split("/")[0])
            elif i >= 808:
                dists.append(
                    self.dataroot + self.folders[5] + "img" + str(i - 808 + 1) + ".bmp"
                )
                typeDist.append(self.folders[5].split("/")[0])

        df = pd.DataFrame(columns=["ref", "dist"])

        df["ref"] = refs
        df["typeDist"] = typeDist
        df["dist"] = dists
        df["orgs"] = io.loadmat(self.dataroot + "dmos.mat")["orgs"].reshape(-1)
        df["psnr"] = [float(self.calcPSNR(x, y).numpy()) for x, y in zip(refs, dists)]
        DMOS = io.loadmat(self.dataroot + "dmos_realigned.mat")["dmos_new"].reshape(-1)
        DMOS[DMOS[:] >= 100.0] = 99.9999
        DMOS[DMOS[:] <= 0.0] = 0.0001
        df["dmos"] = DMOS
        df["std"] = io.loadmat(self.dataroot + "dmos_realigned.mat")[
            "dmos_std"
        ].reshape(-1)
        # df["sensitivity"] = [
        #     self.sensitivity(
        #         torch.tensor(psnr, dtype=torch.float),
        #         torch.tensor(dmos, dtype=torch.float),
        #     )
        #     for psnr, dmos in zip(list(df["psnr"]), list(df["dmos"]))
        # ]

        trainset = []
        validationset = []
        testset = []

        imgs = self.refImgs
        np.random.shuffle(imgs)

        for img in imgs[:17]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img and df.iloc[i]["orgs"] == 0:
                    trainset.append(i)

        for img in imgs[17 : 17 + 6]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img and df.iloc[i]["orgs"] == 0:
                    validationset.append(i)

        for img in imgs[23:]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img and df.iloc[i]["orgs"] == 0:
                    testset.append(i)

        df.iloc[trainset].to_pickle(self.dataroot + str(self.run) + "_trainset.pkl")
        df.iloc[validationset].to_pickle(
            self.dataroot + str(self.run) + "_validationset.pkl"
        )
        df.iloc[testset].to_pickle(self.dataroot + str(self.run) + "_testset.pkl")

        return df.iloc[trainset], df.iloc[validationset], df.iloc[testset]

    def openBatch(self, batch):
        ref = batch["ref"]
        dist = batch["dist"]

        RefImg = np.array(cv2.cvtColor(cv2.imread(ref), cv2.COLOR_RGB2GRAY), np.float32)
        DistImg = np.array(
            cv2.cvtColor(cv2.imread(dist), cv2.COLOR_RGB2GRAY), np.float32
        )

        RefPatches = []
        DistPatches = []

        while len(RefPatches) < self.batchSize:
            img, dist = self.crop(RefImg, DistImg)
            RefPatches.append(img)
            DistPatches.append(dist)

        refs = torch.tensor(RefPatches, dtype=torch.float).view(
            self.batchSize, 1, 32, 32
        )
        dists = torch.tensor(DistPatches, dtype=torch.float).view(
            self.batchSize, 1, 32, 32
        )

        psnr = torch.tensor(
            [self.calcPatchPSNR(x, y) for x, y in zip(refs, dists)], dtype=torch.float
        )
        dmos = torch.tensor(np.array(batch["dmos"]) * np.ones((32)), dtype=torch.float)

        batchSensitivity = torch.tensor(
            [self.sensitivity(x, y) for x, y in zip(psnr, dmos)], dtype=torch.float
        )

        return (refs, dists, batchSensitivity)

    def openBatchValidation(self, batch, cutPoint):
        ref = batch["ref"]
        dist = batch["dist"]
        RefImgs = []
        DistImgs = []

        RefImg = np.array(cv2.cvtColor(cv2.imread(ref), cv2.COLOR_RGB2GRAY), np.float32)
        DistImg = np.array(
            cv2.cvtColor(cv2.imread(dist), cv2.COLOR_RGB2GRAY), np.float32
        )
        for i in range(32):
            RefImgs.append(
                RefImg[
                    cutPoint[i, 0] : cutPoint[i, 0] + 32,
                    cutPoint[i, 1] : cutPoint[i, 1] + 32,
                ]
            )
            DistImgs.append(
                DistImg[
                    cutPoint[i, 0] : cutPoint[i, 0] + 32,
                    cutPoint[i, 1] : cutPoint[i, 1] + 32,
                ]
            )

        refs = torch.tensor(RefImgs, dtype=torch.float).view(32, 1, 32, 32)
        dists = torch.tensor(DistImgs, dtype=torch.float).view(32, 1, 32, 32)

        psnr = torch.tensor(
            [self.calcPatchPSNR(x, y) for x, y in zip(refs, dists)], dtype=torch.float
        )
        dmos = torch.tensor(np.array(batch["dmos"]) * np.ones((32)), dtype=torch.float)
        batchSensitivity = torch.tensor(
            [self.sensitivity(x, y) for x, y in zip(psnr, dmos)], dtype=torch.float
        )

        return (refs, dists, batchSensitivity)

    def openBatchTest(self, batch):
        ref = batch["ref"]
        dist = batch["dist"]

        RefImg = np.array(cv2.cvtColor(cv2.imread(ref), cv2.COLOR_RGB2GRAY), np.float32)
        DistImg = np.array(
            cv2.cvtColor(cv2.imread(dist), cv2.COLOR_RGB2GRAY), np.float32
        )

        h, w = RefImg.shape

        return (
            torch.tensor(RefImg, dtype=torch.float).view(1, 1, h, w),
            torch.tensor(DistImg, dtype=torch.float).view(1, 1, h, w),
            batch["dmos"],
        )

    def cutPoints(self, dataset):
        points = []
        refs = list(dataset["ref"])

        n = len(refs)

        for i in range(n):
            RefImg = np.array(
                cv2.cvtColor(cv2.imread(refs[i]), cv2.COLOR_RGB2GRAY), np.float32
            )
            h, w = RefImg.shape
            auxList = []
            for j in range(32):
                top = np.random.randint(0, h - 32)
                left = np.random.randint(0, w - 32)
                auxList.append([top, left])
            points.append(auxList)

        return np.array(points)

    def iterate_minibatches(
        self, batchsize=1, mode="train", distortion=None, shuffle=False
    ):
        distTypes = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
        if mode.lower() == "train":
            dataset = self.trainset
        elif mode.lower() == "validation":
            dataset = self.validationset
            points = self.cutPoints(dataset)
        elif mode.lower() == "test":
            dataset = self.testset

        if distortion == None:
            pass
        else:
            dataset = dataset[dataset["typeDist"] == distTypes[distortion]]

        if mode.lower() == "train":
            if shuffle:
                dataset = dataset.sample(frac=1).reset_index(drop=True)
            for i in range(len(dataset)):
                yield self.openBatch(dataset.iloc[i])
        elif mode.lower() == "validation":
            for i in range(len(dataset)):
                yield self.openBatchValidation(dataset.iloc[i], points[i])
        elif mode.lower() == "test":
            for i in range(len(dataset)):
                yield self.openBatchTest(dataset.iloc[i])


def sigmoid(x, c, d):
    return 100.0 - 100.0 / (1.0 + np.exp(-c * (x - d)))


if __name__ == "__main__":
    dataset = GenDataset("./databaserelease2/", 32, 0, generate=True)
    trainset = dataset.trainset[dataset.trainset["typeDist"] == "jpeg"]
    folders = dataset.folders
    ref = dataset.refImgs

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    error = 10000000000000000000000000000
    popt = []
    for i in range(1000):
        pop, pcov = curve_fit(sigmoid, list(trainset["psnr"]), list(trainset["dmos"]))
        pcov = np.array(pcov)
        aux = (
            np.array(trainset["dmos"])
            - np.array([sigmoid(x, *pop) for x in list(trainset["psnr"])])
        ).mean()
        if aux < error:
            error = aux
            popt = pop
    print(popt)

    refs = glob("./databaserelease2/refimgs/*.bmp")

    ax = plt.subplot(111)

    for ref in refs:
        if (
            list((trainset[trainset["ref"] == ref]).sort_values(by=["psnr"])["psnr"])
            != []
        ):
            ax.plot(
                list(
                    (trainset[trainset["ref"] == ref]).sort_values(by=["psnr"])["psnr"]
                ),
                list(
                    (trainset[trainset["ref"] == ref]).sort_values(by=["psnr"])["dmos"]
                ),
                "o-.",
                label="fit",
                linewidth=1,
            )

    auxPSNR = list(trainset["psnr"])
    auxPSNR.sort()

    ax.plot(
        list(auxPSNR),
        [sigmoid(x, *popt) for x in auxPSNR],
        "-",
        label="data",
        linewidth=4,
        color="black",
    )
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.grid()

    plt.xlabel("PSNR")
    plt.ylabel("DMOS")
    # plt.legend()
    plt.show()
