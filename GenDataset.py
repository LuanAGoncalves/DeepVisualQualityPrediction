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
    def __init__(self, dataroot, outputSize, batchSize):
        super(GenDataset, self).__init__()
        self.dataroot = dataroot
        self.folders = ["refimgs/", "jp2k/", "jpeg/", "wn/", "gblur/", "fastfading/"]
        self.trainset, self.validationset, self.testset = self.genDataset()
        self.outputSize = outputSize
        self.batchSize = batchSize
        self.crop = RandomCrop(output_size=32)

    def genDataset(self):
        refs = []
        dists = []
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
            elif i >= 227 and i < 460:
                dists.append(
                    self.dataroot + self.folders[2] + "img" + str(i - 227 + 1) + ".bmp"
                )
            elif i >= 460 and i < 634:
                dists.append(
                    self.dataroot + self.folders[3] + "img" + str(i - 460 + 1) + ".bmp"
                )
            elif i >= 634 and i < 808:
                dists.append(
                    self.dataroot + self.folders[4] + "img" + str(i - 634 + 1) + ".bmp"
                )
            elif i >= 808:
                dists.append(
                    self.dataroot + self.folders[5] + "img" + str(i - 808 + 1) + ".bmp"
                )

        df = pd.DataFrame(columns=["ref", "dist"])

        df["ref"] = refs
        df["dist"] = dists
        df["dmos"] = io.loadmat(self.dataroot + "dmos.mat")["dmos"].reshape(-1) / 100.0

        trainset = []
        validationset = []
        testset = []

        imgs = glob("./databaserelease2/refimgs/*.bmp")
        np.random.shuffle(imgs)

        for img in imgs[:17]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img:
                    trainset.append(i)

        for img in imgs[17 : 17 + 6]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img:
                    validationset.append(i)

        for img in imgs[23:]:
            for i in range(len(df)):
                if df.iloc[i]["ref"] == img:
                    testset.append(i)

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

        return (
            torch.tensor(RefPatches, dtype=torch.float).view(32, 1, 32, 32),
            torch.tensor(DistPatches, dtype=torch.float).view(32, 1, 32, 32),
            torch.tensor(batch["dmos"], dtype=torch.float),
        )

    def iterate_minibatches(self, batchsize=1, mode="train", shuffle=False):
        if mode.lower() == "train":
            dataset = self.trainset
        elif mode.lower() == "validation":
            dataset = self.validationset

        if shuffle:
            dataset = dataset.sample(frac=1).reset_index(drop=True)
        for i in range(len(dataset)):
            yield self.openBatch(dataset.iloc[i])


if __name__ == "__main__":
    dataset = GenDataset("./databaserelease2/", 32, 32)
    for i, batch in enumerate(dataset.iterate_minibatches(mode="train", shuffle=True)):
        print(i, batch[0].shape, batch[1].shape, batch[2])

