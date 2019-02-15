import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from GenDataset import GenDataset


def sensitivity(psnr, dmos):
    psnr = torch.tensor(psnr, dtype=torch.float)
    dmos = torch.tensor(dmos, dtype=torch.float)
    a, b, c = [100.0, 0.0, 0.23689602]
    if dmos - a >= 0:
        print(dmos)
    s = torch.log(((b - a) / (dmos - a)) - 1) / c + psnr
    return s


dataset = GenDataset("./databaserelease2/", 32, 32)

trainset = dataset.trainset

s = [
    sensitivity(psnr, dmos)
    for psnr, dmos in zip(list(trainset["psnr"]), list(trainset["dmos"]))
]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(list(trainset["psnr"]), list(trainset["dmos"]), s, c="r", marker="o")

ax.set_xlabel("PSNR")
ax.set_ylabel("DMOS")
ax.set_zlabel("Sencitivity")

plt.show()
