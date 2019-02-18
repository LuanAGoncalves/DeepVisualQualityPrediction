import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from GenDataset import GenDataset


def sensitivity(psnr, dmos):
    psnr = torch.tensor(psnr, dtype=torch.float)
    dmos = torch.tensor(dmos, dtype=torch.float)
    a, b, c = [96.16552146, 0.4231659, 0.17716917]
    if dmos - a >= 0:
        print(dmos)
    s = torch.log(((b - a) / (dmos - a)) - 1) / c + psnr
    return s


dataset = GenDataset("./databaserelease2/", 32, 32)

trainset = dataset.trainset

PSNR, DMOS = np.array(trainset["psnr"]), np.array(trainset["dmos"])

DMOS[DMOS[:] >= 96.16552146] = 96.1654
DMOS[DMOS[:] <= 0.4231659] = 0.4232

s = [sensitivity(psnr, dmos) for psnr, dmos in zip(PSNR, DMOS)]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(PSNR, DMOS, s, c="r", marker="o")

ax.set_xlabel("PSNR")
ax.set_ylabel("DMOS")
ax.set_zlabel("Sencitivity")

plt.show()
