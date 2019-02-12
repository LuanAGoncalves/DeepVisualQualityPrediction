import torch
import torch.nn as nn
from math import e


class paPSNR(nn.Module):
    def forward(self, Pr, Pd, dmos, s):
        s = s.view(-1)
        n = s.shape[0]
        MSE = torch.zeros(n, dtype=torch.float)
        for i in range(n):
            MSE[i] = 10 ** (s[i] / 10) * ((Pr[i] - Pd[i]) ** 2).mean()

        paPSNR = 10 * torch.log10(
            torch.tensor(255 ** 2, dtype=torch.float) / MSE.mean()
        )

        if MSE.mean() != 0.0:
            return torch.abs(1.0 / (1.0 + e ** (-paPSNR)) - dmos)
        else:
            return MSE.mean()

