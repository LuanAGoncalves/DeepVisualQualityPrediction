import torch
import torch.nn as nn


class paPSNR(nn.Module):
    def forward(self, Pr, Pd, s, dmos):
        s = s.view(-1)
        n = s.shape[0]
        MSE = torch.zeros(n, dtype=torch.float)
        for i in range(n):
            MSE[i] = s[i] * ((Pr[i] - Pd[i]) ** 2).mean()

        return torch.abs(
            10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / MSE.mean())
            - dmos
        )

