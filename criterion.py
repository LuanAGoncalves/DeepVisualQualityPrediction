import torch
import torch.nn as nn


class paPSNR(nn.Module):
    def forward(self, Pr, Pd, dmos, s):
        a, b, c = [63.34354131, 21.01046112, 0.28983582]
        s = s.view(-1)
        n = s.shape[0]
        MSE = torch.zeros(n, dtype=torch.float)
        dmos = dmos * torch.ones(n, dtype=torch.float)

        for i in range(n):
            MSE[i] = 10 ** (s[i] / 10) * ((Pr[i] - Pd[i]) ** 2).mean()

        paPSNR = 10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / MSE)

        return torch.abs(
            (a + ((b - a) / (1.0 - torch.exp(-c * (paPSNR))))) - dmos
        ).mean()
