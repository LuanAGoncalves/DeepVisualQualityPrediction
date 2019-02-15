import torch
import torch.nn as nn


class paPSNR(nn.Module):
    def sensitivity(self, psnr, dmos):
        # psnr = torch.tensor(psnr, dtype=torch.float)
        # dmos = torch.tensor(dmos, dtype=torch.float)
        a, b, c = [63.34354131, 21.01046112, 0.28983582]
        return torch.log(((b - a) / (dmos - a)) - 1) / c + psnr

    def forward(self, Pr, Pd, dmos, s):
        a, b, c = [63.34354131, 21.01046112, 0.28983582]
        s = s.view(-1)
        s = torch.tensor(s, dtype=torch.float)
        dmos = torch.tensor(dmos, dtype=torch.float)
        n = s.shape[0]
        MSE = torch.zeros(n, dtype=torch.float)
        dmos = dmos * torch.ones(n, dtype=torch.float)

        for i in range(n):
            MSE[i] = ((Pr[i] - Pd[i]) ** 2).mean()

        PSNR = 10 * torch.log10(torch.tensor(255 ** 2, dtype=torch.float) / MSE)

        sGT = self.sensitivity(PSNR, dmos)

        print(type(s), type(sGT))

        return torch.abs(s - sGT).mean()
