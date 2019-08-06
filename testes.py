import torch

from models import Default, MultiscaleDQP
from thop import profile

input = torch.randn(1, 1, 32, 32)

netD = Default()
netM = MultiscaleDQP()

flopsD, paramsD = profile(netD, inputs=(input,))
flopsM, paramsM = profile(netM, inputs=(input,))

print(
    "Default: FLOPS = %.4E\tParams = %.4E\nMultiscaleDQP: FLOPS = %.4E\tParams = %.4E"
    % (flopsD, paramsD, flopsM, paramsM)
)
