import torch

from models import Default, DenseDQP, MultiscaleDQP
from thop import profile

input = torch.randn(1, 1, 32, 32)

net = Default()
netD = DenseDQP()
netM = MultiscaleDQP()

flops, params = profile(net, inputs=(input,))
flopsD, paramsD = profile(netD, inputs=(input,))
flopsM, paramsM = profile(netM, inputs=(input,))

print(
    "Default: FLOPS = %.4E\tParams = %.4E\nDenseDQP: FLOPS = %.4E\tParams = %.4E\nMultiscaleDQP: FLOPS = %.4E\tParams = %.4E"
    % (flops, params, flopsD, paramsD, flopsM, paramsM)
)
