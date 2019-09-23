import torch

from models import Default, DenseDQP, MultiscaleDQP, MultiscaleQP
from thop import profile

input = torch.randn(1, 1, 32, 32)

net = Default()
netD = DenseDQP()
netMDQP = MultiscaleDQP()
netMQP = MultiscaleQP()

flops, params = profile(net, inputs=(input,))
flopsD, paramsD = profile(netD, inputs=(input,))
flopsMDQP, paramsMDQP = profile(netMDQP, inputs=(input,))
flopsMQP, paramsMQP = profile(netMQP, inputs=(input,))

print(
    "Default: FLOPS = %.4E\tParams = %.4E\nDenseDQP: FLOPS = %.4E\tParams = %.4E\nMultiscaleDQP: FLOPS = %.4E\tParams = %.4E\nMultiscaleQP: FLOPS = %.4E\tParams = %.4E"
    % (flops, params, flopsD, paramsD, flopsMDQP, paramsMDQP, flopsMQP, paramsMQP)
)
