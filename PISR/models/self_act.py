import torch
import torch.nn as nn


class PowerActivation(nn.Module):
    def __init__(self, init_alpha=1.):
        super(PowerActivation, self).__init__()
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.empty(1).fill_(init_alpha))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input) * torch.abs(input) ** self.alpha

    def extra_repr(self) -> str:
        return 'init_alpha={}\n alpha={}'.format(self.init_alpha, self.alpha)

