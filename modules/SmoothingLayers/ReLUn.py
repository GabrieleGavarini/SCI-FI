import torch
from torch.nn.functional import hardtanh, relu
from torch.nn import Module


class ReLUn(Module):


    def __init__(self,
                 in_channels,
                 device: torch.device = torch.cpu,
                 min_value: float = 0):

        super(ReLUn, self).__init__()

        # Device
        self.device = device

        # Number of channels
        self.in_channels = in_channels

        # min is self.weight[0], max is self.weight[1]
        min_tensor = torch.ones(self.in_channels, device=self.device).mul(min_value)
        max_tensor = torch.ones(self.in_channels, device=self.device).mul(torch.inf)
        self.weight = torch.nn.Parameter(data=torch.stack([min_tensor, max_tensor]), requires_grad=False)


    def forward(self, x):
        # If in training mode, learn the maximum input
        if self.training:
            self.weight[0] = torch.amin(x, dim=(0, 2, 3)).detach()
            self.weight[1] = torch.amax(x, dim=(0, 2, 3)).detach()

        # Create a min tensor and a max tensor to apply a channel-wise cap to the relu
        min_tensor = self.weight[0].view(1, self.in_channels, 1, 1).repeat(x[:, :1].shape)
        max_tensor = self.weight[1].view(1, self.in_channels, 1, 1).repeat(x[:, :1].shape)
        x = torch.maximum(x, min_tensor)
        x = torch.minimum(x, max_tensor)

        return x
