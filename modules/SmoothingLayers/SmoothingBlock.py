import torch
from torch.nn import Sequential, BatchNorm2d, Conv2d, ReLU, Hardtanh
from modules.SmoothingLayers.SmoothingLayer import SmoothingLayer
from modules.SmoothingLayers.ReLUn import ReLUn


class SmoothingBlock(Sequential):

    def __init__(self,
                 conv_layer: Conv2d,
                 smooth_layer: SmoothingLayer,
                 device: torch.device,
                 use_smoothing_block: bool = True,
                 use_relu_n: bool = True):
        super().__init__()

        self.device = device

        self.use_smoothing_block = use_smoothing_block
        self.use_relu_n = use_relu_n

        self.conv_layer = conv_layer
        self.bn_layer = BatchNorm2d(self.conv_layer.out_channels, device=device)

        if self.use_relu_n:
            self.relu_n = ReLUn(in_channels=self.conv_layer.out_channels, device=device)

        if self.use_smoothing_block:
            self.smooth_layer = smooth_layer


    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)

        if self.use_relu_n:
            x = self.relu_n(x)

        if self.use_smoothing_block:
            x = self.smooth_layer(x)

        return x
