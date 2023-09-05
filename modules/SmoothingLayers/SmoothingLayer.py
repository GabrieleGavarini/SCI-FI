import math

import numpy as np

import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
from torch.nn import Conv2d
import torch


class SmoothingLayer(Conv2d):
    """
    Smoothing layer works as a depth-wise conv2d: it applies a convolution to all the element of the input tensor to
    produce an output tensor that has the same dimension of the input tensor.
    """

    def __init__(self,
                 channels: int,
                 stride: int = 1,
                 kernel: str = 'Gaussian',
                 kernel_size: int = 3,
                 kernel_mean: float = 0,
                 kernel_var: float = .2,
                 device: torch.device = 'cpu'):

        # TODO: manage stride != 1
        assert stride == 1

        super().__init__(in_channels=channels,
                         out_channels=channels,
                         groups=channels,
                         kernel_size=kernel_size,
                         padding=math.floor(kernel_size/2),
                         stride=stride,
                         bias=False,
                         device=device)

        # The device used to run the inference
        self.device = device

        # Parameters of the kernel
        self.kernel = kernel
        self.kernel_mean = kernel_mean
        self.kernel_var = kernel_var

        # Initialize the weights to the right kernel
        self.__initialize_kernel()

    def __initialize_kernel(self) -> None:
        """
        Initialize the kernel used for smoothing according to the specified kernel type
        """
        if self.kernel == 'Gaussian':
            # Create the kernel
            gaussian_kernel = self.__gaussian_kernel2d(kernel_size=self.kernel_size[0],
                                                       mean=self.kernel_mean,
                                                       sigma=math.sqrt(self.kernel_var))


            # Expand to match the weight size
            gaussian_kernel = np.tile(gaussian_kernel, (self.out_channels, 1, 1))

            # Convert the kernel to pytorch
            gaussian_kernel = torch.tensor(gaussian_kernel,
                                           dtype=torch.float32,
                                           device=self.device).unsqueeze(dim=1)

            # Assign the kernel to the weights of the layer
            self.weight.data = gaussian_kernel
        elif self.kernel == 'Median':
            median_kernel = 1
        else:
            raise AttributeError(f'No support for kernel {self.kernel}')

        # Disable gradient (i.e. no training)
        self.weight.requires_grad = False

    @staticmethod
    def __gaussian_kernel2d(kernel_size,
                            mean: float = 0,
                            sigma: float = 1):
        """
        Create a Gaussian Filter of size kernel size, using
        :param kernel_size: The size of the kernel
        :param sigma: The Gaussian sigma
        :param mean: The gaussian mean
        :return:
        """

        # Compute the 1d vector for the kernel size
        space_1d = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size) + mean

        # Compute the 1d gaussian distribution
        gaussian_1d_exponent = np.exp(-0.5 * ((space_1d - mean) / sigma)**2)
        gaussian_1d = (1 / sigma**2 * (math.sqrt(2 * math.pi))) * gaussian_1d_exponent

        # Compute the kernel 2d
        gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)

        # Normalize the kernel
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

        return gaussian_kernel

    def forward(self, x):
        if self.kernel == 'Gaussian':
            output = super().forward(x)
        elif self.kernel == 'Median':
            padded = F.pad(x, _quadruple(self.padding[0]), mode='reflect')
            unfolded = padded.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
            output = unfolded.contiguous().view(unfolded.size()[:4] + (-1,)).median(dim=-1)[0]
        else:
            raise NotImplementedError

        assert x.shape == output.shape

        return output
