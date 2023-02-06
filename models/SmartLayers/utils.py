from typing import Type

import torch
from torch.nn.modules import Sequential


class NoChangeOFMException(Exception):
    """
    Exception thrown when the execution of a faulty neural network doesn't produce any difference between its own
    output feature map and the output feature map of a clean network execution
    """
    pass


class DelayedStartModule(Sequential):

    def __init__(self):
        super(DelayedStartModule).__init__()

        self.layers = None

        self.starting_layer = None
        self.starting_module = None

    def forward(self,
                input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Smart forward used for fault delayed start. With this smart function, the inference starts from the first layer
        marked as starting layer and the input of that layer is loaded from disk
        :param input_tensor: The module input tensor
        :return: The module output tensor
        """

        # If the starting layer and starting module are set, proceed with the smart forward
        if self.starting_layer is not None:
            # Execute the layers iteratively, starting from the one where the fault is injected
            layer_index = self.layers.index(self.starting_layer)
        else:
            layer_index = 0

        if self.starting_module is not None:
            # Create a dummy input
            x = torch.zeros(size=self.starting_module.input_size, device='cuda')

            # Specify that the first module inside this layer should load the input from memory and not read from previous
            # layer
            self.starting_module.start_from_this_layer()
        else:
            x = input_tensor

        # Iteratively execute modules in the layer
        for layer in self.layers[layer_index:]:
            x = layer(x)

        if self.starting_module is not None:
            # Clear the marking on the first module
            self.starting_module.do_not_start_from_this_layer()

        return x
