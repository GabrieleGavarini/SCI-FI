import copy
from functools import reduce
import types

from typing import List

import torchinfo
import torch
from torch.nn import Module

from models.SmartLayers.SmartConv2d import SmartConv2d


class SmartLayersManager:

    def __init__(self,
                 network: Module,
                 module: Module = None):

        self.network = network
        self.module = module

    @staticmethod
    def __smart_forward(self, x):
        """
        Smart forward used for fault delayed start. With this smart function, the inference starts from the first layer
        marked as starting layer and the input of that layer is loaded from disk
        :param self:
        :param x:
        :return:
        """

        # Execute the layers iteratively, starting from the one where the fault is injected
        layer_index = self.layers.index(self.starting_layer)
        # Load the input of the layer
        # TODO: manage this: what happens if the convolutional layer is not the first layer of the network?
        x = self.starting_convolutional_layer.get_golden_ifm()
        for layer in self.layers[layer_index:]:
            x = layer(x)

        return x


    def replace_module_forward(self) -> None:
        """
        Replace the module forward function with a smart version
        """

        # Add the starting layer attribute
        self.module.starting_layer = None
        self.module.starting_convolutional_layer = None

        # Replace with the smart module function
        self.module.forward = types.MethodType(SmartLayersManager.__smart_forward, self.module)


    def replace_conv_layers(self,
                            device: torch.device,
                            fm_folder: str,
                            threshold: float = 0,
                            input_size: torch.Size = torch.Size((1, 3, 32, 32))) -> List[SmartConv2d]:
        """
        Replace all the convolutional layers of the network with injectable convolutional layers
        :param device: The device where the network is loaded
        :param fm_folder: The folder containing the input and output feature maps
        :param threshold: The threshold under which a folder has no impact
        :param input_size: torch.Size((32, 32, 3)). The torch.Size of an input image
        :return A list of all the new InjectableConv2d
        """

        # Find a list of all the convolutional layers
        convolutional_layers = [(name, copy.deepcopy(module)) for name, module in self.network.named_modules() if isinstance(module, torch.nn.Conv2d)]


        # Create a summary of the network
        summary = torchinfo.summary(self.network,
                                    device=device,
                                    input_size=input_size,
                                    verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_sizes = [torch.Size(info.output_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]
        input_sizes = [torch.Size(info.input_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]
        kernel_sizes = [torch.Size(info.kernel_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]

        # Initialize the list of all the injectable layers
        injectable_layers = list()

        # Replace all convolution layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(convolutional_layers):
            # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
            # ResNet, first separate the layer names using the '.'
            formatted_names = layer_name.split(sep='.')

            # If there are more than one names as a result of the separation, the Module containing the convolutional layer
            # is a nester Module
            if len(formatted_names) > 1:
                # In this case, access the nested layer iteratively using itertools.reduce and getattr
                container_layer = reduce(getattr, formatted_names[:-1], self.network)
            else:
                # Otherwise, the containing layer is the network itself (no nested blocks)
                container_layer = self.network


            # Create the injectable version of the convolutional layer
            faulty_convolutional_layer = SmartConv2d(conv_layer=layer_module,
                                                     device=device,
                                                     layer_name=layer_name,
                                                     input_size=input_sizes[layer_id],
                                                     output_size=output_sizes[layer_id],
                                                     kernel_size=kernel_sizes[layer_id],
                                                     fm_folder=fm_folder,
                                                     threshold=threshold)

            # Append the layer to the list
            injectable_layers.append(faulty_convolutional_layer)

            # Change the convolutional layer to its injectable counterpart
            setattr(container_layer, formatted_names[-1], faulty_convolutional_layer)

        # If the network has a layer list, regenerate to update the layers in the list
        if self.module is not None and callable(getattr(self.module, "generate_layer_list", None)):
            self.module.generate_layer_list()

        return injectable_layers