import copy
from copy import deepcopy
import numpy as np
from functools import reduce

import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Identity

from models.SmoothingLayers.SmoothingLayer import SmoothingLayer
from models.SmoothingLayers.SmoothingBlock import SmoothingBlock

from models.utils import load_from_dict


class ModelSmoothing:

    def __init__(self,
                 model,
                 model_name,
                 device,
                 root_folder: str = '.'):

        self.model = copy.deepcopy(model)
        self.model_name = model_name

        self.device = device

        self.root_folder = root_folder
        self.weights_file_name = None


    def smooth_model(self,
                     kernel: str = 'Gaussian',
                     kernel_size: int = 5,
                     kernel_mean: float = 0,
                     kernel_var: float = .2,
                     replaceable_module_class: type(Module) = Conv2d,
                     use_smoothing_block: bool = False,
                     use_relu_n: bool = False,
                     remove_bn: bool = True,
                     load_weight: bool = False
                     ):

        # Find all the replaceable_module_class module that will be replaced with their smooth counterpart
        replaceable_modules = [(name, mod) for name, mod in self.model.named_modules() if isinstance(mod, replaceable_module_class)]

        # Find all the batch norm layers
        bn_layer = [(name, mod) for name, mod in self.model.named_modules() if isinstance(mod, BatchNorm2d) and remove_bn]

        # Replace all the convolutional layers
        for layer_name, layer_module in np.concatenate([replaceable_modules, bn_layer]):
            formatted_names = layer_name.split(sep='.')

            # If there are more than one names as a result of the separation, the Module containing the convolutional layer
            # is a nester Module
            if len(formatted_names) > 1:
                # In this case, access the nested layer iteratively using itertools.reduce and getattr
                container_layer = reduce(getattr, formatted_names[:-1], self.model)
            else:
                # Otherwise, the containing layer is the network itself (no nested blocks)
                container_layer = self.model

            if isinstance(layer_module, Conv2d):
                # Smooth module construction
                new_layer = SmoothingBlock(conv_layer=deepcopy(layer_module),
                                           smooth_layer=SmoothingLayer(channels=layer_module.out_channels,
                                                                       kernel=kernel,
                                                                       kernel_size=kernel_size,
                                                                       kernel_mean=kernel_mean,
                                                                       kernel_var=kernel_var,
                                                                       device=self.device),
                                           device=self.device)

            elif isinstance(layer_module, BatchNorm2d):
                new_layer = Identity()
            else:
                raise AttributeError(f'Unknown later module: {layer_module}')

            # Change the convolutional layer to its injectable counterpart
            setattr(container_layer, formatted_names[-1], new_layer)

        # Create the weight name
        self.weights_file_name = f'{kernel}_{self.model_name}_k{kernel_size}_v{int(kernel_var*100):03}'
        if use_smoothing_block:
            self.weights_file_name = f'{self.weights_file_name}_sb'
        if use_relu_n:
            self.weights_file_name = f'{self.weights_file_name}_rn'
        self.weights_file_name = f'{self.weights_file_name}.pt'

        if load_weight:
            load_from_dict(network=self.model,
                           device=self.device,
                           path=f'{self.root_folder}/models/pretrained_models/{self.weights_file_name}')


    def save_model(self):
        torch.save(self.model.state_dict(),
                   f'{self.root_folder}/models/pretrained_models/{self.weights_file_name}')
