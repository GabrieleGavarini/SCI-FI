import os
import argparse

import numpy as np
import pandas as pd

from typing import Union, List, Tuple

import torch
from torch.nn import Sequential, Module
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights, resnet18, resnet50
from torch.utils.data import DataLoader

import models
from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.WeightFault import WeightFault
from models.SmartLayers.SmartModulesManager import SmartModulesManager
from models.utils import load_from_dict, load_ImageNet_validation_set, load_CIFAR10_datasets
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202


class UnknownNetworkException(Exception):
    pass


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--force-reload', action='store_true',
                        help='Force the computation of the output feature map.')
    parser.add_argument('--no-log-results', action='store_true',
                        help='Forbid logging the results of the fault injection campaigns')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault model used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        choices=['ResNet18', 'ResNet50',
                                 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121',
                                 'EfficientNet'])
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='The threshold under which an error is undetected')

    parsed_args = parser.parse_args()

    return parsed_args


def load_network(network_name: str,
                 device: torch.device) -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :return: The loaded network
    """

    if 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50']:
            if network_name == 'ResNet18':
                network_function = resnet18
            elif network_name == 'ResNet50':
                network_function = resnet50
            else:
                raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

            # Load the weights
            network = network_function(weights='DEFAULT')

        else:
            if network_name == 'ResNet20':
                network_function = resnet20
            elif network_name == 'ResNet32':
                network_function = resnet32
            elif network_name == 'ResNet44':
                network_function = resnet44
            elif network_name == 'ResNet56':
                network_function = resnet56
            elif network_name == 'ResNet110':
                network_function = resnet110
            elif network_name == 'ResNet1202':
                network_function = resnet1202
            else:
                raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

            # Instantiate the network
            network = network_function()

            # Load the weights
            network_path = f'models/pretrained_models/{network_name}.th'

            load_from_dict(network=network,
                           device=device,
                           path=network_path)
    elif 'DenseNet' in network_name:
        if network_name == 'DenseNet121':
            network = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of DenseNet: {network_name}')

    elif network_name == 'EfficientNet':
        network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    else:
        raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

    # Send network to device and set for inference
    network.to(device)
    network.eval()

    return network


def get_loader(network_name: str,
               batch_size: int) -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :return: The DataLoader
    """
    if 'ResNet' in network_name and network_name not in ['ResNet18', 'ResNet50']:
        _, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size)
    else:
        loader = load_ImageNet_validation_set(batch_size=batch_size,
                                              image_per_class=1)

    return loader


def get_delayed_start_module(network: Module,
                             network_name: str,
                             fault_delayed_start=False) -> Module:
    """
    Get the delayed_start_module of the given network
    :param network: The instance of the network where to look for the fault_delayed_start module
    :param network_name: The name of the network
    :param fault_delayed_start: Whether fault_delayed_start is enabled or not
    :return: An instance of the delayed_start_module
    """

    # If fault delayed start is enabled, set the module where this function is enabled, otherwise set the module
    # to None
    if fault_delayed_start:
        # The module to change is dependent on the network. This is the module for which to enable delayed start
        if 'ResNet' in network_name:
            delayed_start_module = network
        elif 'DenseNet' in network_name:
            delayed_start_module = network.features
        elif 'EfficientNet' in network_name:
            delayed_start_module = network.features
        else:
            raise UnknownNetworkException
    else:
        delayed_start_module = None

    return delayed_start_module


def get_module_classes(network_name: str) -> Union[List[type], type]:
    """
    Get the module_classes of a given network. The module classes represent the classes that can be replaced by smart
    modules in the network. Notice that the instances of these classes that will be replaced are only the children of
    the delayed_start_module
    :param network_name: The name of the network
    :return: The type of modules (or of a single module) that will should be replaced by smart modules in the target
    network
    """
    if 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50']:
            module_classes = Sequential
        else:
            module_classes = models.resnet.BasicBlock
    elif 'DenseNet' in network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {network_name}')

    return module_classes


def get_fault_list(fault_model: str,
                   fault_list_generator: FaultListGenerator) -> Tuple[Union[List[NeuronFault], List[WeightFault]], List[Module]]:
    """
    Get the fault list corresponding to the specific fault model, using the fault list generator passed as argument
    :param fault_model: The name of the fault model
    :param fault_list_generator: An instance of the fault generator
    :return: A tuple of fault_list, injectable_modules. The latter is a list of all the modules that can be injected in
    case of neuron fault injections
    """
    if fault_model == 'byzantine_neuron':
        fault_list = fault_list_generator.get_neuron_fault_list(load_fault_list=True,
                                                                save_fault_list=True)
        injectable_modules = fault_list_generator.injectable_output_modules_list

    elif fault_model == 'stuck-at_params':
        fault_list = fault_list_generator.get_weight_fault_list(load_fault_list=True,
                                                                save_fault_list=True)
        injectable_modules = None
    else:
        raise ValueError(f'Invalid fault model {fault_model}')

    return fault_list, injectable_modules


def get_device(forbid_cuda: bool,
               use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
        if use_cuda:
            print('WARNING: cuda forcibly disabled even if set_cuda is set')
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = ''
                print('ERROR: cuda not available even if use-cuda is set')
                exit(-1)
        else:
            device = 'cpu'

    return torch.device(device)


def formatted_print(fault_list: list,
                    network_name: str,
                    batch_size: int,
                    batch_id: int,
                    faulty_prediction_dict: dict,
                    fault_dropping: bool = False,
                    fault_delayed_start: bool = False) -> None:
    """
    A function that prints to csv the results of the fault injection campaign on a single batch
    :param fault_list: A list of the faults
    :param network_name: The name of the network
    :param batch_size: The size of the batch of the data loader
    :param batch_id: The id of the batch
    :param faulty_prediction_dict: A dictionary where the key is the fault index and the value is a list of all the
    top_1 prediction for all the image of the given the batch
    :param fault_dropping: Whether fault dropping is used or not
    :param fault_delayed_start: Whether fault delayed start is used or not
    """

    fault_list_rows = [[fault_id,
                       fault.layer_name,
                        fault.tensor_index[0],
                        fault.tensor_index[1] if len(fault.tensor_index) > 1 else np.nan,
                        fault.tensor_index[2] if len(fault.tensor_index) > 2 else np.nan,
                        fault.tensor_index[3] if len(fault.tensor_index) > 3 else np.nan,
                        fault.bit,
                        fault.value
                        ]
                       for fault_id, fault in enumerate(fault_list)
                       ]

    fault_list_columns = [
        'Fault_ID',
        'Fault_Layer',
        'Fault_Index_0',
        'Fault_Index_1',
        'Fault_Index_2',
        'Fault_Index_3',
        'Fault_Bit',
        'Fault_Value'
    ]

    prediction_rows = [
        [
            fault_id,
            batch_id,
            prediction_id,
            prediction[0],
            prediction[1],
        ]
        for fault_id in faulty_prediction_dict for prediction_id, prediction in enumerate(faulty_prediction_dict[fault_id])
    ]

    prediction_columns = [
        'Fault_ID',
        'Batch_ID',
        'Image_ID',
        'Top_1',
        'Top_Score',
    ]

    fault_list_df = pd.DataFrame(fault_list_rows, columns=fault_list_columns)
    prediction_df = pd.DataFrame(prediction_rows, columns=prediction_columns)

    complete_df = fault_list_df.merge(prediction_df, on='Fault_ID')

    file_prefix = 'combined_' if fault_dropping and fault_delayed_start \
        else 'delayed_' if fault_delayed_start \
        else 'dropping_' if fault_dropping \
        else ''

    output_folder = f'output/fault_campaign_results/{network_name}/{batch_size}'
    os.makedirs(output_folder, exist_ok=True)
    complete_df.to_csv(f'{output_folder}/{file_prefix}fault_injection_batch_{batch_id}.csv', index=False)


def enable_optimizations(
        network: Module,
        delayed_start_module: Module,
        module_classes: Union[List[type], type],
        device: torch.device,
        fm_folder: str,
        fault_list_generator: FaultListGenerator,
        fault_list: Union[List[NeuronFault], List[WeightFault]],
        injectable_modules: List[Module] = None,
        fault_delayed_start: bool = True,
        fault_dropping: bool = True):

    # Replace the convolutional layers
    if fault_dropping or fault_delayed_start:

        smart_layers_manager = SmartModulesManager(network=network,
                                                   delayed_start_module=delayed_start_module,
                                                   device=device,
                                                   input_size=torch.Size((1, 3, 32, 32)))

        if fault_delayed_start:
            # Replace the forward module of the target module to enable delayed start
            smart_layers_manager.replace_module_forward()

        # Replace the smart layers of the network
        smart_modules_list = smart_layers_manager.replace_smart_modules(module_classes=module_classes,
                                                                        fm_folder=fm_folder,
                                                                        fault_list=fault_list)

        # Update the network. Useful to update the list of injectable layers when injecting in the neurons
        if injectable_modules is not None:
            fault_list_generator.update_network(network)
            injectable_modules = fault_list_generator.injectable_output_modules_list

        network.eval()
    else:
        smart_modules_list = None

    return injectable_modules, smart_modules_list
