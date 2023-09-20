import _csv
import os
import argparse

import numpy as np
import pandas as pd

from typing import Union, List, Tuple, TextIO

import torch
from torch.nn import Sequential, Module
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import densenet121, densenet161, DenseNet121_Weights, DenseNet161_Weights
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.utils.data import DataLoader

import modules
from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.WeightFault import WeightFault
from modules.SmartLayers.SmartModulesManager import SmartModulesManager
from modules.utils import load_from_dict, load_ImageNet_validation_set, load_CIFAR10_datasets, load_MNIST_datasets, load_GTSRB_datasets
from modules.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from modules.lenet import LeNet5


class UnknownNetworkException(Exception):
    pass


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ENV
    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='If the use_cuda, specify the device number')
    parser.add_argument('--force-reload', action='store_true',
                        help='Force the computation of the output feature map.')
    parser.add_argument('--no-log-results', action='store_true',
                        help='Forbid logging the results of the fault injection campaigns')
    parser.add_argument('--save-compressed', action='store_true',
                        help='Save OFM as compressed .npz files')

    # NETWORK
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        choices=['LeNet5', 'LeNet5_MNIST',
                                 'ResNet18', 'ResNet50', 'ResNet50_GTSRB',
                                 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121',
                                 'EfficientNet_B0', 'EfficientNet_B4', 'EfficientNet_B4_GTSRB'])
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')

    # FAULT MODEL
    parser.add_argument('--exhaustive', action='store_true',
                        help='Forbid logging the results of the fault injection campaigns')
    parser.add_argument('--bit-wise', action='store_true',
                        help='Inject fault in a bit-wise manner')
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault model used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])

    parser.add_argument('--target_layer_index', type=int, default=None,
                        help='If set together with --target-layer-n injected only n faults in the indexed layer')
    parser.add_argument('--target_layer_n', type=int, default=None,
                        help='If set together with --target-layer-index injected only n faults in the indexed layer')
    parser.add_argument('--target_layer_bit', type=int, default=None,
                        help='If set together with --target-layer-index and --target-layer-n injected only n faults in '
                             'the indexed layer in the specified bit')

    parser.add_argument('--multiple_fault_number', type=int, default=None,
                        help='If the fault model is stuck-at params, how many fault to inject in a single inference in'
                             'absolute number')
    parser.add_argument('--multiple_fault_percentage', type=float, default=None,
                        help='If the fault model is stuck-at params, how many fault to inject in a single inference in'
                             'percentage of the total number of network\'s neurons')


    # SMOOTHING
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='The threshold under which an error is undetected')
    parser.add_argument('--enable-gaussian-filter', action='store_true',
                        help='Apply the gaussian filter to the ofm to decrease fault impact')


    # QUANTIZATION
    parser.add_argument('--quantization-dtype', '-q', type=str,
                        help='The quantization model',
                        choices=['int8', 'int32', 'float16'])

    parsed_args = parser.parse_args()

    # Check that only one between multiple_fault_number and multiple_fault_percentage is set
    if parsed_args.multiple_fault_number is not None and parsed_args.multiple_fault_percentage is not None:
        print('ERROR: only one between multiple_fault_number and multiple_fault_percentage can be set.')

    return parsed_args


def get_network(network_name: str,
                device: torch.device,
                load_weights_from_dict: bool = True,
                root: str = '.') -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :param load_weights_from_dict: Wheteher to load the weights of the network or not
    :param root: the directory where to look for weights
    :return: The loaded network
    """

    if 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            if network_name == 'ResNet18':
                network_function = resnet18
                weights = ResNet18_Weights.DEFAULT
            if network_name == 'ResNet34':
                network_function = resnet34
                weights = ResNet34_Weights.DEFAULT
            elif network_name == 'ResNet50':
                network_function = resnet50
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

            # Load the weights
            network = network_function(weights=weights)

        elif network_name == 'ResNet50_GTSRB':
            network_function = resnet50
            network = network_function()
            linear_input_features = list(network.named_children())[-1][1].in_features
            network.fc = torch.nn.Linear(in_features=linear_input_features, out_features=43, device=device)
            network_path = f'{root}/modules/pretrained_models/{network_name}.pt'

            if load_weights_from_dict:
                load_from_dict(network=network,
                               device=device,
                               path=network_path)

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
            network_path = f'{root}/modules/pretrained_models/{network_name}.th'

            if load_weights_from_dict:
                load_from_dict(network=network,
                               device=device,
                               path=network_path)
    elif 'DenseNet' in network_name:
        if network_name == 'DenseNet121':
            network = densenet121(weights=DenseNet121_Weights.DEFAULT)
        if network_name == 'DenseNet161':
            network = densenet161(weights=DenseNet161_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of DenseNet: {network_name}')

    elif 'LeNet5' in network_name:
        network = LeNet5()

        # Load the weights
        network_path = f'{root}/modules/pretrained_models/{network_name}.pt'

        if load_weights_from_dict:
            load_from_dict(network=network,
                           device=device,
                           path=network_path)


    elif 'EfficientNet' in network_name:
        if 'EfficientNet_B0' in network_name:
            network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        elif 'EfficientNet_B4' in network_name:
            network = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

        if 'GTSRB' in network_name:
            linear_input_features = list(network.named_children())[-1][1][1].in_features
            network.classifier = torch.nn.Linear(in_features=linear_input_features, out_features=43,
                                                 device=device)

            if load_weights_from_dict:
                network.load_state_dict(torch.load(f'{root}/modules/pretrained_models/EfficientNet_B4_GTSRB'))

    else:
        raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

    # Send network to device and set for inference
    network.to(device)
    network.eval()

    if load_weights_from_dict:
        print('Weights loaded from device')

    return network


def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               network: torch.nn.Module = None,
               get_train_loader:bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :param get_train_loader: Default False. Whether to get also the train loader
    :return: The Train DataLoader and the test DataLoader
    """
    if 'ResNet' in network_name and network_name not in ['ResNet18', 'ResNet50', 'ResNet50_GTSRB']:
        train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
    elif 'LeNet' in network_name:
        train_loader, _, loader = load_MNIST_datasets(test_batch_size=batch_size)
    elif 'GTSRB' in network_name:
        train_loader, _, loader = load_GTSRB_datasets(test_batch_size=batch_size, train_split=.9)
    else:
        if image_per_class is None:
            image_per_class = 5
        loader = load_ImageNet_validation_set(batch_size=batch_size,
                                              image_per_class=image_per_class,
                                              network=network,)
        train_loader = load_ImageNet_validation_set(batch_size=batch_size,
                                                   image_per_class=None,
                                                   network=network,)

    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return train_loader, loader

def get_train_validation_loader_by_name(dataset_name: str,
                                        batch_size: int) -> (DataLoader, DataLoader):
    """
    Return the train and validation dataset
    :param dataset_name: The name of the dataset
    :param batch_size: The batch size
    :return: The train and validation ataLoader
    """

    if dataset_name == 'GTSRB':
        train_loader, val_loader, test_loader = load_GTSRB_datasets(train_batch_size=batch_size,
                                                                    train_split=.8)
    elif dataset_name == 'MNIST':
        train_loader, val_loader, test_loader = load_MNIST_datasets()
    else:
        raise AttributeError(f'Unknown dataset {dataset_name}')

    return train_loader, val_loader


def get_delayed_start_module(network: Module,
                             network_name: str) -> Module:
    """
    Get the delayed_start_module of the given network
    :param network: The instance of the network where to look for the fault_delayed_start module
    :param network_name: The name of the network
    :return: An instance of the delayed_start_module
    """

    # The module to change is dependent on the network. This is the module for which to enable delayed start
    if 'LeNet' in network_name:
        delayed_start_module = network
    elif 'ResNet' in network_name:
        delayed_start_module = network
    elif 'DenseNet' in network_name:
        delayed_start_module = network.features
    elif 'EfficientNet' in network_name:
        delayed_start_module = network.features
    else:
        raise UnknownNetworkException

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
    if 'LeNet' in network_name:
        module_classes = Sequential
    elif 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50', 'ResNet50_GTSRB']:
            module_classes = Sequential
        else:
            module_classes = modules.resnet.BasicBlock
    elif 'DenseNet' in network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {network_name}')

    return module_classes


def get_fault_list(fault_model: str,
                   fault_list_generator: FaultListGenerator,
                   exhaustive: bool = False,
                   bit_wise: bool = False,
                   target_layer_index: int = None,
                   target_layer_n: int = None,
                   target_layer_bit: int = None,
                   e: float = .01,
                   t: float = 2.58,
                   multiple_fault_number: int = None,
                   multiple_fault_percentage: float = None,
                   total_neurons: int = None,
                   ) -> Tuple[_csv.reader, TextIO, int, List[Module]]:
    """
    Get the fault list corresponding to the specific fault model, using the fault list generator passed as argument
    :param fault_model: The name of the fault model
    :param fault_list_generator: An instance of the fault generator
    :param exhaustive: Default False. Get an exhaustive instead of a statistic one
    :param bit_wise: Default False. Perform a bit-wise statistical FI to study the criticality of each bit. Only valid
    for faults in the parameters
    :param target_layer_index: Default None. If set together with target_layer_n, inject n faults in the specified
    layer
    :param target_layer_n: Default None. If set together with target_layer_index, inject n faults in the specified
    layer
    :param target_layer_bit: Default None. If set together with target_layer_index and target_layer_n, inject n faults
    in the specified layer in the specified bit position
    :param e: The desired error margin
    :param t: The t related to the desired confidence level
    :param multiple_fault_number: Default None. The number of multiple fault to inject in case of faults in the neurons
    :param multiple_fault_percentage: Default None. The percentage of multiple fault to inject in case of faults in the
     neurons
    :param total_neurons: Default None. The total number of neurons in the network
    :return: A tuple of fault_list_reader, fault_list_file, fault_list_length, injectable_modules. The latter is a list
    of all the modules that can be injected in case of neuron fault injections
    """

    # TODO: implement n and index for neurons faults

    if fault_model == 'byzantine_neuron':

        # Manage possible attribute errors
        if multiple_fault_percentage is not None and multiple_fault_number is not None:
            raise AttributeError('ERROR: Ambiguous fault model specified; both number and percentage of neurons where '
                                 'specified. Set only one.')

        if multiple_fault_percentage is not None and total_neurons is None:
            raise AttributeError('ERROR: Impossible to inject a percentage of neurons when the total number of neurons '
                                 'is not set')

        # Generate the fault list
        fault_list_reader, fault_list_file, fault_list_length = fault_list_generator.get_neuron_fault_list(load_fault_list=True,
                                                                                                           save_fault_list=True,
                                                                                                           exhaustive=exhaustive,
                                                                                                           e=e,
                                                                                                           t=t,
                                                                                                           multiple_fault_number=multiple_fault_number,
                                                                                                           multiple_fault_percentage=multiple_fault_percentage,
                                                                                                           total_neurons=total_neurons)
    elif fault_model == 'stuck-at_params':
        fault_list_reader, fault_list_file, fault_list_length = fault_list_generator.get_weight_fault_list(load_fault_list=True,
                                                                                                           save_fault_list=True,
                                                                                                           exhaustive=exhaustive,
                                                                                                           bit_wise=bit_wise,
                                                                                                           target_layer_index=target_layer_index,
                                                                                                           target_layer_n=target_layer_n,
                                                                                                           target_layer_bit=target_layer_bit,
                                                                                                           e=e,
                                                                                                           t=t)
    else:
        raise ValueError(f'Invalid fault model {fault_model}')

    injectable_modules = fault_list_generator.injectable_output_modules_list

    return fault_list_reader, fault_list_file, fault_list_length, injectable_modules


def get_device(forbid_cuda: bool,
               use_cuda: bool,
               cuda_device:int = 0) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :param cuda_device: Default 0. Specifies the CUDA device if use_cuda is True
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
                device = f'cuda:{cuda_device}'
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
        input_size: torch.Size = torch.Size((1, 3, 32, 32)),
        injectable_modules: List[Module] = None,
        fault_delayed_start: bool = True,
        fault_dropping: bool = True):

    # Replace the convolutional layers
    if fault_dropping or fault_delayed_start:

        smart_layers_manager = SmartModulesManager(network=network,
                                                   delayed_start_module=delayed_start_module,
                                                   device=device,
                                                   input_size=input_size)

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
