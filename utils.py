import os
import argparse

import numpy as np
import pandas as pd

import torch

from models.utils import load_from_dict
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights


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
                        choices=['byzantine_neuron', 'stuckat_params'])
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
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
