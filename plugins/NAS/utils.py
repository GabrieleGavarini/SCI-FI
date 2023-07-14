import _csv
import os
import argparse

import torch

from typing import Tuple, Union, List, TextIO

from plinio.methods import PITSuperNet
from torch.nn import Module

from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.WeightFault import WeightFault
from models import ResNet8PITSN
from modules.SmartLayers.SmartModulesManager import SmartModulesManager

SAVE_DIR = "./"
DATA_DIR = './'

def parse_args():
    """
    Parse the argument of the main
    :return: The parsed arguments
    """

    parser = argparse.ArgumentParser(description='FI on NAS tuned networks')

    # ENV
    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='If the use_cuda, specify the device number')
    parser.add_argument('--save-compressed', action='store_true',
                        help='Save OFM as compressed .npz files')
    parser.add_argument('--force-reload', action='store_true',
                        help='Force the computation of the output feature map.')

    # FAULT MODEL
    parser.add_argument('--exhaustive', action='store_true',
                        help='Forbid logging the results of the fault injection campaigns')
    parser.add_argument('--layer-wise', action='store_true',
                        help='FPerform a layer-wise fault injection campaign')
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault nas_name used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])
    parser.add_argument('--multiple_fault_number', type=int, default=None,
                        help='If the fault nas_name is stuck-at params, how many fault to inject in a single inference in'
                             'absolute number')
    parser.add_argument('--multiple_fault_percentage', type=float, default=None,
                        help='If the fault nas_name is stuck-at params, how many fault to inject in a single inference in'
                             'percentage of the total number of network\'s neurons')

    # NETWORK
    parser.add_argument('--nas-name', '-n', type=str, default="Supernet",
                        choices=['PIT', 'Supernet', 'doublenas_Supernet'],
                        help='PIT, Supernet')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')

    # NAS
    parser.add_argument('--cd-size', type=float, default=0.0001, metavar='CD',
                        help='complexity decay size (default: 0.0)')
    parser.add_argument('--cd-ops', type=float, default=0.0, metavar='CD',
                        help='complexity decay ops (default: 0.0)')
    args = parser.parse_args()

    return args

def get_fault_list(fault_model: str,
                   fault_list_generator: FaultListGenerator,
                   exhaustive: bool = False,
                   layer_wise: bool = False,
                   e: float = .01,
                   t: float = 2.58,
                   multiple_fault_number: int = None,
                   multiple_fault_percentage: float = None,
                   total_neurons: int = None,
                   ) -> Tuple[_csv.reader, TextIO, int, List[Module]]:
    """
    Get the fault list corresponding to the specific fault nas_name, using the fault list generator passed as argument
    :param fault_model: The name of the fault nas_name
    :param fault_list_generator: An instance of the fault generator
    :param exhaustive: Default False. Get an exhaustive instead of a statistic one
    :param layer_wise: Default False. Perform a layer-wise statistical FI to study the criticality of each layer
    :param e: The desired error margin
    :param t: The t related to the desired confidence level
    :param multiple_fault_number: Default None. The number of multiple fault to inject in case of faults in the neurons
    :param multiple_fault_percentage: Default None. The percentage of multiple fault to inject in case of faults in the
     neurons
    :param total_neurons: Default None. The total number of neurons in the network
    :return: A tuple of fault_list_reader, fault_list_file, fault_list_length, injectable_modules. The latter is a list
    of all the modules that can be injected in case of neuron fault injections
    """
    if fault_model == 'byzantine_neuron':

        # Manage possible attribute errors
        if multiple_fault_percentage is not None and multiple_fault_number is not None:
            raise AttributeError('ERROR: Ambiguous fault nas_name specified; both number and percentage of neurons where '
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
                                                                                                           layer_wise=layer_wise,
                                                                                                           e=e,
                                                                                                           t=t)
    else:
        raise ValueError(f'Invalid fault nas_name {fault_model}')

    injectable_modules = fault_list_generator.injectable_output_modules_list

    return fault_list_reader, fault_list_file, fault_list_length, injectable_modules

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


def load(model, path, device):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_NAS_fine_tuned(nas_name: str,
                       cd_size: float,
                       cd_ops: float,
                       device: torch.device) -> Tuple[torch.nn.Module, str]:
    """
    Load a network fine-tuned over a specific dataset
    :param cd_size: The cd_size of the NAS
    :param cd_ops: The cd_ops of the NAS
    :param nas_name: The name of the NAS model to load
    :param device: The device where to load the network
    :return: The loaded network and the name of the fine-tuned model
    """
    data_dir = DATA_DIR

    input_shape = [3, 32, 32]
    PITSuperNet.get_size_binarized = PITSuperNet.get_size
    PITSuperNet.get_macs_binarized = PITSuperNet.get_macs
    nas_model = ResNet8PITSN(gumbel=True).to(device)
    nas_model = PITSuperNet(nas_model, input_shape=input_shape, autoconvert_layers=False).to(device)

    log = 0
    for file in os.listdir(SAVE_DIR + "search_checkpoints"):
        if f"ck_icl_opt_{nas_name}_{cd_size}_{cd_ops}_" in file:
            log = str(int(file.split(".")[-2].split("_")[-1])).zfill(3)

    file_name = f'ck_icl_opt_{nas_name}_{cd_size}_{cd_ops}_{log}'

    nas_model = load(nas_model,
                     SAVE_DIR + f'search_checkpoints/{file_name}.pt',
                     device=device)

    print("final nas_name size:", nas_model.get_size_binarized())
    print("final nas_name MACs:", nas_model.get_macs_binarized())
    # Convert pit nas_name into pytorch nas_name
    exported_model = nas_model.arch_export()
    exported_model = exported_model.to(device)
    log = 0
    for file in os.listdir(SAVE_DIR + "finetuning_checkpoints"):
        if f"ck_icl_opt_{nas_name}_{cd_size}_{cd_ops}_" in file:
            log = str(int(file.split(".")[-2].split("_")[-1])).zfill(3)

    file_name = f'ck_icl_opt_{nas_name}_{cd_size}_{cd_ops}_{log}'

    exported_model = load(exported_model,
                          SAVE_DIR + f'finetuning_checkpoints/{file_name}.pt',
                          device=device)

    return exported_model, file_name