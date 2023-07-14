import os
from tqdm import tqdm
import numpy as np

import torch

from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultInjectionManager import FaultInjectionManager
from OutputFeatureMapsManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from modules.utils import load_CIFAR10_datasets
from plugins.NAS.models import ConvBlock
from utils import parse_args, get_NAS_fine_tuned, get_device, get_fault_list, enable_optimizations

WORKING_DIR = '.'

def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Check CUDA availability
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda,
                        cuda_device=args.cuda_device)

    # Load the nas_name
    nas_model, nas_model_name = get_NAS_fine_tuned(nas_name=args.nas_name,
                                                   cd_size=args.cd_size,
                                                   cd_ops=args.cd_ops,
                                                   device=device)


    # Load the dataset
    _, _, loader = load_CIFAR10_datasets(normalize_input=False,
                                         test_batch_size=args.batch_size)

    # Set up the directories
    fm_folder = f'{WORKING_DIR}/output/feature_maps/{nas_model_name}/batch_{args.batch_size}'
    os.makedirs(fm_folder, exist_ok=True)

    # Folder containing the clean output
    clean_output_folder = f'{WORKING_DIR}/output/clean_output/{nas_model_name}/batch_{args.batch_size}'
    os.makedirs(clean_output_folder, exist_ok=True)

    # Set up the OFM Manager
    ofm_manager = OutputFeatureMapsManager(network=nas_model,
                                           loader=loader,
                                           module_classes=ConvBlock,
                                           device=device,
                                           fm_folder=fm_folder,
                                           clean_output_folder=clean_output_folder,
                                           save_compressed=args.save_compressed)

    # Load the clean input
    ofm_manager.load_clean_output(force_reload=args.force_reload)

    # Generate fault list
    fault_list_generator = FaultListGenerator(network=nas_model,
                                              network_name=nas_model_name,
                                              device=device,
                                              module_class=torch.nn.Conv2d,
                                              input_size=loader.dataset[0][0].unsqueeze(0).shape)

    total_neurons = None
    if args.fault_model == 'byzantine_neuron':
        total_neurons = sum([np.prod(layer.output_shape) for layer in fault_list_generator.injectable_output_modules_list])

    # Manage the fault models
    fault_list, fault_list_file, fault_list_length, injectable_modules = get_fault_list(fault_model=args.fault_model,
                                                                                        fault_list_generator=fault_list_generator,
                                                                                        exhaustive=args.exhaustive,
                                                                                        layer_wise=args.layer_wise,
                                                                                        total_neurons=total_neurons,
                                                                                        e=.01,
                                                                                        t=1.68,
                                                                                        multiple_fault_percentage=args.multiple_fault_percentage,
                                                                                        multiple_fault_number=args.multiple_fault_number)

    delayed_start_module = nas_model

    # Enable fault delayed start and fault dropping
    injectable_modules, smart_modules_list = enable_optimizations(
        network=nas_model,
        delayed_start_module=delayed_start_module,
        module_classes=ConvBlock,
        device=device,
        fm_folder=fm_folder,
        fault_list_generator=fault_list_generator,
        fault_list=fault_list,
        injectable_modules=injectable_modules,
        fault_delayed_start=False,
        fault_dropping=False)

    # Execute the fault injection campaign with the smart network
    fault_injection_executor = FaultInjectionManager(network=nas_model,
                                                     network_name=nas_model_name,
                                                     device=device,
                                                     smart_modules_list=smart_modules_list,
                                                     loader=loader,
                                                     clean_output=ofm_manager.clean_output,
                                                     injectable_modules=injectable_modules,
                                                     layer_wise=args.layer_wise)

    # Run the fault injection campaign
    elapsed_time, avg_memory_occupation = fault_injection_executor.run_fault_injection_campaign(
        fault_model=args.fault_model,
        fault_list=fault_list,
        fault_list_file=fault_list_file,
        fault_list_length=fault_list_length,
        exhaustive=args.exhaustive,
        chunk_size=int(1e5) if args.exhaustive else None,
        fault_dropping=False,
        fault_delayed_start=False,
        delayed_start_module=delayed_start_module,
        golden_ifm_file_extension='npz',
        first_batch_only=False,
        save_output=True,
        multiple_fault_percentage=args.multiple_fault_percentage,
        multiple_fault_number=args.multiple_fault_number)


if __name__ == '__main__':
    main(args=parse_args())
