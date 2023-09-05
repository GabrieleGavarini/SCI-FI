import copy
import os
import csv

import torch
from torch.nn import Conv2d

from OutputFeatureMapsManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator
from masked_analysis.AnalyzableConv2d import AnalyzableConv2d

from utils import parse_args, get_network
from utils import get_device, get_loader, get_module_classes, get_fault_list, get_delayed_start_module
from utils import enable_optimizations


def main(args):
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    # Load the network
    network = get_network(network_name=args.network_name,
                          device=device)

    # Load the dataset
    loader = get_loader(network_name=args.network_name,
                        batch_size=args.batch_size,
                        image_per_class=10)

    # Get the module class for the smart operations
    module_classes = get_module_classes(network_name=args.network_name)

    # get the delayed_start_module
    delayed_start_module = get_delayed_start_module(network=network,
                                                    network_name=args.network_name)

    # Folder containing the feature maps
    fm_folder = f'output/feature_maps/{args.network_name}/batch_{args.batch_size}'
    os.makedirs(fm_folder, exist_ok=True)

    # Folder containing the clean output
    clean_output_folder = f'output/clean_output/{args.network_name}/batch_{args.batch_size}'


    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=loader,
                                           module_classes=module_classes,
                                           device=device,
                                           fm_folder=fm_folder,
                                           clean_output_folder=clean_output_folder)

    # Try to load the clean input
    ofm_manager.load_clean_output(force_reload=args.force_reload)

    # Generate fault list
    fault_list_generator = FaultListGenerator(network=network,
                                              network_name=args.network_name,
                                              device=device,
                                              module_class=torch.nn.Conv2d,
                                              input_size=loader.dataset[0][0].unsqueeze(0).shape)

    fault_dropping = True
    fault_delayed_start = True

    # Manage the fault models
    clean_fault_list, clean_fault_list_file, clean_fault_list_length, injectable_modules = get_fault_list(fault_model=args.fault_model,
                                                                                                          fault_list_generator=fault_list_generator)

    # Create a copy of the fault list, to avoid that consecutive executions create bugs
    fault_list = copy.deepcopy(clean_fault_list)

    if not args.forbid_cuda and args.use_cuda:
        print('Clearing cache')
        torch.cuda.empty_cache()

    # Enable fault delayed start and fault dropping
    injectable_modules, smart_modules_list = enable_optimizations(
        network=network,
        delayed_start_module=delayed_start_module,
        module_classes=module_classes,
        device=device,
        fm_folder=fm_folder,
        fault_list_generator=fault_list_generator,
        fault_list=fault_list,
        input_size=loader.dataset[0][0].unsqueeze(dim=0).shape,
        injectable_modules=injectable_modules,
        fault_delayed_start=fault_delayed_start,
        fault_dropping=fault_dropping)

    # Replace Conv2d with AnalyzableConv2d
    analyzable_module_list = list()
    for module_name, module in network.named_modules():
        if isinstance(module, Conv2d):
            module.__class__ = AnalyzableConv2d
            module.initialize_params(layer_name=module_name,
                                     network_name=args.network_name,
                                     batch_size=args.batch_size,
                                     fault_model=args.fault_model)
            analyzable_module_list.append(module)

    # Execute the fault injection campaign with the smart network
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     device=device,
                                                     smart_modules_list=smart_modules_list,
                                                     loader=loader,
                                                     clean_output=ofm_manager.clean_output,
                                                     injectable_modules=injectable_modules)

    # Clean run to save the output of the conv layers
    fault_injection_executor.run_clean_campaign()
    for analyzable_module in analyzable_module_list:
        analyzable_module.reset_batch()
        analyzable_module.set_faulty_inference()

    elapsed_time, avg_memory_occupation = fault_injection_executor.run_faulty_campaign_on_weight(fault_model=args.fault_model,
                                                                                                 fault_list=fault_list,
                                                                                                 fault_dropping=fault_dropping,
                                                                                                 fault_delayed_start=fault_delayed_start,
                                                                                                 delayed_start_module=delayed_start_module,
                                                                                                 first_batch_only=False,
                                                                                                 save_output=True,
                                                                                                 save_feature_maps_statistics=True)

    if not args.no_log_results:
        os.makedirs('log', exist_ok=True)
        log_path = f'log/{args.network_name}.csv'
        with open(log_path, 'a') as file_log:
            writer = csv.writer(file_log)

            # For the first row write the header first
            if os.stat(log_path).st_size == 0:
                writer.writerow(['Fault Model', 'Batch Size', 'Fault Dropping', 'Fault Delayed Start', 'Time', 'Avg. Memory Occupation'])

            # Log the results of the fault injection campaign
            writer.writerow([args.fault_model, args.batch_size, fault_dropping, fault_delayed_start, elapsed_time, avg_memory_occupation])


if __name__ == '__main__':
    main(args=parse_args())
