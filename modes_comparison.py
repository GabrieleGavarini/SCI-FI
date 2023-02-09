import copy
import os
import shutil
import csv
import itertools

import torch

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from utils import load_network, get_device, parse_args, UnknownNetworkException, get_loader, get_module_classes, \
    get_delayed_start_module, enable_optimizations, get_fault_list


def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    # Load the network
    network = load_network(network_name=args.network_name,
                           device=device)

    # Load the dataset
    loader = get_loader(network_name=args.network_name,
                        batch_size=args.batch_size)

    # Folder containing the feature maps
    fm_folder = f'output/feature_maps/{args.network_name}/batch_{args.batch_size}'
    os.makedirs(fm_folder, exist_ok=True)

    # Folder containing the clean output
    clean_output_folder = f'output/clean_output/{args.network_name}/batch_{args.batch_size}'

    # Se the module class for the smart operations
    module_classes = get_module_classes(network_name=args.network_name)

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

    for fault_dropping, fault_delayed_start in reversed(list(itertools.product([True, False], repeat=2))):

        # ----- DEBUG ----- #

        # Skip all FI with delayed start
        # if fault_delayed_start:`
        #     continue

        # Skip all FI with fault dropping
        # if fault_dropping:
        #     continue

        # Skip all FI without delayed start
        # if not fault_delayed_start:
        #     continue

        # Skip all FI without fault dropping
        # if not fault_dropping:
        #     continue

        # Only fully optimized FI
        # if not (fault_delayed_start and fault_dropping):
        #     continue

        # Only unoptimized FI
        # if fault_delayed_start and fault_dropping:
        #     continue

        # Only unoptimized or fully optimized FI
        if not (not (fault_delayed_start and fault_dropping)) or (fault_delayed_start and fault_dropping):
            continue

        # ----- DEBUG ----- #

        # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
        smart_network = copy.deepcopy(network)
        fault_list_generator.update_network(smart_network)

        # Manage the fault models
        clean_fault_list, injectable_modules = get_fault_list(fault_model=args.fault_model,
                                                              fault_list_generator=fault_list_generator)

        # Create a copy of the fault list, to avoid that consecutive executions create bugs
        fault_list = copy.deepcopy(clean_fault_list)

        if not args.forbid_cuda and args.use_cuda:
            print('Clearing cache')
            torch.cuda.empty_cache()

        delayed_start_module = get_delayed_start_module(network=smart_network,
                                                        network_name=args.network_name,
                                                        fault_delayed_start=fault_delayed_start)

        # Enable fault delayed start and fault dropping
        injectable_modules, smart_modules_list = enable_optimizations(
            network=smart_network,
            delayed_start_module=delayed_start_module,
            module_classes=module_classes,
            device=device,
            fm_folder=fm_folder,
            fault_list_generator=fault_list_generator,
            fault_list=fault_list,
            injectable_modules=injectable_modules,
            fault_delayed_start=fault_delayed_start,
            fault_dropping=fault_dropping)


        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                         network_name=args.network_name,
                                                         device=device,
                                                         smart_modules_list=smart_modules_list,
                                                         loader=loader,
                                                         clean_output=ofm_manager.clean_output,
                                                         injectable_modules=injectable_modules)

        elapsed_time, avg_memory_occupation = fault_injection_executor.run_faulty_campaign_on_weight(fault_model=args.fault_model,
                                                                                                     fault_list=fault_list,
                                                                                                     fault_dropping=fault_dropping,
                                                                                                     fault_delayed_start=fault_delayed_start,
                                                                                                     delayed_start_module=delayed_start_module,
                                                                                                     first_batch_only=True,
                                                                                                     save_output=True)

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
