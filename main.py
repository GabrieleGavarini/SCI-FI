import copy
import os
import shutil
import csv
import itertools

import torch

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.utils import load_from_dict, load_CIFAR10_datasets

from models.SmartLayers.SmartLayersManager import SmartLayersManager

from utils import get_device, parse_args, UnknownNetworkException


def main(args):

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True)

    # Select the device
    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    if args.network_name == 'ResNet20':
        network_function = resnet20
    elif args.network_name == 'ResNet32':
        network_function = resnet32
    elif args.network_name == 'ResNet44':
        network_function = resnet44
    elif args.network_name == 'ResNet56':
        network_function = resnet56
    elif args.network_name == 'ResNet110':
        network_function = resnet110
    elif args.network_name == 'ResNet1202':
        network_function = resnet1202
    else:
        network_function = None
        print(f'ERROR: Invalid network name {args.network_name}')
        exit(-1)

    network_path = f'models/pretrained_models/{args.network_name}.th'

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)

    network = network_function()
    network.to(device)
    load_from_dict(network=network,
                   device=device,
                   path=network_path)
    network.eval()

    # Folder containing the feature maps
    fm_folder = f'output/feature_maps/{args.network_name}/batch_{args.batch_size}'
    # Folder containing the clean output
    clean_output_folder = f'output/clean_output/{args.network_name}/batch_{args.batch_size}'

    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=test_loader,
                                           device=device,
                                           fm_folder=fm_folder,
                                           clean_output_folder=clean_output_folder)

    # Create the fm dir if it doesn't exist
    os.makedirs(fm_folder, exist_ok=True)
    # Create the clean output dir if it doesn't exist
    os.makedirs(clean_output_folder, exist_ok=True)

    if args.force_reload:
        # Delete folder if already exists
        shutil.rmtree(fm_folder, ignore_errors=True)

        # Save the intermediate layer
        ofm_manager.save_intermediate_layer_outputs()
    else:
        # Try to load the clean input
        ofm_manager.load_clean_output()

    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       injectable_layer_names=ofm_manager.feature_maps_layer_names,
                                       device=device)

    fault_list = fault_manager.get_weight_fault_list(load_fault_list=True,
                                                     save_fault_list=True)

    for fault_dropping, fault_delayed_start in reversed(list(itertools.product([True, False], repeat=2))):

        if not args.forbid_cuda and args.use_cuda:
            print('Clearing cache')
            torch.cuda.empty_cache()

        # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
        smart_network = copy.deepcopy(network)

        # If fault delayed start is enabled, set the module where this function is enabled, otherwise set the module
        # to None
        if fault_delayed_start:
            # The module to change is dependent on the network. This is the module for which to enable delayed start
            if 'ResNet' in args.network_name:
                delayed_start_module = smart_network
            elif 'DenseNet' in args.network_name:
                delayed_start_module = smart_network.features
            else:
                raise UnknownNetworkException
        else:
            delayed_start_module = None

        # Replace the convolutional layers
        if fault_dropping or fault_delayed_start:

            smart_layers_manager = SmartLayersManager(network=smart_network,
                                                      module=delayed_start_module)

            # Replace the convolutional layers of the network
            smart_convolutions = smart_layers_manager.replace_conv_layers(device=device,
                                                                          fm_folder=fm_folder,
                                                                          threshold=args.threshold)

            if fault_delayed_start:
                # Replace the forward module of the target module to enable delayed start
                smart_layers_manager.replace_module_forward()
        else:
            smart_convolutions = None


        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                         network_name=f'Smart{args.network_name}',
                                                         device=device,
                                                         smart_convolutions=smart_convolutions,
                                                         loader=test_loader,
                                                         clean_output=ofm_manager.clean_output)

        elapsed_time = fault_injection_executor.run_faulty_campaign_on_weight(fault_list=copy.deepcopy(fault_list),
                                                                              fault_dropping=fault_dropping,
                                                                              fault_delayed_start=fault_delayed_start,
                                                                              delayed_start_module=delayed_start_module,
                                                                              first_batch_only=True)

        if not args.no_log_results:
            os.makedirs('log', exist_ok=True)
            log_path = f'log/{args.network_name}.csv'
            with open(log_path, 'a') as file_log:
                writer = csv.writer(file_log)

                # For the first row write the header first
                if os.stat(log_path).st_size == 0:
                    writer.writerow(['Batch Size', 'Fault Dropping', 'Fault Delayed Start', 'Time'])

                # Log the results of the fault injection campaign
                writer.writerow([args.batch_size, fault_dropping, fault_delayed_start, elapsed_time])


if __name__ == '__main__':
    main(args=parse_args())
