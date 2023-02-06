import copy
import os
import shutil
import csv

import torch
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import MBConv, Conv2dNormActivation
from torch.nn import Conv2d

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator
from masked_analysis.AnalyzableConv2d import AnalyzableConv2d
from models.resnet import BasicBlock

from models.utils import load_ImageNet_validation_set, load_CIFAR10_datasets

from models.SmartLayers.SmartLayersManager import SmartLayersManager

from utils import load_network, get_device, parse_args, UnknownNetworkException


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
    if 'ResNet' in args.network_name:
        _, _, loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)
    else:
        loader = load_ImageNet_validation_set(batch_size=args.batch_size,
                                              image_per_class=1)

    # Folder containing the feature maps
    fm_folder = f'output/feature_maps/{args.network_name}/batch_{args.batch_size}'
    os.makedirs(fm_folder, exist_ok=True)

    # Folder containing the clean output
    clean_output_folder = f'output/clean_output/{args.network_name}/batch_{args.batch_size}'

    # Se the module class for the smart operations
    if 'ResNet' in args.network_name:
        module_classes = BasicBlock
    elif 'DenseNet' in args.network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in args.network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {args.network_name}')

    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=loader,
                                           module_classes=module_classes,
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

        # Create the fm dir if it doesn't exist
        os.makedirs(fm_folder, exist_ok=True)
        # Create the clean output dir if it doesn't exist
        os.makedirs(clean_output_folder, exist_ok=True)

        # Save the intermediate layer
        ofm_manager.save_intermediate_layer_outputs()
    else:
        # Try to load the clean input
        ofm_manager.load_clean_output()


    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       device=device,
                                       module_class=torch.nn.Conv2d,
                                       input_size=loader.dataset[0][0].unsqueeze(0).shape)
    fault_dropping = True
    fault_delayed_start = True
    
    # Manage the fault models
    if args.fault_model == 'byzantine_neuron':
        clean_fault_list = fault_manager.get_neuron_fault_list(load_fault_list=True,
                                                               save_fault_list=True)
        injectable_modules = fault_manager.injectable_output_modules_list
    elif args.fault_model == 'stuckat_params':
        clean_fault_list = fault_manager.get_weight_fault_list(load_fault_list=True,
                                                               save_fault_list=True)
        injectable_modules = None
    else:
        raise ValueError(f'Invalid fault model {args.fault_model}')

    # Create a copy of the fault list, to avoid that consecutive executions create bugs
    fault_list = copy.deepcopy(clean_fault_list)

    # If fault delayed start is enabled, set the module where this function is enabled, otherwise set the module
    # to None
    if fault_delayed_start:
        # The module to change is dependent on the network. This is the module for which to enable delayed start
        if 'ResNet' in args.network_name:
            delayed_start_module = network
        elif 'DenseNet' in args.network_name:
            delayed_start_module = network.features
        elif 'EfficientNet' in args.network_name:
            delayed_start_module = network.features
        else:
            raise UnknownNetworkException
    else:
        delayed_start_module = None

    # Replace the convolutional layers
    if fault_dropping or fault_delayed_start:

        smart_layers_manager = SmartLayersManager(network=network,
                                                  delayed_start_module=delayed_start_module,
                                                  device=device,
                                                  input_size=torch.Size((1, 3, 32, 32)))

        if fault_delayed_start:
            # Replace the forward module of the target module to enable delayed start
            smart_layers_manager.replace_module_forward()

        # Replace the smart layers of the network
        smart_modules_list = smart_layers_manager.replace_smart_modules(module_classes=module_classes,
                                                                        fm_folder=fm_folder,
                                                                        threshold=args.threshold,
                                                                        fault_list=fault_list)

        # Update the network. Useful to update the list of injectable layers when injecting in the neurons
        if injectable_modules is not None:
            fault_manager.update_network(network)
            injectable_modules = fault_manager.injectable_output_modules_list

        network.eval()
    else:
        smart_modules_list = None

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
