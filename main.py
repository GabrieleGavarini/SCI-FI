import copy
import os
import shutil

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.utils import load_from_dict, load_CIFAR10_datasets, replace_conv_layers

from utils import get_device, parse_args


def main(args):

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
        ofm_manager.load_clean_output()

    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       injectable_layer_names=ofm_manager.feature_maps_layer_names,
                                       device=device)

    fault_list = fault_manager.get_weight_fault_list(load_fault_list=True,
                                                     save_fault_list=True)

    # ----- DEBUG ----- #
    fault_list = [f for f in fault_list if f.bit == 30]
    # ----- DEBUG ----- #

    # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
    smart_network = copy.deepcopy(network)

    print(f'Fault dropping on {args.network_name} (threshold: {args.threshold})')

    # Replace the convolutional layers
    smart_convolutions = replace_conv_layers(network=smart_network,
                                             device=device,
                                             fm_folder=fm_folder,
                                             threshold=args.threshold)

    # Execute the fault injection campaign with the smart network
    fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                     network_name=f'Smart{args.network_name}',
                                                     device=device,
                                                     smart_convolutions=smart_convolutions,
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           fault_dropping=True,
                                                           first_batch_only=True)

    # Execute the fault injection campaign with the "dumb" network
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     device=device,
                                                     smart_convolutions=smart_convolutions,
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           fault_dropping=False,
                                                           first_batch_only=True)


if __name__ == '__main__':
    main(args=parse_args())
