import copy
import os
import shutil

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.smart_resnet import smart_resnet20, smart_resnet32, smart_resnet44, smart_resnet56, smart_resnet110, smart_resnet1202
from models.utils import load_from_dict, load_CIFAR10_datasets

from utils import get_device, parse_args


def main(args):

    device = get_device(forbid_cuda=args.forbid_cuda,
                        use_cuda=args.use_cuda)
    print(f'Using device {device}')

    if args.network_name == 'ResNet20':
        network_function = resnet20
        smart_network_function = smart_resnet20
    elif args.network_name == 'ResNet32':
        network_function = resnet32
        smart_network_function = smart_resnet32
    elif args.network_name == 'ResNet44':
        network_function = resnet44
        smart_network_function = smart_resnet44
    elif args.network_name == 'ResNet56':
        network_function = resnet56
        smart_network_function = smart_resnet56
    elif args.network_name == 'ResNet110':
        network_function = resnet110
        smart_network_function = smart_resnet110
    elif args.network_name == 'ResNet1202':
        network_function = resnet1202
        smart_network_function = smart_resnet1202
    else:
        network_function = None
        smart_network_function = None
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
    fm_folder = f'feature_maps/{args.network_name}/batch_{args.batch_size}'

    # Delete folder if already exists
    # TODO: make this an option
    shutil.rmtree(fm_folder, ignore_errors=True)
    # Create the dir if it doesn't exist
    os.makedirs(fm_folder, exist_ok=True)
    ofm_paths = [f'./{fm_folder}/ofm_batch_{i}' for i in range(0, len(test_loader))]
    ifm_paths = [f'./{fm_folder}/ifm_batch_{i}' for i in range(0, len(test_loader))]

    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=test_loader,
                                           device=device,
                                           ofm_paths=ofm_paths,
                                           ifm_paths=ifm_paths)

    ofm_manager.save_intermediate_layer_outputs()
    injectable_layers = ofm_manager.feature_maps_layer_names

    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       loader=copy.deepcopy(test_loader),
                                       device=device)
    fault_list = fault_manager.get_weight_fault_list(save_fault_list=True)


    # Create the smart network
    smart_network = smart_network_function(ofm_dict_paths=ofm_paths,
                                           ifm_dict_paths=ifm_paths,
                                           threshold=0.1)
    smart_network.to(device)
    load_from_dict(network=smart_network,
                   device=device,
                   path=network_path)
    smart_network.eval()

    # Execute the fault injection campaign
    fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                     network_name=f'Smart{args.network_name}',
                                                     injectable_layers=injectable_layers,
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           # fault_dropping=args.fault_dropping,
                                                           fault_dropping=True,
                                                           first_batch_only=True)

    # Compare with results from non-smart network
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     injectable_layers=injectable_layers,
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           fault_dropping=False,
                                                           first_batch_only=True)


if __name__ == '__main__':
    main(args=parse_args())
