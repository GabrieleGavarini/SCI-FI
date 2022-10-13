import copy

import torch.utils.data

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionManager import FaultInjectionManager
from FaultGenerators.FaultListGenerator import FaultListGenerator

from models.resnet import resnet20, resnet32, resnet110
from models.smart_resnet import smart_resnet20, smart_resnet32, smart_resnet110
from models.utils import load_from_dict, load_CIFAR10_datasets

import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fault_dropping', action='store_true', help='Drop fault that lead to no change in the OFM')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Test set batch size')
    parser.add_argument('--network-name', '-n', type=str, help='Target network',
                        choices=['ResNet20', 'ResNet32', 'ResNet110'])

    parsed_args = parser.parse_args()

    return parsed_args


def main(args):

    if args.network_name == 'ResNet20':
        network_function = resnet20
        smart_network_function = smart_resnet20
    elif args.network_name == 'ResNet32':
        network_function = resnet32
        smart_network_function = smart_resnet32
    elif args.net == 'ResNet110':
        network_function = resnet110
        smart_network_function = smart_resnet110
    else:
        network_function = None
        smart_network_function = None
        print(f'ERROR: Invalid network name {args.network_name}')
        exit(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_path = f'models/pretrained_models/{args.network_name}.th'

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=args.batch_size)

    network = network_function()
    network.to(device)
    load_from_dict(network=network,
                   device=device,
                   path=network_path)
    network.eval()


    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=test_loader,
                                           device=device)

    ofm_manager.save_intermediate_layer_outputs()

    # Generate fault list
    fault_manager = FaultListGenerator(network=network,
                                       network_name=args.network_name,
                                       loader=copy.deepcopy(test_loader),
                                       device=device)
    fault_list = fault_manager.get_weight_fault_list(save_fault_list=True)

    # Create the smart network
    smart_network = smart_network_function(ofm_dict=ofm_manager.output_feature_maps_dict,
                                           ifm_dict=ofm_manager.output_feature_maps_dict,
                                           threshold=0.1)
    smart_network.to(device)
    load_from_dict(network=smart_network,
                   device=device,
                   path=network_path)
    smart_network.eval()

    # Execute the fault injection campaign
    fault_injection_executor = FaultInjectionManager(network=network,
                                                     network_name=args.network_name,
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output,
                                                     ofm=ofm_manager.output_feature_maps_dict)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           fault_dropping=False,
                                                           first_batch_only=True)


    fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                     network_name=f'Smart{args.network_name}',
                                                     loader=test_loader,
                                                     clean_output=ofm_manager.clean_output,
                                                     ofm=ofm_manager.output_feature_maps_dict)

    fault_injection_executor.run_faulty_campaign_on_weight(fault_list=fault_list,
                                                           # fault_dropping=args.fault_dropping,
                                                           fault_dropping=True,
                                                           first_batch_only=True)


if __name__ == '__main__':
    main(args=parse_args())
