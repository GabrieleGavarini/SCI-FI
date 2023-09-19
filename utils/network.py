from typing import Tuple

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import densenet121, densenet161, DenseNet121_Weights, DenseNet161_Weights
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.utils.data import DataLoader

from modules.lenet import LeNet5
from modules.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from modules.utils import load_from_dict, load_CIFAR10_datasets, load_MNIST_datasets, load_ImageNet_validation_set, \
    load_GTSRB_datasets


class UnknownNetworkException(Exception):
    pass


def get_network(network_name: str,
                device: torch.device,
                load_weights_from_dict: bool = True,
                root: str = '.') -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :param load_weights_from_dict: Wheteher to load the weights of the network or not
    :param root: the directory where to look for weights
    :return: The loaded network
    """

    if 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            if network_name == 'ResNet18':
                network_function = resnet18
                weights = ResNet18_Weights.DEFAULT
            if network_name == 'ResNet34':
                network_function = resnet34
                weights = ResNet34_Weights.DEFAULT
            elif network_name == 'ResNet50':
                network_function = resnet50
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                raise UnknownNetworkException(f'ERROR: unknown version of ResNet: {network_name}')

            # Load the weights
            network = network_function(weights=weights)

        elif network_name == 'ResNet50_GTSRB':
            network_function = resnet50
            network = network_function()
            linear_input_features = list(network.named_children())[-1][1].in_features
            network.fc = torch.nn.Linear(in_features=linear_input_features, out_features=43, device=device)
            network_path = f'{root}/modules/pretrained_models/{network_name}.pt'

            if load_weights_from_dict:
                load_from_dict(network=network,
                               device=device,
                               path=network_path)

        else:
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
            network_path = f'{root}/modules/pretrained_models/{network_name}.th'

            if load_weights_from_dict:
                load_from_dict(network=network,
                               device=device,
                               path=network_path)
    elif 'DenseNet' in network_name:
        if network_name == 'DenseNet121':
            network = densenet121(weights=DenseNet121_Weights.DEFAULT)
        if network_name == 'DenseNet161':
            network = densenet161(weights=DenseNet161_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of DenseNet: {network_name}')

    elif 'LeNet5' in network_name:
        network = LeNet5()

        # Load the weights
        network_path = f'{root}/modules/pretrained_models/{network_name}.pt'

        if load_weights_from_dict:
            load_from_dict(network=network,
                           device=device,
                           path=network_path)


    elif 'EfficientNet' in network_name:
        if 'EfficientNet_B0' in network_name:
            network = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        elif 'EfficientNet_B4' in network_name:
            network = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        else:
            raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

        if 'GTSRB' in network_name:
            linear_input_features = list(network.named_children())[-1][1][1].in_features
            network.classifier = torch.nn.Linear(in_features=linear_input_features, out_features=43,
                                                 device=device)

            if load_weights_from_dict:
                network.load_state_dict(torch.load(f'{root}/modules/pretrained_models/EfficientNet_B4_GTSRB'))

    else:
        raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

    # Send network to device and set for inference
    network.to(device)
    network.eval()

    if load_weights_from_dict:
        print('Weights loaded from device')

    return network


def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               network: torch.nn.Module = None,
               get_train_loader:bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :param get_train_loader: Default False. Whether to get also the train loader
    :return: The Train DataLoader and the test DataLoader
    """
    if 'ResNet' in network_name and network_name not in ['ResNet18', 'ResNet50', 'ResNet50_GTSRB']:
        train_loader, _, loader = load_CIFAR10_datasets(test_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
    elif 'LeNet' in network_name:
        train_loader, _, loader = load_MNIST_datasets(test_batch_size=batch_size)
    elif 'GTSRB' in network_name:
        train_loader, _, loader = load_GTSRB_datasets(test_batch_size=batch_size, train_split=.9)
    else:
        if image_per_class is None:
            image_per_class = 5
        loader = load_ImageNet_validation_set(batch_size=batch_size,
                                              image_per_class=image_per_class,
                                              network=network,)
        train_loader = load_ImageNet_validation_set(batch_size=batch_size,
                                                   image_per_class=None,
                                                   network=network,)

    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return train_loader, loader