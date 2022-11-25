import copy
from functools import reduce
import types

from typing import List

import torchinfo
import torch
from torch.nn import Module
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.SmartLayers.SmartConv2d import SmartConv2d


def smart_forward(self, x):
    # Execute the layers iteratively, starting from the one where the fault is injected
    layer_index = self.layers.index(self.starting_layer)
    x = self.starting_convolutional_layer.get_golden_ifm()
    for layer in self.layers[layer_index:]:
        x = layer(x)


    return x


def replace_network_forward(network: Module) -> None:
    """
    Replace the network forward function with a smart version
    :param network: The network to modify
    """
    # Add the starting layer attribute
    network.starting_layer = None
    network.starting_convolutional_layer = None

    # Replace with the smart network function
    network.forward = types.MethodType(smart_forward, network)


def replace_conv_layers(network: Module,
                        device: torch.device,
                        fm_folder: str,
                        threshold: float = 0,
                        input_size: torch.Size = torch.Size((1, 3, 32, 32))) -> List[SmartConv2d]:
    """
    Replace all the convolutional layers of the network with injectable convolutional layers
    :param network: The network where to apply the substitution
    :param device: The device where the network is loaded
    :param fm_folder: The folder containing the input and output feature maps
    :param threshold: The threshold under which a folder has no impact
    :param input_size: torch.Size((32, 32, 3)). The torch.Size of an input image
    :return A list of all the new InjectableConv2d
    """

    # Find a list of all the convolutional layers
    convolutional_layers = [(name, copy.deepcopy(module)) for name, module in network.named_modules() if isinstance(module, torch.nn.Conv2d)]


    # Create a summary of the network
    summary = torchinfo.summary(network,
                                device=device,
                                input_size=input_size,
                                verbose=False)

    # Extract the output, input and kernel shape of all the convolutional layers of the network
    output_sizes = [torch.Size(info.output_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]
    input_sizes = [torch.Size(info.input_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]
    kernel_sizes = [torch.Size(info.kernel_size) for info in summary.summary_list if isinstance(info.module, torch.nn.Conv2d)]

    # Initialize the list of all the injectable layers
    injectable_layers = list()

    # Replace all convolution layers with injectable convolutional layers
    for layer_id, (layer_name, layer_module) in enumerate(convolutional_layers):
        # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
        # ResNet, first separate the layer names using the '.'
        formatted_names = layer_name.split(sep='.')

        # If there are more than one names as a result of the separation, the Module containing the convolutional layer
        # is a nester Module
        if len(formatted_names) > 1:
            # In this case, access the nested layer iteratively using itertools.reduce and getattr
            container_layer = reduce(getattr, formatted_names[:-1], network)
        else:
            # Otherwise, the containing layer is the network itself (no nested blocks)
            container_layer = network


        # Create the injectable version of the convolutional layer
        faulty_convolutional_layer = SmartConv2d(conv_layer=layer_module,
                                                 device=device,
                                                 layer_name=layer_name,
                                                 input_size=input_sizes[layer_id],
                                                 output_size=output_sizes[layer_id],
                                                 kernel_size=kernel_sizes[layer_id],
                                                 fm_folder=fm_folder,
                                                 threshold=threshold)

        # Append the layer to the list
        injectable_layers.append(faulty_convolutional_layer)

        # Change the convolutional layer to its injectable counterpart
        setattr(container_layer, formatted_names[-1], faulty_convolutional_layer)

    network.generate_layer_list()

    return injectable_layers


def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10('weights/files/',
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10('weights/files/',
                           train=False,
                           transform=transform_test,
                           download=True)

    # If only a number of images is required per class, modify the test set
    if test_image_per_class is not None:
        image_tensors = list()
        label_tensors = list()
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                image_tensors.append(test_image[0])
                label_tensors.append(test_image[1])
                image_class_counter[test_image[1]] += 1
        test_dataset = TensorDataset(torch.stack(image_tensors),
                                     torch.tensor(label_tensors))

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=train_batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader


def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
    else:
        state_dict = torch.load(path, map_location=device)

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
