import itertools
import os
import csv
import math
import copy
import numpy as np

from functools import reduce
from tqdm import tqdm
from ast import literal_eval as make_tuple

from typing import Type, List

from FaultGenerators.WeightFault import WeightFault
from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.modules.InjectableOutputModule import InjectableOutputModule

from torch.nn import Module, Conv2d
import torch
import torchinfo


class FaultListGenerator:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 module_classes: Type[Module] = None,
                 input_size: torch.Size = None):

        self.network = network
        self.network_name = network_name

        self.device = device

        # The class of the injectable modules
        self.module_classes = module_classes

        # Create the list of injectable module if the module_class is set
        if self.module_classes is not None:
            self.__replace_injectable_output_modules(input_size=input_size)

        # Name of the injectable layers
        injectable_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                  if isinstance(module, module_classes)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

        # The fault list
        self.fault_list = None

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = None

    @staticmethod
    def __compute_date_n(N: int,
                         p: float = 0.5,
                         e: float = 0.01,
                         t: float = 2.58):
        """
        Compute the number of faults to inject according to the DATE09 formula
        :param N: The total number of parameters
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: the number of fault to inject
        """
        return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))


    def __replace_injectable_output_modules(self,
                                            input_size: torch.Size):
        """
        Replace the target modules with a version that is injectable
        :param input_size: The size of the input of the network. Used to extract the output shape of each layer
        """

        modules_to_replace = [(name, copy.deepcopy(module)) for name, module in self.network.named_modules() if
                              isinstance(module, self.module_classes)]

        # Initialize the list of all the injectable layers
        self.injectable_output_modules_list = list()

        # Create a summary of the network
        summary = torchinfo.summary(self.network,
                                    device=self.device,
                                    input_size=input_size,
                                    verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_sizes = [torch.Size(info.output_size) for info in summary.summary_list if
                        isinstance(info.module, self.module_classes)]
        input_sizes = [torch.Size(info.input_size) for info in summary.summary_list if
                       isinstance(info.module, self.module_classes)]
        kernel_sizes = [info.module.weight.shape for info in summary.summary_list if
                        isinstance(info.module, self.module_classes)]

        # Replace all layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):
            # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
            # ResNet) first separate the layer names using the '.'
            formatted_names = layer_name.split(sep='.')

            # If there are more than one names as a result of the separation, the Module containing the convolutional layer
            # is a nested Module
            if len(formatted_names) > 1:
                # In this case, access the nested layer iteratively using itertools.reduce and getattr
                container_layer = reduce(getattr, formatted_names[:-1], self.network)
            else:
                # Otherwise, the containing layer is the network itself (no nested blocks)
                container_layer = self.network

            # Create the injectable version of the convolutional layer
            injectable_output_module = InjectableOutputModule(clean_module=layer_module,
                                                              device=self.device,
                                                              layer_name=layer_name,
                                                              input_size=input_sizes[layer_id],
                                                              output_size=output_sizes[layer_id],
                                                              kernel_size=kernel_sizes[layer_id])

            # Append the layer to the list
            self.injectable_output_modules_list.append(injectable_output_module)

            # Change the convolutional layer to its injectable counterpart
            setattr(container_layer, formatted_names[-1], injectable_output_module)


    def update_network(self,
                       network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules() if isinstance(module, InjectableOutputModule)]


    def get_neuron_fault_list(self,
                              load_fault_list: bool = False,
                              save_fault_list: bool = True,
                              seed: int = 51195,
                              p: float = 0.5,
                              e: float = 0.01,
                              t: float = 2.58):
        """
        Generate a fault list for the neurons according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list
        """

        cwd = os.getcwd()
        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}'

        try:
            if load_fault_list:
                with open(f'{fault_list_filename}/{seed}_neuron_fault_list.csv', newline='') as f_list:
                    reader = csv.reader(f_list)

                    fault_list = list(reader)[1:]

                    fault_list = [NeuronFault(layer_name=str(fault[1]),
                                              layer_index=int(fault[2]),
                                              feature_map_index=make_tuple(fault[3]),
                                              value=float(fault[-1])) for fault in fault_list]

                print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
            else:
                raise FileNotFoundError

        except FileNotFoundError:

            # Initialize the random number generator
            random_generator = np.random.default_rng(seed=seed)

            # Compute how many fault can be injected per layer
            possible_faults_per_layer = [injectable_layer.output_size[1] * injectable_layer.output_size[2] * injectable_layer.output_size[3]
                                         for injectable_layer in self.injectable_output_modules_list]

            # The population of faults
            total_possible_faults = np.sum(possible_faults_per_layer)

            # The percentage of fault to inject in each layer
            probability_per_layer = [possible_faults / total_possible_faults for possible_faults in possible_faults_per_layer]

            # Compute the total number of fault to inject
            n = self.__compute_date_n(N=int(total_possible_faults),
                                      p=p,
                                      t=t,
                                      e=e)

            # Compute the number of fault to inject in each layer
            injected_faults_per_layer = [math.ceil(probability * n) for probability in probability_per_layer]

            fault_list = list()

            pbar = tqdm(zip(injected_faults_per_layer, self.injectable_output_modules_list),
                        desc='Generating fault list',
                        colour='green')

            # For each layer, generate the fault list
            for layer_index, (n_per_layer, injectable_layer) in enumerate(pbar):
                for i in range(n_per_layer):

                    channel = random_generator.integers(injectable_layer.output_size[1])
                    height = random_generator.integers(injectable_layer.output_size[2])
                    width = random_generator.integers(injectable_layer.output_size[3])
                    value = random_generator.random() * 2 - 1

                    fault_list.append(NeuronFault(layer_name=injectable_layer.layer_name,
                                                  layer_index=layer_index,
                                                  feature_map_index=(channel, height, width),
                                                  value=value))

            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{seed}_neuron_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'LayerName',
                                           'LayerIndex',
                                           'FeatureMapIndex',
                                           'Value'])
                    for index, fault in enumerate(fault_list):
                        writer_fault.writerow([index, fault.layer_name, fault.layer_index, fault.feature_map_index, fault.value])


            print('Fault List Generated')

        self.fault_list = fault_list
        return fault_list


    def get_weight_fault_list(self,
                              load_fault_list=False,
                              save_fault_list=True,
                              seed=51195,
                              p=0.5,
                              e=0.01,
                              t=2.58):
        """
        Generate a fault list for the weights according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list
        """

        cwd = os.getcwd()
        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}'

        try:
            if load_fault_list:
                with open(f'{fault_list_filename}/{seed}_parameters_fault_list.csv', newline='') as f_list:
                    reader = csv.reader(f_list)

                    fault_list = list(reader)[1:]

                    fault_list = [WeightFault(layer_name=fault[1],
                                              tensor_index=make_tuple(fault[2]),
                                              bit=int(fault[-1])) for fault in fault_list]

                print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
            else:
                raise FileNotFoundError

        except FileNotFoundError:
            # Initialize the random number generator
            random_generator = np.random.default_rng(seed=seed)

            # Compute how many fault can be injected per layer
            possible_faults_per_layer = [np.prod(injectable_layer.kernel_size)
                                         for injectable_layer in self.injectable_output_modules_list]

            # The population of faults
            total_possible_faults = np.sum(possible_faults_per_layer)

            # The percentage of fault to inject in each layer
            probability_per_layer = [possible_faults / total_possible_faults
                                     for possible_faults in possible_faults_per_layer]

            # Compute the total number of fault to inject
            n = self.__compute_date_n(N=int(total_possible_faults),
                                      p=p,
                                      t=t,
                                      e=e)

            # Compute the number of fault to inject in each layer
            injected_faults_per_layer = [math.ceil(probability * n) for probability in probability_per_layer]

            # Initialize the fault list
            fault_list = list()

            # Initialize the progress bar
            pbar = tqdm(zip(injected_faults_per_layer, self.injectable_output_modules_list),
                        desc='Generating fault list',
                        colour='green')

            # For each layer, generate the fault list
            for layer_index, (n_per_layer, injectable_layer) in enumerate(pbar):
                for i in range(n_per_layer):

                    k = random_generator.integers(injectable_layer.kernel_size[0])
                    dim1 = random_generator.integers(injectable_layer.kernel_size[1]) if len(injectable_layer.kernel_size) > 1 else [None]
                    dim2 = random_generator.integers(injectable_layer.kernel_size[2]) if len(injectable_layer.kernel_size) > 2 else [None]
                    dim3 = random_generator.integers(injectable_layer.kernel_size[3]) if len(injectable_layer.kernel_size) > 3 else [None]
                    bits = random_generator.integers(0, 32)

                    fault_list.append(WeightFault(layer_name=injectable_layer.layer_name,
                                                  tensor_index=(k, dim1, dim2, dim3),
                                                  bit=bits))

            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{seed}_parameters_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'Layer',
                                           'TensorIndex',
                                           'Bit'])
                    for index, fault in enumerate(fault_list):
                        writer_fault.writerow([index, fault.layer_name, fault.tensor_index, fault.bit])

            print('Fault List Generated')

        self.fault_list = fault_list
        return fault_list
