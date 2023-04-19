import os
import csv
import math
import re

import numpy as np

from tqdm import tqdm
from ast import literal_eval as make_tuple

from typing import Type, List, Tuple

from FaultGenerators.WeightFault import WeightFault
from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.modules.InjectableOutputModule import injectable_output_module_class

from torch.nn import Module
import torch
import torchinfo


class FaultListGenerator:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 module_class: Type[Module] = None,
                 input_size: torch.Size = None):

        self.network = network
        self.network_name = network_name

        self.device = device

        # The class of the injectable modules
        # TODO: extend to multiple module class
        self.module_class = module_class
        self.injectable_module_class = injectable_output_module_class(self.module_class)

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = None

        # Create the list of injectable module if the module_class is set
        if self.module_class is not None:
            self.__replace_injectable_output_modules(input_size=input_size)

        # Name of the injectable layers
        injectable_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                  if isinstance(module, self.module_class)]

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

        # The fault list
        self.fault_list = None

    @staticmethod
    def __compute_date_n(N: int,
                         p: float = 0.5,
                         e: float = 0.01,
                         t: float = 2.58):
        """
        Compute the number of faults to inject according to the DATE09 formula
        :param N: The total number of parameters. If None, compute the infinite population version
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: the number of fault to inject
        """
        if N is None:
            return p * (1-p) * t ** 2 / e ** 2
        else:
            return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))


    @staticmethod
    def __get_list_of_tuples_from_str(string: str,
                                      element_in_tuple: int = 3) -> List[Tuple]:
        """
        Get a list of tuples from a string
        :param string: The string to convert
        :param element_in_tuple: How many elements are in a single tuple
        :return: A list of tuples
        """
        return [make_tuple(match[0]) for match in re.findall(f'(\(([0-9]+(, )?){{{element_in_tuple}}}\))', string)]

    @staticmethod
    def __get_list_from_str(string: str,
                            cast_type: Type = float) -> List:
        """
        Convert a string in a list of elements of type cast_type
        :param string: The string to convert
        :param cast_type: The type to cast the elements
        :return: The list
        """
        return [cast_type(entry) for entry in string.replace('[', '').replace(']', '').split(',')]



    def __replace_injectable_output_modules(self,
                                            input_size: torch.Size):
        """
        Replace the target modules with a version that is injectable
        :param input_size: The size of the input of the network. Used to extract the output shape of each layer
        """

        modules_to_replace = [(name, module) for name, module in self.network.named_modules() if
                              isinstance(module, self.module_class)]

        # Initialize the list of all the injectable layers
        self.injectable_output_modules_list = list()

        # Create a summary of the network
        summary = torchinfo.summary(self.network,
                                    device=self.device,
                                    input_size=input_size,
                                    verbose=False)

        # Extract the output, input and kernel shape of all the convolutional layers of the network
        output_shapes = [torch.Size(info.output_size) for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]
        input_shapes = [torch.Size(info.input_size) for info in summary.summary_list if
                        isinstance(info.module, self.module_class)]
        kernel_shapes = [info.module.weight.shape for info in summary.summary_list if
                         isinstance(info.module, self.module_class)]

        # Replace all layers with injectable convolutional layers
        for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):

            layer_module.__class__ = self.injectable_module_class
            layer_module.init_as_copy(device=self.device,
                                      layer_name=layer_name,
                                      input_shape=input_shapes[layer_id],
                                      output_shape=output_shapes[layer_id],
                                      kernel_shape=kernel_shapes[layer_id])

            # Append the layer to the list
            self.injectable_output_modules_list.append(layer_module)


    def update_network(self,
                       network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules()
                                               if isinstance(module, self.injectable_module_class)]


    def get_neuron_fault_list(self,
                              load_fault_list: bool = False,
                              save_fault_list: bool = True,
                              seed: int = 51195,
                              p: float = 0.5,
                              e: float = 0.01,
                              t: float = 2.58,
                              multiple_fault_number: int = 1):
        """
        Generate a fault list for the neurons according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :param multiple_fault_number: Default 1. How many faults to inject in a single inference
        :return: The fault list
        """

        cwd = os.getcwd()
        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}/multiple_faults_{multiple_fault_number}'

        try:
            if load_fault_list:
                with open(f'{fault_list_filename}/{seed}_neuron_fault_list.csv', newline='') as f_list:
                    reader = csv.reader(f_list)

                    fault_list = list(reader)[1:]

                    fault_list = [NeuronFault(layer_name=str(fault[1]),
                                              layer_index=int(fault[2]),
                                              feature_map_indices=self.__get_list_of_tuples_from_str(fault[3]),
                                              value_list=self.__get_list_from_str(fault[-1])) for fault in fault_list]

                print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
            else:
                raise FileNotFoundError

        except FileNotFoundError:

            # Initialize the random number generator
            random_generator = np.random.default_rng(seed=seed)

            # Compute how many fault can be injected per layer
            possible_faults_per_layer = [injectable_layer.output_shape[1] * injectable_layer.output_shape[2] * injectable_layer.output_shape[3]
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

                    feature_map_indices = list()
                    value_list = list()

                    for _ in range(0, multiple_fault_number):

                        channel = random_generator.integers(injectable_layer.output_shape[1])
                        height = random_generator.integers(injectable_layer.output_shape[2])
                        width = random_generator.integers(injectable_layer.output_shape[3])
                        value = random_generator.random() * 2 - 1

                        feature_map_indices.append((channel, height, width))
                        value_list.append(value)

                    fault_list.append(NeuronFault(layer_name=injectable_layer.layer_name,
                                                  layer_index=layer_index,
                                                  feature_map_indices=feature_map_indices,
                                                  value_list=value_list))

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
                        writer_fault.writerow([index, fault.layer_name, fault.layer_index, fault.feature_map_indices, fault.value_list])


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
            possible_faults_per_layer = [np.prod(injectable_layer.kernel_shape)
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

                    k = random_generator.integers(injectable_layer.kernel_shape[0])
                    dim1 = random_generator.integers(injectable_layer.kernel_shape[1]) if len(injectable_layer.kernel_shape) > 1 else [None]
                    dim2 = random_generator.integers(injectable_layer.kernel_shape[2]) if len(injectable_layer.kernel_shape) > 2 else [None]
                    dim3 = random_generator.integers(injectable_layer.kernel_shape[3]) if len(injectable_layer.kernel_shape) > 3 else [None]
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
