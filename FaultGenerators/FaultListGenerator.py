import _csv
import os
import sys
import csv
import math
import re

import numpy as np

from tqdm import tqdm
from ast import literal_eval as make_tuple

from typing import Type, List, Tuple, TextIO

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
                 dtype: torch.dtype = torch.float,
                 module_class: List[Type[Module]] = None,
                 input_size: torch.Size = None,
                 avoid_last_lst_fc_layer: bool = False):

        # Set maxsize to load/store large fault lists
        csv.field_size_limit(sys.maxsize)

        self.network = network
        self.network_name = network_name

        self.device = device

        # The class of the injectable modules
        # TODO: extend to multiple module class
        self.module_class = tuple(module_class)
        self.injectable_module_class = tuple(injectable_output_module_class(c) for c in self.module_class)

        # List of injectable modules. Used only for neurons injection
        self.injectable_output_modules_list = list()
        self.avoid_last_lst_fc_layer = avoid_last_lst_fc_layer

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
        self.fault_list_file = None

        # Manage the datatype used for the fault injection
        self.dtype = dtype
        if self.dtype == torch.float:
            self.dtype_bit_width = 32
        elif self.dtype == torch.float16:
            self.dtype_bit_width = 16
        elif self.dtype == torch.qint8:
            self.dtype_bit_width = 8
        elif self.dtype == torch.qint32:
            self.dtype_bit_width = 32
        else:
            raise NotImplementedError(f'Fault injected not implemented for data represented as {self.dtype}')

    @staticmethod
    def __compute_date_n(N: int,
                         p: float = 0.5,
                         e: float = 0.01,
                         t: float = 2.58):
        """
        Compute the number of faults to inject according to the DATE23 formula
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

        for module_class, injectable_module_class in zip(self.module_class, self.injectable_module_class):
            modules_to_replace = [(name, module) for name, module in self.network.named_modules() if
                                  isinstance(module, module_class)]


            # Create a summary of the network
            summary = torchinfo.summary(self.network,
                                        device=self.device,
                                        input_size=input_size,
                                        verbose=False)

            # Extract the output, input and kernel shape of all the convolutional layers of the network
            output_shapes = [torch.Size(info.output_size) for info in summary.summary_list if
                             isinstance(info.module, module_class)]
            input_shapes = [torch.Size(info.input_size) for info in summary.summary_list if
                            isinstance(info.module, module_class)]
            kernel_shapes = [info.module.weight().shape if callable(info.module.weight) else info.module.weight.shape
                             for info in summary.summary_list if isinstance(info.module, module_class)]

            # Replace all layers with injectable convolutional layers
            for layer_id, (layer_name, layer_module) in enumerate(modules_to_replace):

                layer_module.__class__ = injectable_module_class
                layer_module.init_as_copy(device=self.device,
                                          layer_name=layer_name,
                                          input_shape=input_shapes[layer_id],
                                          output_shape=output_shapes[layer_id],
                                          kernel_shape=kernel_shapes[layer_id])

                # Append the layer to the list
                self.injectable_output_modules_list.append(layer_module)

        if self.avoid_last_lst_fc_layer:
            self.injectable_output_modules_list = self.injectable_output_modules_list[:-1]


    def update_network(self,
                       network):
        self.network = network
        self.injectable_output_modules_list = [module for module in self.network.modules()
                                               if isinstance(module, self.injectable_module_class)]

        if self.avoid_last_lst_fc_layer:
            self.injectable_output_modules_list = self.injectable_output_modules_list[:-1]


    def get_neuron_fault_list(self,
                              load_fault_list: bool = False,
                              save_fault_list: bool = True,
                              exhaustive: bool = False,
                              seed: int = 51195,
                              p: float = 0.5,
                              e: float = 0.01,
                              t: float = 2.58,
                              multiple_fault_number: int = None,
                              multiple_fault_percentage: float = None,
                              total_neurons: int = None,):
        """
        Generate a fault list for the neurons according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param exhaustive: Default False. Get an exhaustive instead of a statistic one
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :param multiple_fault_number: Default None. The number of multiple fault to inject in case of faults in the
         neurons
        :param multiple_fault_percentage: Default None. The percentage of multiple fault to inject in case of faults in
        the neurons
        :param total_neurons: Default None. The total number of neurons in the network
        :return: The fault list
        """

        # TODO: fix this function to work like the param fault injection

        if multiple_fault_percentage is not None:

            if exhaustive:
                return NotImplementedError('ERROR: exhaustive injection not supported for multiple neurons')

            multiple_fault_number = math.ceil(total_neurons * multiple_fault_percentage)
            print(f'Injecting {multiple_fault_number} faults for each inference - '
                  f'{multiple_fault_number / total_neurons:.0E}% of {total_neurons} total neurons.')

        cwd = os.getcwd()

        # Set the postfix for the folder
        if multiple_fault_percentage is not None:
            fault_list_postfix = f'/percentage_{multiple_fault_percentage:.0E}'
        elif multiple_fault_number is not None:
            fault_list_postfix = f'/number_{multiple_fault_number}'
        else:
            fault_list_postfix = ''

        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}{fault_list_postfix}'

        # Set the prefix for the file
        if exhaustive:
            fault_list_prefix = 'exhaustive'
        else:
            fault_list_prefix = f'{seed}'

        fault_list_path = f'{fault_list_filename}/{fault_list_prefix}_neuron_fault_list.csv'

        # TODO: rewrite as for params

        try:

            if load_fault_list:
                with open(f'{fault_list_filename}/{fault_list_prefix}_neuron_fault_list.csv', newline='') as f_list:
                    reader = csv.reader(f_list)

                    fault_list = list(reader)[1:]

                    # fault_list = [NeuronFault(layer_name=str(fault[1]),
                    #                           layer_index=int(fault[2]),
                    #                           feature_map_indices=self.__get_list_of_tuples_from_str(fault[3]),
                    #                           value_list=self.__get_list_from_str(fault[-1])) for fault in fault_list]

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
            if exhaustive:
                probability_per_layer = np.ones(len(possible_faults_per_layer))
            else:
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

                    # If not injecting multiple faults, cycle only once
                    if multiple_fault_percentage is None and multiple_fault_number is None:
                        multiple_fault_number = 1

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
                with open(f'{fault_list_filename}/{fault_list_prefix}_neuron_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'LayerName',
                                           'LayerIndex',
                                           'FeatureMapIndex',
                                           'Value'])
                    for index, fault in enumerate(fault_list):
                        writer_fault.writerow([index, fault.layer_name, fault.layer_index, fault.feature_map_indices, fault.value_list])


            print('Fault List Generated')

        fault_list_reader, fault_list_file, fault_list_length = self.__open_as_csv(file_name=fault_list_path)
        print('Fault list loaded from file')

        return fault_list_reader, fault_list_file, fault_list_length


    @staticmethod
    def __open_as_csv(file_name:str,
                      skip_header: bool = True) -> Tuple[_csv.reader, TextIO, int]:
        """
        Open a file and return the csv file handler
        :param file_name: The name of the file to be read
        :param skip_header: Default True. Whether to read the header from the file before returning the handler
        :return: The csv file handler and the length of the file
        """

        # Get the length of the fault list
        with open(file_name, 'r') as file:
            fault_list_length = sum(1 for line in file) - (1 if skip_header else 0)

        # Open the file to return
        file_handler = open(file_name, newline='')
        csv_reader = csv.reader(file_handler)

        # Remove the header
        if skip_header:
            _ = next(csv_reader)

        return csv_reader, file_handler, fault_list_length


    def get_weight_fault_list(self,
                              load_fault_list=False,
                              save_fault_list=True,
                              exhaustive: bool = False,
                              layer_wise: bool = False,
                              bit_wise: bool = False,
                              target_layer_index: int = None,
                              target_layer_n: int = None,
                              target_layer_bit: int = None,
                              seed=51195,
                              p=0.5,
                              e=0.01,
                              t=2.58) -> Tuple[_csv.reader, TextIO, int]:
        """
        Generate a fault list for the weights according to the DATE09 formula
        :param load_fault_list: Default False. Try to load an existing fault list if it exists, otherwise generate it
        :param save_fault_list: Default True. Whether to save the fault list to file
        :param exhaustive: Default False. Get an exhaustive instead of a statistic one
        :param layer_wise: Default False. Perform a layer-wise statistical FI to study the criticality of each layer
        :param bit_wise: Default False. Perform a bit-wise statistical FI to study the criticality of each bit
        :param target_layer_index: Default None. If set together with target_layer_n, inject n faults in the specified
        layer
        :param target_layer_n: Default None. If set together with target_layer_index, inject n faults in the specified
        layer
        :param target_layer_bit: Default None. If set together with target_layer_index and target_layer_n, inject n
        faults in the specified layer in the specified bit position
        :param seed: Default 51195. The seed of the fault list
        :param p: Default 0.5. The probability of a fault
        :param e: Default 0.01. The desired error rate
        :param t: Default 2.58. The desired confidence level
        :return: The fault list reader, the fault list file and the length of the fault list
        """

        assert (target_layer_n is not None and target_layer_index is not None) or (target_layer_n is None and target_layer_index is None)

        # TODO: move method for retrieving fault list from utils to this class
        # TODO: implement layer wise also for non-NAS

        # Set the prefix for the file
        if exhaustive:
            fault_list_prefix = 'exhaustive'
        elif layer_wise:
            fault_list_prefix = f'{seed}_layer_wise'
        elif bit_wise:
            fault_list_prefix = f'{seed}_bit_wise'
        elif target_layer_index is not None and target_layer_n is not None:
            fault_list_prefix = f'{seed}_layer_{target_layer_index}_n_{target_layer_n}'
            if target_layer_bit is not None:
                fault_list_prefix = f'{fault_list_prefix}_bit_{target_layer_bit}'
        else:
            fault_list_prefix = f'{seed}'


        # Set the file names
        cwd = os.getcwd()
        fault_list_filename = f'{cwd}/output/fault_list/{self.network_name}'
        fault_list_path = f'{fault_list_filename}/{fault_list_prefix}_parameters_fault_list.csv'

        try:
            # If specified, try to load the csv from file
            if load_fault_list:
                fault_list_reader, fault_list_file, fault_list_length = self.__open_as_csv(file_name=fault_list_path)
                print('Fault list loaded from file')

            # If you don't have to load the fault list raise the Exception and force the generation
            else:
                raise FileNotFoundError

        except FileNotFoundError:

            # Write the header
            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{fault_list_prefix}_parameters_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'Layer',
                                           'TensorIndex',
                                           'Bit'])

            # Initialize the random number generator
            random_generator = np.random.default_rng(seed=seed)

            # Compute how many fault can be injected per layer
            possible_faults_per_layer = [np.prod(injectable_layer.kernel_shape)
                                         for injectable_layer in self.injectable_output_modules_list]

            # The population of faults
            total_possible_faults = np.sum(possible_faults_per_layer)


            # Compute the number of fault to inject in each layer
            if exhaustive:
                injected_faults_per_layer = possible_faults_per_layer
            elif layer_wise:
                # Compute the number to inject in each layer
                injected_faults_per_layer = [math.ceil(self.__compute_date_n(N=int(faults),
                                                                             p=p,
                                                                             t=t,
                                                                             e=e)) for faults in possible_faults_per_layer]
            elif target_layer_index is not None and target_layer_n is not None:
                injected_faults_per_layer = [0] * len(possible_faults_per_layer)
                injected_faults_per_layer[target_layer_index] = target_layer_n
            else:
                # Compute the total number of fault to inject
                n = self.__compute_date_n(N=int(total_possible_faults),
                                          p=p,
                                          t=t,
                                          e=e)

                # The percentage of fault to inject in each layer
                probability_per_layer = [possible_faults / total_possible_faults
                                         for possible_faults in possible_faults_per_layer]

                # Compute the number of faults to inject in the network
                injected_faults_per_layer = [math.ceil(probability * n) for probability in probability_per_layer]

            # Initialize the progress bar
            pbar = tqdm(zip(injected_faults_per_layer, self.injectable_output_modules_list),
                        desc='Generating fault list',
                        colour='green',
                        total=len(self.injectable_output_modules_list))

            # For each layer, generate the fault list
            with open(f'{fault_list_filename}/{fault_list_prefix}_parameters_fault_list.csv', 'a', newline='') as f_list:
                writer_fault = csv.writer(f_list)
                index = 0

                for layer_index, (n_per_layer, injectable_layer) in enumerate(pbar):

                    print(f"[DEBUG] Layer Index {layer_index}")
                    weight_size = np.prod(injectable_layer.kernel_shape)
                    print(f"[DEBUG] Weight size {weight_size}")
                    print(f"[DEBUG] N per layer {n_per_layer}")

                    # It has been adjusted to avoid that n_per_layer > weight_size and the random_generator enters in exception (the fault list is not generated)
                    if n_per_layer > weight_size:
                        replace_value = True
                    else:
                        replace_value = False

                    layer_fault_positions = random_generator.choice(range(0, weight_size), n_per_layer, replace=replace_value)
                    print(f"[DEBUG] Layer fault positions {layer_fault_positions}")
                    layer_fault_positions.sort()

                    for layer_fault_position in layer_fault_positions:

                        for bit in range(0, self.dtype_bit_width):

                            if not bit_wise:
                                if target_layer_bit is not None:
                                    bit = target_layer_bit
                                else:
                                    bit = random_generator.integers(0, self.dtype_bit_width)

                            fault = WeightFault(layer_name=injectable_layer.layer_name,
                                                tensor_index=layer_fault_position,
                                                bit=bit)

                            # Write the fault
                            if save_fault_list:
                                writer_fault.writerow([index, fault.layer_name, fault.tensor_index, fault.bit])
                                index += 1

                            if not bit_wise:
                                break

                print('Fault List Generated')

            # Read the saved file
            fault_list_reader, fault_list_file, fault_list_length = self.__open_as_csv(file_name=fault_list_path)
            print('Fault list loaded from file')

        return fault_list_reader, fault_list_file, fault_list_length
