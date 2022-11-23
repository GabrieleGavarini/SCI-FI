import itertools
import os
import csv
from tqdm import tqdm
import numpy as np
from ast import literal_eval as make_tuple

from FaultGenerators.WeightFault import WeightFault


class FaultListGenerator:

    def __init__(self,
                 network,
                 network_name,
                 injectable_layer_names,
                 device):
        
        self.network = network
        self.network_name = network_name

        self.device = device

        # List of the shape of all the layers that contain weight
        self.net_layer_shape = {name.replace('.weight', ''): param.shape for name, param in self.network.named_parameters()
                                if name.replace('.weight', '') in injectable_layer_names}

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
                with open(f'{fault_list_filename}/{seed}_fault_list.csv', newline='') as f_list:
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

            exhaustive_fault_list = []
            pbar = tqdm(self.net_layer_shape.items(), desc='Generating fault list', colour='green')
            for layer_name, layer_shape in pbar:

                # TODO: move here the selection of number of faults per layer
                # Add all the possible faults to the fault list
                k = np.arange(layer_shape[0])
                dim1 = np.arange(layer_shape[1]) if len(layer_shape) > 1 else [None]
                dim2 = np.arange(layer_shape[2]) if len(layer_shape) > 2 else [None]
                dim3 = np.arange(layer_shape[3]) if len(layer_shape) > 3 else [None]
                bits = np.arange(0, 32)

                exhaustive_fault_list = exhaustive_fault_list + list(
                    itertools.product(*[[layer_name], k, dim1, dim2, dim3, bits]))

            random_generator = np.random.default_rng(seed=seed)
            n = self.__compute_date_n(N=len(exhaustive_fault_list),
                                      p=p,
                                      e=e,
                                      t=t)
            fault_list = random_generator.choice(exhaustive_fault_list, int(n))
            del exhaustive_fault_list
            fault_list = [WeightFault(layer_name=fault[0],
                                      tensor_index=tuple([int(i) for i in fault[1:-1]if i is not None]),
                                      bit=int(fault[-1])) for fault in fault_list]

            if save_fault_list:
                os.makedirs(fault_list_filename, exist_ok=True)
                with open(f'{fault_list_filename}/{seed}_fault_list.csv', 'w', newline='') as f_list:
                    writer_fault = csv.writer(f_list)
                    writer_fault.writerow(['Injection',
                                           'Layer',
                                           'TensorIndex',
                                           'Bit'])
                    for index, fault in enumerate(fault_list):
                        writer_fault.writerow([index, fault.layer_name, fault.tensor_index, fault.bit])

            print('Fault List Generated')

        return fault_list
