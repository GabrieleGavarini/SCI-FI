import torch
from torch.nn.modules import Sequential
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np


class OutputFeatureMapsManager:

    def __init__(self,
                 network: Sequential,
                 loader: DataLoader,
                 device: torch.device):
        """
        Manges the recording of output feature maps for a given network on a given database
        :param network: The network to analyze
        :param loader: the data loader for which to save the output feature maps
        """

        self.network = network
        self.loader = loader
        self.device = device

        # An integer that measures the size of a single batch in memory
        batch = next(iter(self.loader))[0]
        self.batch_memory_occupation = batch.nelement() * batch.element_size()

        # An integer that measures the size weights in memory
        parameters_memory_occupation_list = [parameter.nelement() * parameter.element_size()
                                             for name, parameter in self.network.named_parameters()]
        self.parameters_memory_occupation = np.sum(parameters_memory_occupation_list)

        # A dictionary of lists of tensor. Each entry in the dictionary represent a layer: the key is the layer name
        # while the value is a list contains a list of 4-dimensional tensor, where each tensor is the output feature map
        # for a bath of input images
        self.output_feature_maps_dict = dict()

        # An integer indicating the number of bytes occupied by the Output Feature Maps (without taking into account
        # the overhead required by the lists and the dictionary)
        self.output_feature_maps_dict_size = 0

        # An integer that measures the size of one single batch of Output Feature Maps (without taking into account the
        # overhead required by the dict
        self.output_feature_maps_size = 0

        # List containing all the registered forward hooks
        self.hooks = list()

    def __get_layer_hook(self,
                         layer_name: str,
                         save_to_cpu: bool):
        """
        Returns a hook function that saves the output feature map of the layer name
        :param layer_name: Name of the layer for which to save the output feature maps
        :param save_to_cpu: Default True. Whether to save the output feature maps to cpu or not
        :return: the hook function to register as a forward hook
        """
        def save_output_feature_map_hook(_, in_tensor, out_tensor):
            output_to_save = out_tensor.detach().cpu() if save_to_cpu else out_tensor.detach()

            if layer_name in self.output_feature_maps_dict.keys():
                self.output_feature_maps_dict[layer_name].append(output_to_save)
            else:
                self.output_feature_maps_dict[layer_name] = [output_to_save]

            self.output_feature_maps_dict_size += output_to_save.nelement() * output_to_save.element_size()

        return save_output_feature_map_hook

    def save_intermediate_layer_outputs(self,
                                        target_layers_names: list = None,
                                        save_to_cpu: bool = True) -> None:
        """
        Save the intermediate layer outputs of the network for the given dataset, saving the resulting output as a
        dictionary, where each entry corresponds to a layer and contains a list of 4-dimensional tensor NCHW
        :param target_layers_names: Default None. A list of layers name. If None, save the intermediate feature maps of
        all the layers that have at least one parameter. Otherwise, save the output feature map of the specified layers
        :param save_to_cpu: Default True. Whether to save the output feature maps to cpu or not
        """

        print(f'Batch size: {self.loader.batch_size}\n'
              f'\tInput Memory occupation: {self.batch_memory_occupation * len(self.loader) * 1e-6:.2f} MB'
              f' - Single batch size: {self.batch_memory_occupation * 1e-6:.2f} MB')
        print(f'\tWeight Memory occupation:'
              f'\t{self.parameters_memory_occupation * 1e-6:.2f} MB')
        print(f'\tTotal Memory occupation:'
              f'\t{(self.parameters_memory_occupation + self.batch_memory_occupation) * 1e-6:.2f} MB')

        # TODO: find a more elegant way to do this
        if target_layers_names is None:
            target_layers_names = [name.replace('.weight', '').replace('.bias', '')
                                   for name, module in self.network.named_parameters()]

        for name, module in self.network.named_modules():
            if name in target_layers_names:
                self.hooks.append(module.register_forward_hook(self.__get_layer_hook(layer_name=name,
                                                                                     save_to_cpu=save_to_cpu)))

        self.network.eval()
        self.network.to(self.device)

        pbar = tqdm(self.loader, colour='green', desc='Saving Output Feature Maps')

        with torch.no_grad():
            for batch in pbar:
                data, _ = batch
                data = data.to(self.device)

                _ = self.network(data)

        self.output_feature_maps_size = self.output_feature_maps_dict_size / len(self.loader)

        # How much more space is required to store a batch output feature map when compared with the batched images in
        # percentage
        input_relative_occupation = 100 * self.output_feature_maps_size / self.batch_memory_occupation
        total_occupation = (self.parameters_memory_occupation + self.batch_memory_occupation)
        total_relative_occupation = 100 * self.output_feature_maps_size / total_occupation

        print(f'Saved output feature maps')
        print(f'Total occupied memory: {self.output_feature_maps_dict_size * 1e-9:.2f} GB'
              f' - Single batch size: {self.output_feature_maps_size * 1e-6:.2f} MB')
        print(f'\tRelative Input Overhead:'
              f'\t{input_relative_occupation:.2f}%')
        print(f'\tRelative Total Overhead:'
              f'\t{total_relative_occupation:.2f}%')
