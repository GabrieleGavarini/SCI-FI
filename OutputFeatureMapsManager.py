import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import pickle


class OutputFeatureMapsManager:

    def __init__(self,
                 network: Module,
                 loader: DataLoader,
                 device: torch.device,
                 ofm_paths: list = None,
                 ifm_paths: list = None):
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

        # A list of all the possible layers. This list is equivalent to all the keys from __output_feature_maps
        self.feature_maps_layer_names = list()
        self.feature_maps_layers = list()

        # A string that contains the path where to save the output feature maps
        self.ifm_paths = ifm_paths
        self.ofm_paths = ofm_paths

        # A dictionary of tensors containing the value of the output feature maps of all the layers for the current batch.
        # The dictionary is organized in the following way: the key is the layer name while the value is a 4-dimensional
        # tensor, representing the output feature map for the specific batch and layer
        self.__output_feature_maps = dict()
        # The equivalent dictionary for input feature maps
        self.__input_feature_maps = dict()

        # An integer indicating the number of bytes occupied by the Output Feature Maps (without taking into account
        # the overhead required by the lists and the dictionary)
        self.__output_feature_maps_size = 0
        # The equivalent value for input feature maps
        self.__input_feature_maps_size = 0

        # An integer that measures the size of one single batch of Output Feature Maps (without taking into account the
        # overhead required by the dict
        self.output_feature_maps_size = 0
        # The equivalent value for input feature maps
        self.input_feature_maps_size = 0

        # List containing all the registered forward hooks
        self.hooks = list()

        # Tensor containing all the output of all the batches
        self.clean_output = None

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
            # Move input and output feature maps to main memory and detach
            input_to_save = in_tensor[0].detach().cpu() if save_to_cpu else in_tensor[0].detach()
            output_to_save = out_tensor.detach().cpu() if save_to_cpu else out_tensor.detach()

            # Save the input feature map
            self.__input_feature_maps[layer_name] = input_to_save

            # Save the output feature map
            self.__output_feature_maps[layer_name] = output_to_save

            # Update information about the memory occupation
            self.__input_feature_maps_size += input_to_save.nelement() * input_to_save.element_size()
            self.__output_feature_maps_size += output_to_save.nelement() * output_to_save.element_size()

        return save_output_feature_map_hook


    def __remove_all_hooks(self) -> None:
        """
        Remove all the forward hooks on the network
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()


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
              f'Number of batches: {len(self.loader)}\n'
              f'\tInput Memory occupation: {self.batch_memory_occupation * len(self.loader) * 1e-6:.2f} MB'
              f' - Single batch size: {self.batch_memory_occupation * 1e-6:.2f} MB')
        print(f'\tWeight Memory occupation:'
              f'\t{self.parameters_memory_occupation * 1e-6:.2f} MB')
        print(f'\tTotal Memory occupation:'
              f'\t{(self.parameters_memory_occupation + self.batch_memory_occupation) * 1e-6:.2f} MB')

        # TODO: find a more elegant way to do this
        if target_layers_names is None:
            # target_layers_names = [name.replace('.weight', '') for name, module in self.network.named_parameters()
            #                        if 'weight' in name]
            target_layers_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                   if isinstance(module, torch.nn.Conv2d)]

        self.feature_maps_layers = [module for name, module in self.network.named_modules()
                                    if name.replace('.weight', '') in target_layers_names]

        self.feature_maps_layer_names = target_layers_names
        self.network.eval()
        self.network.to(self.device)

        pbar = tqdm(self.loader, colour='green', desc='Saving Output Feature Maps')

        clean_output_batch_list = list()
        with torch.no_grad():
            for batch_id, batch in enumerate(pbar):
                data, _ = batch
                data = data.to(self.device)

                # Register hooks for current batch
                for name, module in self.network.named_modules():
                    if name in target_layers_names:
                        self.hooks.append(module.register_forward_hook(self.__get_layer_hook(layer_name=name,
                                                                                             save_to_cpu=save_to_cpu)))

                # Execute the network and save the clean output
                clean_output_batch = self.network(data)
                clean_output_batch_list.append(clean_output_batch)

                # Save the output of the batch to file
                with open(self.ofm_paths[batch_id], 'wb') as ofm_file:
                    pickle.dump(self.__output_feature_maps, ofm_file)
                # Clear the dictionary of the current batch
                self.__output_feature_maps = dict()

                # Save the inputs of the batch to file
                with open(self.ifm_paths[batch_id], 'wb') as ifm_file:
                    pickle.dump(self.__input_feature_maps, ifm_file)
                # Clear the dictionary of the current batch
                self.__input_feature_maps = dict()

                # Remove all the hooks
                self.__remove_all_hooks()


        self.clean_output = clean_output_batch_list

        self.input_feature_maps_size = self.__input_feature_maps_size / len(self.loader)
        self.output_feature_maps_size = self.__output_feature_maps_size / len(self.loader)

        # How much more space is required to store a batch output feature map when compared with the batched images in
        # percentage
        input_relative_occupation = 100 * (self.input_feature_maps_size + self.output_feature_maps_size) / self.batch_memory_occupation
        total_occupation = (self.parameters_memory_occupation + self.batch_memory_occupation)
        total_relative_occupation = 100 * (self.input_feature_maps_size + self.output_feature_maps_size) / total_occupation

        print(f'Saved output feature maps')
        print(f'Total occupied memory: {self.__output_feature_maps_size * 1e-9:.2f} GB'
              f' - Single batch size: {self.output_feature_maps_size * 1e-6:.2f} MB')
        print(f'\tRelative Input Overhead:'
              f'\t{input_relative_occupation:.2f}%')
        print(f'\tRelative Total Overhead:'
              f'\t{total_relative_occupation:.2f}%')
