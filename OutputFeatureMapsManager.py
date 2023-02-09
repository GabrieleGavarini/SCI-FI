import os
import shutil

import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from typing import Tuple, Type
import pickle


class OutputFeatureMapsManager:

    def __init__(self,
                 network: Module,
                 loader: DataLoader,
                 module_classes: Tuple[Type[Module]] or Type[Module],
                 device: torch.device,
                 fm_folder: str,
                 clean_output_folder: str):
        """
        Manges the recording of output feature maps for a given network on a given database
        :param network: The network to analyze
        :param loader: the data loader for which to save the output feature maps
        :param module_classes: The class (or tuple of classes) of the module for which to save the feature maps
        :param device: The device where to perform the inference
        :param clean_output_folder: The folder where to load/store the clean output of the network
        :param fm_folder: The folder containing the input and output feature maps
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
        self.feature_maps_layer_names = [name.replace('.weight', '') for name, module in self.network.named_modules()
                                         if isinstance(module, module_classes)]
        self.feature_maps_layers = [module for name, module in self.network.named_modules()
                                    if name.replace('.weight', '') in self.feature_maps_layer_names]

        # A list of dictionary where every element is the file containing the output feature map for a batch and for the
        # layer
        self.__fm_folder = fm_folder
        os.makedirs(self.__fm_folder, exist_ok=True)
        self.ifm_paths = [{j: f'./{fm_folder}/ifm_batch_{i}_layer_{j}.pt' for j in self.feature_maps_layer_names} for i in range(0, len(loader))]

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

        # Name of the file where to save the clean output
        self.__clean_output_folder = clean_output_folder
        os.makedirs(self.__clean_output_folder, exist_ok=True)
        self.__clean_output_path = f'{clean_output_folder}/clean_output.pt'

    def __get_layer_hook(self,
                         batch_id: int,
                         layer_name: str,
                         save_to_cpu: bool):
        """
        Returns a hook function that saves the output feature map of the layer name
        :param batch_id: The index of the current batch
        :param layer_name: Name of the layer for which to save the output feature maps
        :param save_to_cpu: Default True. Whether to save the output feature maps to cpu or not
        :return: the hook function to register as a forward hook
        """
        def save_output_feature_map_hook(_, in_tensor, out_tensor):
            # Move input and output feature maps to main memory and detach
            input_to_save = in_tensor[0].detach().cpu() if save_to_cpu else in_tensor[0].detach()
            output_to_save = out_tensor.detach().cpu() if save_to_cpu else out_tensor.detach()

            # Save the input feature map
            with open(self.ifm_paths[batch_id][layer_name], 'wb') as ifm_file:
                pickle.dump(input_to_save, ifm_file)

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
                                        save_to_cpu: bool = True) -> None:
        """
        Save the intermediate layer outputs of the network for the given dataset, saving the resulting output as a
        dictionary, where each entry corresponds to a layer and contains a list of 4-dimensional tensor NCHW
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

        self.network.eval()
        self.network.to(self.device)

        pbar = tqdm(self.loader, colour='green', desc='Saving Feature Maps')

        clean_output_batch_list = list()
        with torch.no_grad():
            for batch_id, batch in enumerate(pbar):
                data, _ = batch
                data = data.to(self.device)

                # Register hooks for current batch
                for name, module in self.network.named_modules():
                    if name in self.feature_maps_layer_names:
                        self.hooks.append(module.register_forward_hook(self.__get_layer_hook(batch_id=batch_id,
                                                                                             layer_name=name,
                                                                                             save_to_cpu=save_to_cpu)))

                # Execute the network and save the clean output
                clean_output_batch = self.network(data)
                clean_output_batch_list.append(clean_output_batch)

                # Remove all the hooks
                self.__remove_all_hooks()

        self.clean_output = clean_output_batch_list
        # Save the clean output to file
        pickle.dump(self.clean_output, open(self.__clean_output_path, 'wb'))

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

    def load_clean_output(self,
                          force_reload: bool = False) -> None:
        """
        Load the clean output of the network. If the file is not found, compute the clean output (and the clean output
        feature maps)
        :param force_reload: Whether to force the computation of the clean output
        """
        if force_reload:
            # Delete folders if they already exists
            shutil.rmtree(self.__fm_folder, ignore_errors=True)
            shutil.rmtree(self.__clean_output_folder, ignore_errors=True)

            # Create the fm and clean output dir
            os.makedirs(self.__fm_folder, exist_ok=True)
            os.makedirs(self.__clean_output_folder, exist_ok=True)

            # Save the intermediate layer
            self.save_intermediate_layer_outputs()

        else:
            try:
                self.clean_output = [tensor.to(self.device) for tensor in pickle.load(open(self.__clean_output_path, 'rb'))]
            except FileNotFoundError:
                print('No previous clean output found, starting clean inference...')
                self.save_intermediate_layer_outputs()
