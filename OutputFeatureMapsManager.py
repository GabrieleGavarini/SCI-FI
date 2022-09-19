import torch
from torch.nn.modules import Sequential
from torch.utils.data import DataLoader

from tqdm import tqdm


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
        self.batch_size = batch.nelement() * batch.element_size()

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

    def __get_layer_hook(self,
                         layer_name: str):
        """
        Returns a hook function that saves the output feature map of the layer name
        :param layer_name: Name of the layer for which to save the output feature maps
        :return: the hook function to register as a forward hook
        """
        def save_output_feature_map_hook(_, in_tensor, out_tensor):
            if layer_name in self.output_feature_maps_dict.keys():
                self.output_feature_maps_dict[layer_name].append(out_tensor)
            else:
                self.output_feature_maps_dict[layer_name] = [out_tensor]

            self.output_feature_maps_dict_size += out_tensor.nelement() * out_tensor.element_size()

        return save_output_feature_map_hook

    def save_intermediate_layer_outputs(self,
                                        target_layers_names: list = None) -> None:
        """
        Save the intermediate layer outputs of the network for the given dataset, saving the resulting output as a
        dictionary, where each entry corresponds to a layer and contains a list of 4-dimensional tensor NCHW
        :param target_layers_names: Default None. A list of layers name. If None, save the intermediate feature maps of
        all the layers that have at least one parameter. Otherwise, save the output feature map of the specified layers
        """

        print(f'Batch size: {self.loader.batch_size} - Memory occupation: {self.batch_size * 1e-6:.2f} MB')

        # TODO: find a more elegant way to do this
        if target_layers_names is None:
            target_layers_names = [name.replace('.weight', '').replace('.bias', '')
                                   for name, module in self.network.named_parameters()]

        for name, module in self.network.named_modules():
            if name in target_layers_names:
                module.register_forward_hook(self.__get_layer_hook(layer_name=name))

        self.network.eval()
        self.network.to(self.device)

        pbar = tqdm(self.loader, colour='green', desc='Saving Output Feature Maps')

        for batch in pbar:
            data, _ = batch
            data = data.to(self.device)

            _ = self.network(data)

        self.output_feature_maps_size = self.output_feature_maps_dict_size / len(self.loader)

        # How much more space is required to store a batch output feature map when compared with the batched images in
        # percentage
        relative_occupation = 100 * self.output_feature_maps_size / self.batch_size

        print(f'Saved output feature maps')
        print(f'Total occupied memory: {self.output_feature_maps_dict_size * 1e-9:.2f} GB'
              f' - Single batch size: {self.output_feature_maps_size * 1e-6:.2f} MB ({relative_occupation:.2f}%)')
