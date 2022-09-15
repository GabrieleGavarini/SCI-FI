from torch.nn.modules import Sequential
from torch.utils.data import DataLoader


class OutputFeatureMapsManager:

    def __init__(self,
                 network: Sequential,
                 loader: DataLoader):
        """
        Manges the recording of output feature maps for a given network on a given database
        :param network: The network to analyze
        :param loader: the data loader for which to save the output feature maps
        """

        self.network = network
        self.loader = loader

        # a dictionary of lists of tensor. Each entry in the dictionary represent a layer: the key is the layer name
        # while the value is a list contains a list of 4-dimensional tensor, where each tensor is the output feature map
        # for a bath of input images
        self.output_feature_maps_dict = dict()

    def __get_layer_hook(self, layer_name):
        """
        Returns a hook function that saves the output feature map of the layer name
        :param layer_name: Name of the layer for which to save the output feature maps
        :return:
        """
        def save_output_feature_map_hook(module, input, output):
            if layer_name in self.output_feature_maps_dict.keys():
                self.output_feature_maps_dict[layer_name].append(output)
            else:
                self.output_feature_maps_dict[layer_name] = [output]

        return save_output_feature_map_hook

    def save_intermediate_layer_outputs(self):
        """
        Save the intermediate layer outputs of the network for the given dataset, saving the resulting output as a
        dictionary, where each entry corresponds to a layer and contains a list of 4-dimensional tensor NCHW
        """

        for name, module in self.network.named_modules():
            # TODO: check if it is ok to check only non sequential modules
            if isinstance(module, Sequential):
                continue
            else:
                module.register_module_forward_hook(self.__get_layer_hook(layer_name=name))

        self.network.eval()

        for batch in self.loader:
            data, label = batch
