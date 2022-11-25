import pickle

import torch
from torch import Tensor
from torch.nn import Module, Conv2d

from models.SmartLayers.utils import check_difference


class SmartConv2d(Module):

    def __init__(self,
                 conv_layer: Conv2d,
                 device: torch.device,
                 input_size: torch.Size,
                 output_size: torch.Size,
                 kernel_size: torch.Size,
                 layer_name: str,
                 fm_folder: str,
                 threshold: float = 0) -> None:

        super(SmartConv2d, self).__init__()

        # The masked convolutional layer
        self.__conv_layer = conv_layer

        # The device used for inference
        self.__device = device

        # Size of input, output and kernel tensors
        self.__output_size = output_size
        self.__input_size = input_size
        self.__kernel_size = kernel_size

        # The id of the batch currently used for inference
        self.__batch_id = None

        # The name of the layer
        self.layer_name = layer_name

        # The path of the folder where the output and input feature map file (containing the tensors) are located
        self.__fm_folder = fm_folder

        # The golden input/output of the layer
        self.__golden_ifm = None
        self.__golden_ofm = None

        # Whether the output of this layer should be compared with the golden output
        self.__compare_ofm_with_golden = False

        # The threshold under which a fault ha no impact
        self.__threshold = threshold


    def get_golden_ifm(self):
        return self.__golden_ifm


    def load_golden(self,
                    batch_id: int) -> None:
        """
        Load the golden output feature map from disk, store it into GPU or CPU memory
        :param batch_id: The index of the batch currently used for inference
        """

        self.__batch_id = batch_id

        # Name of the ifm file
        ifm_file_name = f'{self.__fm_folder}/ifm_batch_{self.__batch_id}_layer_{self.layer_name}.pt'

        # Load the golden ifm
        with open(ifm_file_name, 'rb') as ifm_file:
            self.__golden_ifm = pickle.load(ifm_file).to(self.__device)


    def compare_with_golden(self) -> None:
        """
        Mark the layer as comparable, so that the faulty output is compared with the golden output at run time
        """
        # Mark the layer as comparable
        self.__compare_ofm_with_golden = True


    def do_not_compare_with_golden(self) -> None:
        """
        Mark the layer as non-comparable
        """
        self.__compare_ofm_with_golden = False

    def unload_golden_ofm(self) -> None:
        """
        Delete all the stored golden ofm
        """
        if self.__golden_ofm is not None:
            del self.__golden_ofm
            self.__golden_ofm = None


    def forward(self,
                input_tensor: Tensor) -> Tensor:
        """
        Mask of the actual forward function
        :param input_tensor: The input tensor of the layer
        :return: The output tensor of the layer
        """

        # Check for difference with the golden input, if the layer is marked
        check_difference(check_control=self.__compare_ofm_with_golden,
                         golden=self.__golden_ifm,
                         faulty=input_tensor,
                         threshold=self.__threshold)

        # Compute convolutional output
        conv_output = self.__conv_layer(input_tensor)

        return conv_output
