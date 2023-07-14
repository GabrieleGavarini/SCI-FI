import os

import torch
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping

from plugins.Quantized.Quantization.utils import evaluate

class PTQNetworkManager:

    def __init__(self,
                 network: torch.nn.Module,
                 network_name: str,
                 quantization_type: torch.dtype,
                 quantization_name: str,
                 calibration_loader: DataLoader,
                 device: torch.device):
        """
        A wrapper that inserts a network inside a Post Training Static Quantization
        :param network: The network to be quantized
        :param network_name: The name of the network
        :param quantization_type: The data type used for the quantization
        :param quantization_name: The name of the quantized datatype
        :param calibration_loader: The dataloader used for the calibration of the quantized network
        :param device: The device where the network is loaded
        """
        self.network = network
        self.quantized_network = None

        self.network_name = network_name
        self.quantization_name = quantization_name

        self.quantization_type = quantization_type
        self.calibration_loader = calibration_loader

        self.device = device

        self.__config_quantization()


    def __config_quantization(self) -> None:
        """
        Configure the quantization
        4294967296
        """

        if self.quantization_type == torch.qint8:
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.HistogramObserver.with_args(reduce_range=True),
                weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8))
        elif self.quantization_type == torch.qint32:
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.HistogramObserver.with_args(reduce_range=True),
                weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint32))
        elif self.quantization_type == torch.float16:
            qconfig = torch.ao.quantization.float16_dynamic_qconfig
        else:
            raise AttributeError(f'FI not supported for {self.quantization_type}')

        self.non_quantized_accuracy = evaluate(network=self.network,
                                               loader=self.calibration_loader,
                                               device=self.device,
                                               desc='Test the non quantized model accuracy')

        # Create the mapping and add the correct config
        qconfig_mapping = QConfigMapping().set_global(qconfig)

        # Prepare the model
        prepared_network = prepare_fx(self.network,
                                      qconfig_mapping,
                                      self.calibration_loader.dataset[0])

        # Calibrate the model
        evaluate(prepared_network,
                 loader=self.calibration_loader,
                 device=self.device,
                 desc='Model calibration')

        # Quantize the model
        self.quantized_network =  convert_fx(prepared_network)

        # Test the quantized model accuracy
        self.quantized_accuracy = evaluate(network=self.quantized_network,
                                           loader=self.calibration_loader,
                                           device=self.device,
                                           desc='Testing the quantized model accuracy')

        output_folder = f'../../modules/pretrained_models'
        os.makedirs(output_folder, exist_ok=True)
        torch.save(self.quantized_network.state_dict(), f'{output_folder}/{self.network_name}_{self.quantization_type}.pt')