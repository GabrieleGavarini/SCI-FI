import os

import torch
from torch.nn import Conv2d
import numpy as np

from torchmetrics.functional import peak_signal_noise_ratio


class AnalyzableConv2d(Conv2d):

    def __init__(self, module):
        super(AnalyzableConv2d, self).__init__()

        self.__module = module

        self.clean_output = None
        self.clean_inference = True
        self.batch_id = None
        self.fault_id = None

        self.network_name = None
        self.layer_name = None
        self.batch_size = None

        self.fault_model = None

        self.fault_analysis = dict()

        self.output_dir = None

        self.initialize_params(layer_name=None,
                               network_name=None,
                               batch_size=None,
                               fault_model=None)


    def initialize_params(self,
                          layer_name,
                          network_name,
                          batch_size,
                          fault_model):

        self.clean_output = list()
        self.clean_inference = True
        self.batch_id = 0
        self.fault_id = None

        self.layer_name = layer_name
        self.network_name = network_name
        self.batch_size = batch_size

        self.fault_model = fault_model

        self.initialize_fault_analysis_dict()


    def initialize_fault_analysis_dict(self):
        self.fault_analysis = {
            'layer_name': list(),
            'fault_id': list(),
            'PSNR': list(),
            'euclidean_distance': list(),
            'max_diff': list(),
            'avg_diff': list(),
            'num_diff_percentage': list()
        }


    def forward(self, input_tensor):

        output_tensor = super().forward(input_tensor)

        if self.clean_inference:
            self.clean_output.append(output_tensor.detach().cpu())
            self.batch_id += 1
        else:

            clean_tensor = self.clean_output[self.batch_id].cuda()

            # Compute the similarity metrics
            data_range = clean_tensor.max() - clean_tensor.min()
            psnr = peak_signal_noise_ratio(preds=output_tensor,
                                           target=clean_tensor,
                                           reduction=None,
                                           dim=(1, 2, 3),
                                           data_range=data_range).cpu()

            euclidean_distance = ((clean_tensor.flatten(start_dim=1) - output_tensor.flatten(start_dim=1))**2).sum(axis=1).cpu()

            # Compute the difference metric
            difference = torch.abs(output_tensor - clean_tensor)
            max_diff = torch.amax(difference, dim=(1, 2, 3)).cpu()
            avg_diff = torch.mean(difference, dim=(1, 2, 3)).cpu()
            num_diff_percentage = torch.not_equal(difference, torch.zeros(difference.shape, device='cuda')).sum(dim=(1, 2, 3)).cpu() / np.prod(difference.shape[1:])

            # Append results to dict
            self.fault_analysis['layer_name'].append(self.layer_name)
            self.fault_analysis['fault_id'].append(self.fault_id)
            self.fault_analysis['PSNR'].append(psnr)
            self.fault_analysis['euclidean_distance'].append(euclidean_distance)
            self.fault_analysis['max_diff'].append(max_diff)
            self.fault_analysis['avg_diff'].append(avg_diff)
            self.fault_analysis['num_diff_percentage'].append(num_diff_percentage)

        return output_tensor

    def save_to_file(self):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(f'{self.output_dir}/{self.layer_name}_{self.batch_id}', self.fault_analysis)
        self.fault_analysis = list()

    def increase_batch(self):
        self.batch_id += 1

    def reset_batch(self):
        self.batch_id = 0

    def set_clean_inference(self):
        self.clean_inference = True

    def set_faulty_inference(self):
        self.clean_inference = False
