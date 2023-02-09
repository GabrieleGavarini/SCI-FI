import os

import torch
from torch.nn import Conv2d
import numpy as np


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

        self.fault_analysis = None

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

        self.fault_analysis = list()


    def forward(self, input_tensor):

        output_tensor = super().forward(input_tensor)

        if self.clean_inference:
            self.clean_output.append(output_tensor.detach().cpu())
            self.batch_id += 1
        else:
            difference = torch.abs(output_tensor - self.clean_output[self.batch_id].cuda())
            result_dict = {
                'layer_name': self.layer_name,
                'fault_id': self.fault_id,
                'max_diff': torch.amax(difference, dim=(1, 2, 3)).cpu(),
                'avg_diff': torch.mean(difference, dim=(1, 2, 3)).cpu(),
                'num_diff_percentage': torch.not_equal(difference, torch.zeros(difference.shape, device='cuda')).sum(dim=(1, 2, 3)).cpu() / np.prod(difference.shape[1:])
            }
            self.fault_analysis.append(result_dict)

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
