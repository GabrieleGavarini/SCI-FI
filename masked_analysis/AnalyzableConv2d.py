import os

import torch
from torch.nn import Conv2d, Conv1d
import numpy as np

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


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

        self.enable_gaussian_filter = False
        self.gaussian_filter = None

        self.initialize_params(layer_name=None,
                               network_name=None,
                               batch_size=None,
                               fault_model=None)


    def initialize_params(self,
                          layer_name,
                          network_name,
                          batch_size,
                          fault_model,
                          enable_gaussian_filter=False):

        self.clean_output = list()
        self.clean_inference = True
        self.batch_id = 0
        self.fault_id = None

        self.layer_name = layer_name
        self.network_name = network_name
        self.batch_size = batch_size

        self.fault_model = fault_model

        self.initialize_fault_analysis_dict()

        self.enable_gaussian_filter = enable_gaussian_filter
        if self.enable_gaussian_filter:
            print(f'Gaussian filter enabled on layer {self.layer_name}')


    def initialize_fault_analysis_dict(self):
        self.fault_analysis = {
            'layer_name': list(),
            'fault_id': list(),
            'PSNR': list(),
            'SSIM': list(),
            'euclidean_distance': list(),
            'max_diff': list(),
            'avg_diff': list()
        }

    @staticmethod
    def compute_similarity_metrics(clean_tensor, faulty_tensor):

        torch.use_deterministic_algorithms(False)
        data_range = clean_tensor.max() - clean_tensor.min()

        # Peak signal-to-noise ratio
        psnr = peak_signal_noise_ratio(preds=faulty_tensor,
                                   target=clean_tensor,
                                   reduction=None,
                                   dim=(1, 2, 3),
                                   data_range=data_range).cpu()

        # Structural similarity index metric
        ssim = structural_similarity_index_measure(preds=faulty_tensor,
                                                   target=clean_tensor,
                                                   reduction=None,
                                                   data_range=data_range).cpu()

        torch.use_deterministic_algorithms(True)

        return psnr, ssim


    def forward(self, input_tensor):

        output_tensor = super(AnalyzableConv2d, self).forward(input_tensor)

        if self.enable_gaussian_filter:
            self.gaussian_filter = Conv2d(in_channels=self.out_channels,
                                          groups=self.out_channels,
                                          out_channels=self.out_channels,
                                          padding=1,
                                          kernel_size=3,
                                          device='cuda')
            kernel = [[.0005, .002, .0005], [.002, .99, .002], [.0005, .002, .0005]]
            identity = torch.tensor([kernel] * self.out_channels,
                                    dtype=torch.float32,
                                    device='cuda').unsqueeze(dim=1)
            self.gaussian_filter.weight.data = identity
            # torch.nn.init.dirac_(self.gaussian_filter.weight)
            torch.nn.init.zeros_(self.gaussian_filter.bias)
            self.gaussian_filter.weight.requires_grad = False
            output_tensor = self.gaussian_filter(output_tensor)

        if self.clean_inference:
            self.clean_output.append(output_tensor.detach().cpu())
            self.batch_id += 1

        else:

            clean_tensor = self.clean_output[self.batch_id].cpu()

            # Compute the similarity metrics
            psnr, ssim = self.compute_similarity_metrics(clean_tensor=clean_tensor,
                                                         faulty_tensor=output_tensor)


            euclidean_distance = ((clean_tensor.flatten(start_dim=1) - output_tensor.flatten(start_dim=1))**2).sum(axis=1).cpu()

            # Compute the difference metricz
            difference = torch.abs(output_tensor - clean_tensor)
            max_diff = torch.amax(difference, dim=(1, 2, 3)).cpu()
            avg_diff = torch.mean(difference, dim=(1, 2, 3)).cpu()

            # Append results to dict
            self.fault_analysis['layer_name'].append(self.layer_name)
            self.fault_analysis['fault_id'].append(self.fault_id)
            self.fault_analysis['PSNR'].append(psnr)
            self.fault_analysis['SSIM'].append(ssim)
            self.fault_analysis['euclidean_distance'].append(euclidean_distance)
            self.fault_analysis['max_diff'].append(max_diff)
            self.fault_analysis['avg_diff'].append(avg_diff)



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
