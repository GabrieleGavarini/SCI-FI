import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch

from utils import int_bit_flip, float_bit_flip

OUTPUT_ROOT = '../output'
# MODEL_WEIGHT_FOLDER = '../modules/pretrained_models/ResNet20.th'
# DATA_TYPE = 'float32'
DATA_TYPE = 'int8'
DATA_TYPE_NAME = '_torch.qint8'
MODEL_WEIGHT_FOLDER = f'../modules/pretrained_models/LeNet5_MNIST{DATA_TYPE_NAME}.pt'


def bit_flip(value,
             bit: int):
    if value.dtype == torch.qint8 or value.dtype == torch.qint32:
        return int_bit_flip(value, bit)
    elif value.dtype == torch.float:
        return float_bit_flip(value, bit)

def main():
    # network_name = 'ResNet20'
    network_name = 'LeNet5_MNIST'

    # Compute the exponent range (if available)
    if DATA_TYPE == 'float32':
        exponent_range = range(23, 31)
    else:
        exponent_range = None

    # Compute the bit width
    if DATA_TYPE == 'float32' or DATA_TYPE == 'int32':
        bit_width = 32
    elif DATA_TYPE == 'float16':
        bit_width = 16
    elif DATA_TYPE == 'int8':
        bit_width = 8
    else:
        raise AttributeError(f'Unknown data type {DATA_TYPE}')

    if '.th' in MODEL_WEIGHT_FOLDER:
        weights = torch.load(MODEL_WEIGHT_FOLDER, map_location='cpu')['state_dict']
    elif '.pt' in MODEL_WEIGHT_FOLDER:
        weights = torch.load(MODEL_WEIGHT_FOLDER, map_location='cpu')
    else:
        raise FileNotFoundError(f'Unknown file extension for file {MODEL_WEIGHT_FOLDER}')

    weights = {layer_name: layer_weights for layer_name, layer_weights in weights.items() if 'weight' in layer_name}

    # Dict of list containing alle the bit flip distance
    bit_flip_distance_dict = dict()
    bit_flip_ratio_dict = dict()
    faulty_dict = dict()

    # Dictionaries of results
    result_dict = {'MBFD': np.empty(bit_width),
                   'MBFR': np.empty(bit_width),
                   'count_0': np.zeros(bit_width),
                   'count_1': np.zeros(bit_width)}

    for bit in range(bit_width):
    # for bit in exponent_range:
        bit_flip_distance_dict[bit] = list()
        bit_flip_ratio_dict[bit] = list()
        faulty_dict[bit] = list()

        # Initialize the progress bar
        pbar = tqdm(weights.items(),
                    desc=f'Measuring the ABFD for bit {str(bit).zfill(2)}')

        for layer_name, layer_weights in pbar:

            for layer_channel_weights in layer_weights:
                for golden_value in layer_channel_weights.flatten():
                    faulty_value, faulted_bit = bit_flip(golden_value, bit)
                    if DATA_TYPE == 'float32':
                        golden_value = float(golden_value)
                    else:
                        golden_value = golden_value.int_repr()
                    faulty_dict[bit].append(faulty_value)

                    result_dict['count_0'][bit] += int(faulted_bit == 0)
                    result_dict['count_1'][bit] += int(faulted_bit == 1)

                    distance = np.abs(float(golden_value) - float(faulty_value))
                    # relative_distance = distance/np.abs(float(golden_value))
                    bit_flip_distance_dict[bit].append(distance)

                    bit_flip_ratio = faulty_value/golden_value if golden_value != 0 else 0

                    bit_flip_ratio_dict[bit].append(bit_flip_ratio)

            result_dict['MBFD'][bit] = np.mean(bit_flip_distance_dict[bit])
            result_dict['MBFR'][bit] = np.median(bit_flip_ratio_dict[bit])
            pbar.set_postfix({'MBFD': f'{result_dict["MBFD"][bit]:.2E}',
                              'MBFR': f'{result_dict["MBFR"][bit]:.2f}'})

    # Create DataFrame and save to csv
    df = pd.DataFrame(result_dict)
    analysis_output_folder = f'{OUTPUT_ROOT}/analysis/{network_name}/'
    os.makedirs(analysis_output_folder, exist_ok=True)
    df.to_csv(f'{analysis_output_folder}/ABFD.csv')

if __name__ == '__main__':
    main()