from typing import Tuple

import os
import numpy as np
import torch
import pandas as pd

from matplotlib import pyplot as plt

LOG_ROOT = '../plugins/NAS/finetuning_checkpoints'
OUTPUT_ROOT = '../plugins/NAS/output'

NET_NAME_LIST = ['ck_icl_opt_Supernet_0.01_0.0_080',
                 'ck_icl_opt_Supernet_0.001_0.0_026',
                 'ck_icl_opt_Supernet_0.05_0.0_046',
                 'ck_icl_opt_Supernet_0.005_0.0_023',
                 'ck_icl_opt_Supernet_0.0005_0.0_014',
                 'ck_icl_opt_Supernet_0.0001_0.0_021',
                 'ck_icl_opt_Supernet_5e-05_0.0_025',
                 'ck_icl_opt_Supernet_1e-05_0.0_080',
                 'ck_icl_opt_Supernet_3e-06_0.0_067',
                 'ck_icl_opt_Supernet_5e-06_0.0_108',
                 'ck_icl_opt_Supernet_1e-06_0.0_146',
                 'ck_icl_opt_Supernet_5e-07_0.0_169',
                 'ck_icl_opt_Supernet_3e-07_0.0_203',
                 'ck_icl_opt_Supernet_1e-07_0.0_081',
                 'ck_icl_opt_Supernet_1e-08_0.0_079',
                 'ck_icl_opt_Supernet_1e-09_0.0_091',
                 'ck_icl_opt_Supernet_5e-08_0.0_110',
                 'ck_icl_opt_Supernet_5e-09_0.0_109']

def remove_outliers(data: np.array,
                    m: int =2) -> Tuple[np.array, np.array]:
    """
    Remove the outliers from a numpy array
    :param data: The data to be filtered
    :param m: The strength of the filtering
    :return: The filtered data and the outlier data
    """

    outliers_index = ~(abs(data - np.mean(data)) < m * np.std(data))

    return data[~outliers_index], data[outliers_index]

def plot_hist(weight_list: np.array,
              n_bins: int = 100):

    use_log = (weight_list.max() / weight_list.min()) > 100

    if use_log:
        weight_list, outliers = remove_outliers(weight_list)
    else:
        outliers = np.empty(0)

    offset = 0 if weight_list.min() > 0 else (-weight_list.min() + 1)

    min_x = weight_list.min() + offset
    max_x = weight_list.max() + offset


    # Create the bins and remove the offset
    if use_log:
        bins = np.logspace(np.log(min_x), np.log(max_x), n_bins) - offset
    else:
        bins = np.linspace(min_x, max_x, n_bins) - offset

    plt.hist(weight_list, bins=bins, density=True)

    if use_log:
        plt.xscale('log')

    plt.xlabel(f'Weight Value{" with no outliers [log]" if use_log else ""}')
    xlim = max(abs(min_x), abs(max_x))
    plt.xlim(-xlim, xlim)

    plt.show()

def main():
    df_dict = dict()

    for network_name in NET_NAME_LIST:

        weight_file = f'{LOG_ROOT}/{network_name}.pt'

        weight_dict = torch.load(weight_file, map_location='cpu')['model_state_dict']
        weight_list = np.concatenate([weight.numpy().flatten() for layer_name, weight in weight_dict.items() if 'weight' in layer_name])

        weight_mean = weight_list.mean()
        weight_var = weight_list.var()

        plot_hist(weight_list)

        print(f'{network_name} \t {weight_mean:.5f} \t {weight_var:.5f}')

        df_dict[network_name] = {'network_name': network_name,
                                 'weight_mean': weight_mean,
                                 'weight_var': weight_var}

    df = pd.DataFrame(df_dict).T
    analysis_output_folder = f'{OUTPUT_ROOT}/analysis/'
    os.makedirs(analysis_output_folder, exist_ok=True)
    df.to_csv(f'{analysis_output_folder}/networks_weight_distributions.csv')

if __name__ == '__main__':
    main()