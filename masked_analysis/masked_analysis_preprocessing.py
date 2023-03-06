import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from utils import parse_args

# TODO: LeNet-5, DenseNet-121

def main(args):

    network_name = args.network_name
    batch_size = args.batch_size
    fault_model = args.fault_model

    # Initialize the folder names
    clean_output_dir = f'../output/clean_output/{network_name}/batch_{batch_size}'
    faulty_output_dir = f'../output/faulty_output/{network_name}/batch_{batch_size}/{fault_model}'
    masked_analysis_dir = f'../output/masked_analysis/{network_name}/batch_{batch_size}/{fault_model}'

    # Create the output folder
    output_folder = f'../output/masked_analysis_results/{network_name}/batch_{batch_size}/{fault_model}'
    os.makedirs(output_folder, exist_ok=True)

    # Load the clean output
    clean_scores = np.load(f'{clean_output_dir}/clean_output.npy', allow_pickle=True)

    # Initialize the list of pd dataframes
    df_list = list()

    pbar = tqdm(os.listdir(faulty_output_dir)[:-1],
                desc='Collecting run information',
                colour='yellow')

    # Iterate all the batch
    for batch_id, batch_faulty_output_file in enumerate(pbar):

        # Get the faulty scores of the current batch
        batch_faulty_output = np.load(f'{faulty_output_dir}/{batch_faulty_output_file}')

        # Get the clean scores of the current batch
        clean_batch_scores = clean_scores[batch_id]

        # Get the index of critical faults
        critical = ~(clean_batch_scores.argmax(axis=1) == (batch_faulty_output.argmax(axis=2)))
        critical_df = pd.DataFrame(critical).reset_index().rename(columns={'index': 'fault_id'})
        critical_df = critical_df.melt(id_vars=['fault_id'],
                                       var_name='image_id',
                                       value_name='critical')

        # Get the index of different logits
        different_logit = np.any(~(clean_batch_scores == batch_faulty_output), axis=2)
        different_logit_df = pd.DataFrame(different_logit).reset_index().rename(columns={'index': 'fault_id'})
        different_logit_df = different_logit_df.melt(id_vars=['fault_id'],
                                                     var_name='image_id',
                                                     value_name='different_logit')


        # Load the layers information
        batch_file = f'{masked_analysis_dir}/batch_{batch_id}.npz'
        fault_masked_analysis = np.load(batch_file, allow_pickle=True)

        # Create the dataframe
        df = pd.DataFrame(dict(fault_masked_analysis))

        # Explode the collated columns
        df_exploded = df.explode(['PSNR', 'euclidean_distance', 'max_diff', 'avg_diff', 'num_diff_percentage'])
        assert len(df_exploded) <= len(df) * batch_size

        # Add the remaining fields
        num_batches = int(len(df_exploded) / batch_size)
        df_exploded['image_id'] = list(range(batch_id * batch_size, (batch_id + 1) * batch_size)) * num_batches
        df_exploded['batch_id'] = batch_id

        # Cast tensor to float
        df_exploded['fault_id'] = df_exploded['fault_id'].apply(lambda x: int(x))
        df_exploded['PSNR'] = df_exploded['PSNR'].apply(lambda x: float(x))
        df_exploded['euclidean_distance'] = df_exploded['euclidean_distance'].apply(lambda x: float(x))
        df_exploded['max_diff'] = df_exploded['max_diff'].apply(lambda x: float(x))
        df_exploded['avg_diff'] = df_exploded['avg_diff'].apply(lambda x: float(x))
        df_exploded['num_diff_percentage'] = df_exploded['num_diff_percentage'].apply(lambda x: float(x))

        # Add the critical and logit information
        df_exploded = pd.merge(df_exploded, critical_df)
        df_exploded = pd.merge(df_exploded, different_logit_df)

        # Reorder the columns
        df_exploded = df_exploded[['fault_id', 'image_id', 'different_logit', 'critical', 'layer_name', 'PSNR', 'euclidean_distance', 'max_diff', 'avg_diff', 'num_diff_percentage']]

        df_list.append(df_exploded)

    df = pd.concat(df_list)

    df.to_csv(f'{output_folder}/results.csv', index=False)


if __name__ == '__main__':
    main(args=parse_args())
