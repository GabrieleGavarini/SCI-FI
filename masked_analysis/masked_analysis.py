import os
import glob
import re

from tqdm import tqdm

import numpy as np

from utils import parse_args

import pandas as pd


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
    clean_scores = np.load(f'{clean_output_dir}/clean_output.pt', allow_pickle=True)

    # Initialize the list of pd dataframes
    df_list = list()

    pbar = tqdm(total=1,
                desc='Collecting run information',
                colour='yellow')

    # Iterate all the batch
    for batch_id, batch_faulty_output_file in enumerate(os.listdir(faulty_output_dir)):
        batch_faulty_output = np.load(f'{faulty_output_dir}/{batch_faulty_output_file}', allow_pickle=True)

        # Get the clean scores of the current batch
        clean_batch_scores = clean_scores[batch_id]

        # Update the pbar total
        pbar.total = len(os.listdir(faulty_output_dir)) * len(batch_faulty_output)
        pbar.refresh()

        # Load the layers information
        layer_file = f'{masked_analysis_dir}/batch_{batch_id}.npy'
        fault_masked_analysis = np.load(layer_file, allow_pickle=True)

        # Iterate all the faults
        for faulty_output_dict in batch_faulty_output:

            # Get the faulty scores
            fault_id = faulty_output_dict['fault_id']
            faulty_batch_scores = faulty_output_dict['faulty_scores']

            # Get index of the critical faults
            critical = ~clean_batch_scores.argmax(dim=1).eq(faulty_batch_scores.argmax(dim=1)).cpu().numpy()
            critical_index = np.where(critical)

            # Get the layer faulty statistics
            try:
                layer_fault_analysis = fault_masked_analysis[fault_id]
            except Exception:
                continue

            df = pd.DataFrame(layer_fault_analysis)
            df['batch_id'] = batch_id
            df['image_id'] = np.arange(0, batch_size)
            df['critical'] = critical

            df_list.append(df)

            pbar.update(1)

    df = pd.concat(df_list)

    df.to_csv(f'{output_folder}/results.csv', index=False)


if __name__ == '__main__':
    main(args=parse_args())
