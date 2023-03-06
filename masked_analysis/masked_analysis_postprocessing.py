from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from utils import parse_args

OKABE_BLUE = '#0072B2'
OKABE_ORANGE = '#E69F00'
OKABE_RED = '#D55E00'

ALPHA_FILL = .85
ALPHA_BORDER = .95

MAX_FLOAT = np.finfo(np.float32).max


def load_data(fault_list_file, csv_file, metric='max_diff'):

    # Read and modify the fault list df
    fl = pd.read_csv(f'{fault_list_file}')
    fl = fl[['Injection', 'Layer', 'Bit']]
    fl = fl.rename({'Injection': 'fault_id',
                    'Layer': 'fault_layer',
                    'Bit': 'fault_bit'}, axis=1)

    # Read the masked analysis df
    df = pd.read_csv(csv_file)

    # Merge the df
    merged_df = pd.merge(fl, df)

    # Filter only same layer as fault and img 1
    # merged_df = merged_df[(merged_df.image_id == 1) & (merged_df.fault_layer == merged_df.layer_name)]
    merged_df = merged_df[merged_df.fault_layer == merged_df.layer_name]

    # Get three target df
    critical = merged_df[merged_df.critical]
    different_noncritical = merged_df[merged_df.different_logit & ~merged_df.critical]
    non_different = merged_df[~merged_df.different_logit]

    # Filter the infinite results only if the metric is not PSNR
    if metric not in ['PSNR']:
        # Filtering critical infinite results
        critical.loc[~np.isfinite(critical[metric]), metric] = MAX_FLOAT

        # Filtering different_noncritical results
        different_noncritical.loc[~np.isfinite(different_noncritical[metric]), metric] = MAX_FLOAT

        # Filtering non_different results
        non_different.loc[~np.isfinite(non_different[metric]), metric] = MAX_FLOAT
    else:
        max_psnr = merged_df[np.isfinite(merged_df[metric])][metric].max()
        min_psnr = merged_df[np.isfinite(merged_df[metric])][metric].min()

        critical.loc[critical[metric] == np.inf, metric] = max_psnr
        critical.loc[critical[metric] == np.NINF, metric] = min_psnr

        # Filtering different_noncritical results
        different_noncritical.loc[different_noncritical[metric] == np.inf, metric] = max_psnr
        different_noncritical.loc[different_noncritical[metric] == np.NINF, metric] = min_psnr

        # Filtering non_different results
        non_different.loc[non_different[metric] == np.inf, metric] = max_psnr
        non_different.loc[non_different[metric] == np.NINF, metric] = min_psnr

    return critical, different_noncritical, non_different


def plot(critical, different_noncritical, non_different, network_name, metric='max_diff'):

    if metric not in ['PSNR']:
        # Get what to replace 0 with
        min_non_different_metric_log = min(np.log10(non_different[non_different[metric] != 0][metric]))
        min_different_noncritical_metric_log = min(np.log10(different_noncritical[different_noncritical[metric] != 0][metric]))
        min_critical_metric_log = min(np.log10(critical[critical[metric] != 0][metric]))

        # Get the order of magnitude
        non_different[metric] = non_different[metric].apply(lambda x:  np.log10(x) if x != 0 else -37)
        different_noncritical[metric] = different_noncritical[metric].apply(lambda x:  np.log10(x) if x != 0 else min_different_noncritical_metric_log)
        critical[metric] = critical[metric].apply(lambda x:  np.log10(x) if x != 0 else min_critical_metric_log)

    # Get min, max and number of bins
    min_x = min([non_different[metric].min(), different_noncritical[metric].min(), critical[metric].min()])
    max_x = max([non_different[metric].max(), different_noncritical[metric].max(), critical[metric].max()])
    bins = np.linspace(min_x, max_x, 100)

    # Start plot
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.set_xlim(min_x, max_x)

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels and ticks
    ax.tick_params(left=False)
    ax.tick_params(labelleft=False)

    if metric == 'max_diff':
        metric_name = 'Max. Difference'
    elif metric == 'avg_diff':
        metric_name = 'Avg. Difference'
    elif metric == 'euclidean_distance':
        metric_name = 'Euclidean Distance'
    elif metric == 'PSNR':
        metric_name = 'PSNR'
    else:
        raise Exception(f'Unknown metric {metric}')

    if metric not in ['PSNR']:
        metric_name = f'{metric_name} Order of Magnitude'

    ax.set_xlabel(metric_name)

    # Histograms
    kws = dict(histtype="stepfilled", alpha=ALPHA_FILL, linewidth=1.2)
    ax.hist(different_noncritical[metric],
            bins=bins,
            density=True,
            color=OKABE_ORANGE,
            edgecolor=OKABE_ORANGE,
            label='Different Non-Critical',
            **kws)
    ax.hist(non_different[metric],
            bins=bins,
            density=True,
            color=OKABE_BLUE,
            edgecolor=OKABE_BLUE,
            label='Non Different',
            **kws)
    ax.hist(critical[metric],
            bins=bins,
            density=True,
            color=OKABE_RED,
            edgecolor=OKABE_RED,
            label='Critical',
            **kws)

    ax.legend()

    plt.savefig(f'./plots/oom_{network_name}_{metric}.png')
    fig.show()


def main(args):

    metric = 'PSNR'

    # ------------------------------- #

    network_name = args.network_name
    batch_size = args.batch_size
    fault_model = args.fault_model

    # masked analysis csv folder
    csv_folder = f'../output/masked_analysis_results/{network_name}/batch_{batch_size}/{fault_model}'

    # Fault list folder
    if args.fault_model == 'stuck-byzantine_neuron':
        fault_list_file = '51195_neuron_fault_list.csv'
    elif args.fault_model == 'stuck-at_params':
        fault_list_file = '51195_parameters_fault_list.csv'
    else:
        raise IOError(f'No file for the fault model {args.fault_model}')

    fault_list_file = f'../output/fault_list/{network_name}/{fault_list_file}'

    # ------------------------------- #

    critical, different_noncritical, non_different = load_data(fault_list_file=fault_list_file,
                                                               csv_file=f'{csv_folder}/results.csv',
                                                               metric=metric)
    plot(critical, different_noncritical, non_different, network_name=network_name, metric=metric)

    # ------------------------------- #



if __name__ == '__main__':
    main(args=parse_args())
