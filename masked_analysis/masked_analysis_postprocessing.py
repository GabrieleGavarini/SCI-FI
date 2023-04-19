from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager
import pandas as pd
import numpy as np

from utils import parse_args

FONT_SIZE = 60
LABEL_FONT_SIZE = 48

OKABE_BLUE = '#0072B2'
OKABE_ORANGE = '#E69F00'
OKABE_RED = '#D55E00'

ALPHA_FILL = .3
ALPHA_EDGE = .95

N_BINS = 50

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
    df.layer_name = df.layer_name.apply(lambda x: x.replace('._SmartModule__module', ''))

    # Merge the df
    merged_df = pd.merge(fl, df)

    # Filter only same layer as fault and img 1
    # merged_df = merged_df[(merged_df.image_id == 1) & (merged_df.fault_layer == merged_df.layer_name)]
    merged_df = merged_df[merged_df.fault_layer == merged_df.layer_name]

    # Get three target df
    critical = merged_df[merged_df.critical]
    different_noncritical = merged_df[merged_df.different_logit & ~merged_df.critical]
    non_different = merged_df[~merged_df.different_logit]
    # different_noncritical = merged_df[merged_df['sdc_1%'] & ~merged_df.critical]
    # non_different = merged_df[~merged_df['sdc_1%']]

    # Filter the infinite results only if the metric is not PSNR
    if metric not in ['PSNR', 'SSIM']:
        # Filtering critical infinite results
        critical.loc[~np.isfinite(critical[metric]), metric] = MAX_FLOAT

        # Filtering different_noncritical results
        different_noncritical.loc[~np.isfinite(different_noncritical[metric]), metric] = MAX_FLOAT

        # Filtering non_different results
        non_different.loc[~np.isfinite(non_different[metric]), metric] = MAX_FLOAT
    else:
        max_metric = merged_df[np.isfinite(merged_df[metric])][metric].max()
        min_metric = merged_df[np.isfinite(merged_df[metric])][metric].min()

        critical.loc[critical[metric] == np.inf, metric] = min_metric
        critical.loc[np.isnan(critical[metric]), metric] = min_metric
        critical.loc[critical[metric] == np.NINF, metric] = min_metric

        # Filtering different_noncritical results
        different_noncritical.loc[different_noncritical[metric] == np.inf, metric] = max_metric
        different_noncritical.loc[different_noncritical[metric] == np.NINF, metric] = min_metric

        # Filtering non_different results
        non_different.loc[non_different[metric] == np.inf, metric] = max_metric
        non_different.loc[non_different[metric] == np.NINF, metric] = min_metric

    return critical, different_noncritical, non_different


def plot_hist(ax, data, bins, color, label, zorder):

    # Dictionary of hist style
    kws = dict(histtype="stepfilled", linewidth=1.2, clip_on=False)

    # Fill
    ax.hist(data,
            bins=bins,
            density=True,
            facecolor=color,
            edgecolor=color,
            label=label,
            alpha=ALPHA_FILL,
            zorder=zorder,
            **kws)

    # Border
    ax.hist(data,
            bins=bins,
            density=True,
            facecolor='None',
            edgecolor=color,
            alpha=ALPHA_EDGE,
            zorder=zorder,
            **kws)


def plot(critical, different_noncritical, non_different, network_name, metric='max_diff'):


    fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = FONT_SIZE


    if metric not in ['PSNR', 'SSIM']:
        # Get what to replace 0 with
        if len(non_different) > 0:
            min_non_different_metric_log = min(np.log10(non_different[non_different[metric] != 0][metric]))
        if len(different_noncritical) > 0:
            min_different_noncritical_metric_log = min(np.log10(different_noncritical[different_noncritical[metric] != 0][metric]))
        if len(critical) > 0:
            min_critical_metric_log = min(np.log10(critical[critical[metric] != 0][metric]))

        # Get the order of magnitude
        non_different[metric] = non_different[metric].apply(lambda x:  np.log10(x) if x != 0 else -37)
        different_noncritical[metric] = different_noncritical[metric].apply(lambda x:  np.log10(x) if x != 0 else min_different_noncritical_metric_log)
        critical[metric] = critical[metric].apply(lambda x:  np.log10(x) if x != 0 else min_critical_metric_log)

    # Get min, max and number of bins
    min_x = min([non_different[metric].min(), different_noncritical[metric].min(), critical[metric].min()])
    max_x = max([non_different[metric].max(), different_noncritical[metric].max(), critical[metric].max()])
    bins = np.linspace(min_x, max_x, N_BINS)

    # Start plot
    fig, ax = plt.subplots(figsize=(21, 9))
    plt.subplots_adjust(bottom=.4, top=1,
                        left=.05, right=.95)

    ax.set_xlim(min_x, max_x)
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_zorder(100)

    # Labels and ticks
    ax.tick_params(left=False)
    ax.tick_params(labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)

    if metric == 'max_diff':
        metric_name = 'Max. Difference'
    elif metric == 'avg_diff':
        metric_name = 'Avg. Difference'
    elif metric == 'euclidean_distance':
        metric_name = 'Euclidean Distance'
    elif metric == 'PSNR':
        metric_name = 'PSNR'
    elif metric == 'SSIM':
        metric_name = 'SSIM'
    else:
        raise Exception(f'Unknown metric {metric}')

    if metric not in ['PSNR', 'SSIM']:
        metric_name = f'{metric_name} OOM'

    ax.set_xlabel(metric_name,
                  fontweight='bold',
                  labelpad=15)

    # Histograms
    kws = dict(histtype="stepfilled", alpha=ALPHA_FILL, linewidth=1.5)
    plot_hist(ax,
              data=critical[metric],
              bins=bins,
              color=OKABE_RED,
              label='Critical',
              zorder=10)
    plot_hist(ax,
              data=different_noncritical[metric],
              bins=bins,
              color=OKABE_ORANGE,
              label='Non-Critical',
              zorder=0)
    plot_hist(ax,
              data=non_different[metric],
              bins=bins,
              color=OKABE_BLUE,
              label='Masked',
              zorder=5)

    ax.legend(bbox_to_anchor=(0.5, -.3),
              loc='upper center',
              ncol=3)

    plt.savefig(f'./plots/oom_{network_name}_{metric}.png')
    fig.show()


def main(args):

    # ------------------------------- #

    network_name = args.network_name
    batch_size = args.batch_size
    fault_model = args.fault_model

    # masked analysis csv folder
    csv_folder = f'{args.root_folder}/output/masked_analysis_results/{network_name}/batch_{batch_size}/{fault_model}'

    # Fault list folder
    if args.fault_model == 'stuck-byzantine_neuron':
        fault_list_file = '51195_neuron_fault_list.csv'
    elif args.fault_model == 'stuck-at_params':
        fault_list_file = '51195_parameters_fault_list.csv'
    else:
        raise IOError(f'No file for the fault model {args.fault_model}')

    fault_list_file = f'{args.root_folder}/output/fault_list/{network_name}/{fault_list_file}'

    # ------------------------------- #

    for metric in ['SSIM', 'PSNR', 'max_diff']:
        critical, different_noncritical, non_different = load_data(fault_list_file=fault_list_file,
                                                                   csv_file=f'{csv_folder}/results.csv',
                                                                   metric=metric)
        plot(critical, different_noncritical, non_different, network_name=network_name, metric=metric)

    # ------------------------------- #



if __name__ == '__main__':
    main(args=parse_args())
