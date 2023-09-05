import csv
import os
import numpy as np

LOG_ROOT = '../log'
# LOG_ROOT = '../plugins/NAS/log'
# LOG_ROOT = '../plugins/Quantized/log'

def main():
    # network_name = 'ResNet20'
    # network_name = 'ck_icl_opt_Supernet_5e-09_0.0_109'
    # network_name = 'ResNet20_int32'
    network_name = 'EfficientNet_B4_GTSRB'
    fault_model = 'stuck-at_params'
    batch_size = 128

    log_folder = f'{LOG_ROOT}/{network_name}/batch_{batch_size}/{fault_model}'

    accuracy_list = list()

    file_names = os.listdir(log_folder)
    file_names.sort()

    for file_name in file_names:

        if 'all' in file_name:
            continue

        reader = csv.reader(open(f'{log_folder}/{file_name}', 'r'))
        accuracy_batch_list = np.array([float(line[1]) for line in reader])

        if len(accuracy_batch_list) > 0:
            accuracy_list.append(accuracy_batch_list)

    accuracy_list = np.array(accuracy_list)
    print(accuracy_list.mean())


if __name__ == '__main__':
    main()