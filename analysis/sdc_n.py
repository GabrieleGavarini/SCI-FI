import csv

from tqdm import tqdm
import os
import numpy as np
import torch

NETWORK_NAME = 'ResNet20'
# NETWORK_NAME = 'LeNet5_MNIST'
# NETWORK_NAME = 'LeNet5_MNIST_int8'

OUTPUT_ROOT = '../output'
# OUTPUT_ROOT = '../plugins/Quantized/output'
# OUTPUT_ROOT = '../plugins/NAS/output'
LAYER_WISE = False
BIT_WISE = False
BIT_ANALYSIS = False
BIT_WIDTH = 31
SAMPLES_FILE='../n_samples_step0.csv'

if NETWORK_NAME=='ResNet20':
    TOT_LAYERS=20
    TOT_BITPOSITIONS=31

aggregation_file="tot_p_bitposition_step0_new.csv"
SPLIT_BY_LAYER = True
SPLIT_BY_BIT = not SPLIT_BY_LAYER

assert not (SPLIT_BY_BIT and SPLIT_BY_LAYER)
assert  not (LAYER_WISE and BIT_WISE)

def main():
    # network_name = 'ResNet20'
    # network_name = 'ck_icl_opt_Supernet_5e-09_0.0_109'
    # network_name = 'ResNet20_int8'
    # network_name = 'EfficientNet_B4_GTSRB'

    for nlayer in range(TOT_LAYERS):
        for bitpositions in range(TOT_BITPOSITIONS, 0, -1):
            if bitpositions == 30:
                TARGET_LAYER_INDEX = nlayer
                TARGET_LAYER_BIT = bitpositions
                TARGET_LAYER_N = getsamples(readSamples(SAMPLES_FILE=SAMPLES_FILE), BIT_WIDTH - TARGET_LAYER_BIT, TARGET_LAYER_INDEX)

                fault_model = 'stuck-at_params'
                # fault_model = 'byzantine_neuron'

                percentage = 'percentage_1E-07'
                if fault_model == 'byzantine_neuron':
                    fault_model = f'{fault_model}/{percentage}'
                batch_size = 128

                if LAYER_WISE:
                    folder_prefix = "layer_wise/"
                    file_prefix = "layer_wise_"
                elif BIT_WISE:
                    folder_prefix = "bit_wise/"
                    file_prefix = "bit_wise_"
                elif TARGET_LAYER_INDEX is not None and TARGET_LAYER_N is not None:
                    folder_prefix = f'layer_{TARGET_LAYER_INDEX}_n_{TARGET_LAYER_N}/'
                    file_prefix = f'layer_{TARGET_LAYER_INDEX}_n_{TARGET_LAYER_N}_'
                    if TARGET_LAYER_BIT is not None:
                        folder_prefix = f'{folder_prefix[:-1]}_bit_{TARGET_LAYER_BIT}/'
                        file_prefix = f'{file_prefix}bit_{TARGET_LAYER_BIT}_'
                else:
                    folder_prefix = ''
                    file_prefix = ''
                    raise UserWarning('Unmanaged case')


                if SPLIT_BY_LAYER:
                    analysis_prefix = 'layer_'
                elif SPLIT_BY_BIT:
                    analysis_prefix = 'bit_'
                else:
                    analysis_prefix = ''

                # Manage target layer fault injection
                if TARGET_LAYER_INDEX is not None and TARGET_LAYER_N is not None:
                    analysis_prefix = f'target_layer_{TARGET_LAYER_INDEX}_n_{TARGET_LAYER_N}_'
                    if TARGET_LAYER_BIT is not None:
                        analysis_prefix = f'{analysis_prefix}bit_{TARGET_LAYER_BIT}_'

                print(f'Analyzing sdc-n of {NETWORK_NAME} with {fault_model} faults')

                fault_list_folder = f'{OUTPUT_ROOT}/fault_list/{NETWORK_NAME}{f"/{percentage}" if "neuron" in fault_model else ""}'
                faulty_output_folder = f'{OUTPUT_ROOT}/faulty_output/{NETWORK_NAME}/batch_{batch_size}/{folder_prefix}{fault_model}'
                clean_output_folder = f'{OUTPUT_ROOT}/clean_output/{NETWORK_NAME}/batch_{batch_size}'
                labels_folder = f'{OUTPUT_ROOT}/labels/{NETWORK_NAME}/batch_{batch_size}'

                # Analysis Folder
                analysis_folder = f'{OUTPUT_ROOT}/analysis/{NETWORK_NAME}/batch_{batch_size}/{fault_model}'
                os.makedirs(analysis_folder, exist_ok=True)

                
                aggregate_results(analysis_folder=analysis_folder)
                exit()

                # Load the labels
                labels = np.load(f'{labels_folder}/labels.npz', allow_pickle=True)
                labels = labels['arr_0']

                # Load the fault list
                # TODO: manage neuron fault models
                if 'param' in fault_model:
                    fault_name = 'parameters'
                else:
                    fault_name = 'neuron'

                fault_list_filename = f'{fault_list_folder}/51195_{file_prefix}{fault_name}_fault_list.csv'
                # Open the fault list and the fault list reader
                fault_list_file = open(fault_list_filename, 'r')
                fault_list_csv_reader = csv.reader(fault_list_file)
                # Skip the header
                header = next(fault_list_csv_reader)

                # Load the clean output
                clean_output = np.load(f'{clean_output_folder}/clean_output.npy', allow_pickle=True)

                # Dict with the results of each layer
                results_dict = dict()

                pbar = tqdm(os.listdir(faulty_output_folder))

                for file_name in pbar:
                    batch_id = int(file_name.replace('batch_', '').replace('.npz', ''))

                    # Get the labels
                    labels_batch = labels[batch_id * batch_size: (batch_id + 1) * batch_size]

                    # Get the clean top1
                    clean_batch_results = clean_output[batch_id]
                    clean_top_1 = np.argmax(clean_batch_results, axis=1)
                    clean_top_5 = torch.topk(torch.from_numpy(clean_batch_results), k=5, axis=1)[1].numpy()

                    # Load the faulty output
                    faulty_batch_output = np.load(f'{faulty_output_folder}/{file_name}')
                    faulty_batch_output = faulty_batch_output['arr_0']

                    # For each fault
                    for faulty_batch_result_fault in faulty_batch_output:

                        # Compute the clean correct prediction
                        clean_correct = np.sum(np.equal(labels_batch, clean_top_1))

                        fault_description = next(fault_list_csv_reader)

                        if BIT_ANALYSIS or SPLIT_BY_BIT:
                            faulty_bit = int(fault_description[-1])
                            entry_name = faulty_bit
                        elif SPLIT_BY_LAYER:
                            entry_name = fault_description[1]
                        else:
                            entry_name = 'network'

                        # Check if dict has entry
                        if entry_name not in results_dict.keys():
                            results_dict[entry_name] = {'total_predictions': 0,
                                                        'clean_correct_predictions': 0,
                                                        'faulty_correct_predictions': 0,
                                                        'sdc1_predictions': 0,
                                                        'sdc5_predictions': 0,
                                                        'soft_sdc5_predictions': 0}

                        # Get the faulty top1
                        faulty_top_1 = np.argmax(faulty_batch_result_fault, axis=1)
                        faulty_top_5 = torch.topk(torch.from_numpy(faulty_batch_result_fault), k=5, axis=1)[1].numpy()

                        # Compute the faulty correct prediction
                        faulty_correct = np.sum(np.equal(labels_batch, faulty_top_1))

                        sdc1_predictions = np.sum(~(faulty_top_1 == clean_top_1))
                        sdc5_predictions = np.sum(~np.any(faulty_top_1.reshape(-1, 1) == clean_top_5, axis=1))
                        soft_sdc5_predictions = np.sum(~np.all(np.equal(np.sort(clean_top_5), np.sort(faulty_top_5)), axis=1))
                        total_predictions = len(faulty_top_1)

                        # Put info inside the dict
                        results_dict[entry_name]['total_predictions'] += total_predictions
                        results_dict[entry_name]['clean_correct_predictions'] += clean_correct
                        results_dict[entry_name]['faulty_correct_predictions'] += faulty_correct
                        results_dict[entry_name]['sdc1_predictions'] += sdc1_predictions
                        results_dict[entry_name]['sdc5_predictions'] += sdc5_predictions
                        results_dict[entry_name]['soft_sdc5_predictions'] += soft_sdc5_predictions

                        accuracy_clean = results_dict[entry_name]['clean_correct_predictions'] / results_dict[entry_name]['total_predictions']
                        accuracy_faulty = results_dict[entry_name]['faulty_correct_predictions'] / results_dict[entry_name]['total_predictions']
                        sdc_1 = results_dict[entry_name]['sdc1_predictions'] / results_dict[entry_name]['total_predictions']
                        sdc_5 = results_dict[entry_name]['sdc5_predictions'] / results_dict[entry_name]['total_predictions']
                        soft_sdc_5 = results_dict[entry_name]['soft_sdc5_predictions'] / results_dict[entry_name]['total_predictions']
                        pbar.set_postfix({'Delta Accuracy': f'{100 * abs(accuracy_clean - accuracy_faulty):.2f}%',
                                        'SDC-1': f'{100 * sdc_1:.2f}%',
                                        'SDC-5': f'{100 * sdc_5:.2f}%',
                                        'Soft SDC-5': f'{100 * soft_sdc_5:.2f}%'})

                    fault_list_file.seek(0)
                    header = next(fault_list_csv_reader)

                df_dict = dict()
                # Sort the dict
                results_dict = dict(sorted(results_dict.items(), reverse=True))

                # Add a new entry for the whole network
                if TARGET_LAYER_INDEX is None and TARGET_LAYER_N is None and 'network' not in results_dict.keys():
                    results_dict['network'] = {
                        'total_predictions': np.sum([r['total_predictions'] for r in results_dict.values()]),
                        'clean_correct_predictions': np.sum([r['clean_correct_predictions'] for r in results_dict.values()]),
                        'faulty_correct_predictions': np.sum([r['faulty_correct_predictions'] for r in results_dict.values()]),
                        'sdc1_predictions': np.sum([r['sdc1_predictions'] for r in results_dict.values()]),
                        'sdc5_predictions': np.sum([r['sdc5_predictions'] for r in results_dict.values()]),
                        'soft_sdc5_predictions': np.sum([r['soft_sdc5_predictions'] for r in results_dict.values()])}

                with open(f'{analysis_folder}/{analysis_prefix}sdc.csv', 'w') as output_file:

                    csv_writer = csv.writer(output_file)
                    header = ['Layer', 'Golden Accuracy', 'SDC_1', 'SDC_5', 'Soft_SDC_5', 'Delta_Accuracy']
                    if SPLIT_BY_BIT:
                        header[0] = 'Bit'
                    csv_writer.writerow(header)

                    for layer_results in results_dict.items():
                        # Get info
                        entry_name = layer_results[0]
                        clean_accuracy = layer_results[1]['clean_correct_predictions'] / layer_results[1]['total_predictions']
                        faulty_accuracy = layer_results[1]['faulty_correct_predictions'] / layer_results[1]['total_predictions']
                        sdc_1 = layer_results[1]['sdc1_predictions'] / layer_results[1]['total_predictions']
                        sdc_5 = layer_results[1]['sdc5_predictions'] / layer_results[1]['total_predictions']
                        soft_sdc_5 = layer_results[1]['soft_sdc5_predictions'] / layer_results[1]['total_predictions']

                        # Create dict
                        df_dict[entry_name] = {'Golden Accuracy': clean_accuracy,
                                            'SDC_1': sdc_1,
                                            'SDC_5': sdc_5,
                                            'Soft_SDC_5': soft_sdc_5,
                                            'Delta Accuracy': abs(clean_accuracy - faulty_accuracy)}

                        csv_writer.writerow([entry_name] + list(df_dict[entry_name].values()))

                        # Print results
                        print(f'{layer_results[0]}:',
                            f' \t Golden Acc={100 * clean_accuracy:.2f}%',
                            f' \t Delta Acc={100 * abs(clean_accuracy - faulty_accuracy):.2f}%',
                            f' \t SDC-1={100 * sdc_1:.2f}%',
                            f' \t SDC-5={100 * sdc_5:.2f}%',
                            f' \t Soft SDC-5={100 * soft_sdc_5:.2f}%')

    aggregate_results(analysis_folder=analysis_folder)
    


def readSamples(SAMPLES_FILE):
    database_samples=[]
    with open(SAMPLES_FILE) as csv_file:
        csv_reader= csv.reader(csv_file, delimiter=";")
        for row in csv_reader:
            database_samples.append(row)
    return database_samples
            

# This function returns the number of faults (n) to inject in a specific bitposition of a specific layer
def getsamples(database_samples, bitposition, layerposition):
    if database_samples[layerposition][bitposition][0].isnumeric():
        return int(database_samples[layerposition][bitposition])
    else:
        return int(database_samples[layerposition][bitposition][1:])

def aggregate_results(analysis_folder):
    array_results=[]
    with open(aggregation_file, "a") as csv_outputfile:
        csv_writer= csv.writer(csv_outputfile, delimiter=",")
        for bitpositions in range(TOT_BITPOSITIONS):
            #p_bitposition=0
            tot_xsuccesses=0
            tot_n_faults=0
            if bitpositions == 30:
                for nlayer in range(TOT_LAYERS):
                    print("************************************\n")
                    print(f"Running layer {nlayer}")
                    TARGET_LAYER_INDEX = nlayer
                    TARGET_LAYER_BIT = bitpositions
                    TARGET_LAYER_N = getsamples(readSamples(SAMPLES_FILE=SAMPLES_FILE), BIT_WIDTH - TARGET_LAYER_BIT, TARGET_LAYER_INDEX)
                    file_to_open=f"target_layer_{nlayer}_n_{TARGET_LAYER_N}_bit_{bitpositions}_sdc.csv"

                    with open(f'{analysis_folder}/{file_to_open}',"r") as csv_inputfile:
                        csv_reader= csv.reader(csv_inputfile, delimiter=",")
                        next(csv_reader) #this skips the header
                        for row in csv_reader:
                            p_successes=float(row[2])
                            print(f"Percentage of Successes {p_successes}")
                            print(f"Number of Injections {TARGET_LAYER_N}")
                            xsuccess=p_successes*TARGET_LAYER_N
                            tot_xsuccesses= tot_xsuccesses + xsuccess
                            tot_n_faults=tot_n_faults+TARGET_LAYER_N
                            print(f"Number of Successes {xsuccess}")
                            #p_bitposition = p_bitposition + float(row[2])  #SDC-1 WRONG
                print("************************************\n")
                print(f"TOT Number of Successes {tot_xsuccesses}")
                print(f"TOT Number of Injected faults {tot_n_faults}")
                array_results.insert(bitpositions, float(tot_xsuccesses/tot_n_faults))
                print(f'Inserted in array results in position {bitpositions}')

        # Write the list of p1 for every bitposition
        header=[i for i in range(TOT_BITPOSITIONS)]
        csv_writer.writerow(header)
        csv_writer.writerow(array_results)
        print(len(array_results))

if __name__ == '__main__':
    
    main()