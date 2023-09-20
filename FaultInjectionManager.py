import _csv
import csv
import os
import shutil
import time
import math
from datetime import timedelta
from ast import literal_eval as make_tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.WeightFault import WeightFault
from FaultGenerators.WeightFaultInjector import WeightFaultInjector
from FaultGenerators.utils import get_list_of_tuples_from_str, get_list_from_str
from masked_analysis.AnalyzableConv2d import AnalyzableConv2d
from modules.SmartLayers.utils import NoChangeOFMException

from typing import List, Union, TextIO

from modules.SmartLayers.SmartModule import SmartModule
from modules.utils import get_module_by_name


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 smart_modules_list: List[SmartModule],
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 train_loader: DataLoader = None,
                 layer_wise: bool = False,
                 bit_wise: bool = False,
                 injectable_modules: List[Union[Module, List[Module]]] = None,
                 target_layer_index: int = None,
                 target_layer_n: int = None,
                 target_layer_bit: int = None
                 ):

        assert not (layer_wise and bit_wise)

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.train_loader = train_loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        # The folder used to save the labels
        self.__label_folder = f'output/labels/{self.network_name}/batch_{self.loader.batch_size}'
        self.__clean_output_folder = f'output/clean_output/{self.network_name}/batch_{self.loader.batch_size}'

        # The folder used for the logg
        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'
        if layer_wise:
            self.__log_folder = f'{self.__log_folder}/layer_wise'
        elif bit_wise:
            self.__log_folder = f'{self.__log_folder}/bit_wise'
        elif target_layer_n is not None and target_layer_index is not None:
            self.__log_folder = f'{self.__log_folder}/layer_{target_layer_index}_n_{target_layer_n}'
            if target_layer_bit is not None:
                self.__log_folder = f'{self.__log_folder}_bit_{target_layer_bit}'

        # The folder where to save the output
        self.__faulty_output_folder = f'output/faulty_output/{self.network_name}/batch_{self.loader.batch_size}'
        if layer_wise:
            self.__faulty_output_folder = f'{self.__faulty_output_folder}/layer_wise'
        if bit_wise:
            self.__faulty_output_folder = f'{self.__faulty_output_folder}/bit_wise'
        elif target_layer_n is not None and target_layer_index is not None:
            self.__faulty_output_folder = f'{self.__faulty_output_folder}/layer_{target_layer_index}_n_{target_layer_n}'
            if target_layer_bit is not None:
                self.__faulty_output_folder = f'{self.__faulty_output_folder}_bit_{target_layer_bit}'

        # The smart modules in the network
        self.__smart_modules_list = smart_modules_list

        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)

        # The list of injectable module, used only for neuron fault injection
        self.injectable_modules = injectable_modules


    def run_clean_campaign(self, on_train: bool = False):

        torch.cuda.empty_cache()

        if on_train:
            pbar = tqdm(self.train_loader,
                        desc='Clean Inference (training)',
                        colour='green')
        else:
            pbar = tqdm(self.loader,
                        desc='Clean Inference (test)',
                        colour='green')

        label_list = list()

        all_correct_num = 0
        all_sample_num = 0

        clean_output_batch_list = list()

        for batch_id, batch in enumerate(pbar):
            data, label = batch
            data = data.to(self.device)

            label_list.append(label.detach().cpu().numpy())

            predict_y = self.network(data).detach()

            predict_label = torch.argmax(predict_y, dim=-1).cpu()
            current_correct_num = predict_label == label
            all_correct_num += torch.sum(current_correct_num, dim=-1)
            all_sample_num += current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            pbar.set_postfix({'Accuracy': f'{100 * acc:.5f}%'})

            clean_output_batch_list.append(predict_y.detach().cpu().numpy())

        if self.train_loader is not None:
            os.makedirs(self.__clean_output_folder, exist_ok=True)
            np.save(f'{self.__clean_output_folder}/clean_output_train.npy', np.array(clean_output_batch_list, dtype=object), allow_pickle=True)

        label_list = np.concatenate(label_list)

        os.makedirs(self.__label_folder, exist_ok=True)
        np.savez_compressed(f'{self.__label_folder}/labels.npz', label_list)

        torch.cuda.empty_cache()

    def run_fault_injection_campaign(self,
                                     fault_model: str,
                                     fault_list: _csv.reader,
                                     fault_list_file: TextIO,
                                     fault_list_length: int,
                                     exhaustive: bool = False,
                                     fault_dropping: bool = True,
                                     fault_delayed_start: bool = True,
                                     delayed_start_module: Module = None,
                                     golden_ifm_file_extension: str = 'npz',
                                     first_batch_only: bool = False,
                                     save_output: bool = False,
                                     chunk_size:int = None,
                                     save_feature_maps_statistics: bool = False,
                                     multiple_fault_number: int = None,
                                     multiple_fault_percentage: float = None) -> (str, int):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_model: The faut nas_name for the injection
        :param fault_list: the csv file handler indexing the fault list
        :param fault_list_file: The file handled by fault_list
        :param fault_list_length: THe number of fault in the fault list
        :param exhaustive: Default False. Get an exhaustive instead of a statistic one
        :param fault_dropping: Default True. Whether to drop fault or not
        :param fault_delayed_start: Default True. Whether to start the execution from the layer where the faults are
        injected or not
        :param delayed_start_module: Default None. If specified, the module where delayed start is enable. If
        fault_delayed_start = True and this is set to None, the module where delayed start is enabled is assumed to be
        the network
        :param golden_ifm_file_extension: Default 'npz'. The file extension of the file containing the golden ifm loaded
        when using delayed start and fault dropping techniques
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
        :param save_output: Default False. Whether to save the output of the network or not
        :param save_feature_maps_statistics: Default False. Whether to save statistics about the feature maps after the
        fault injection
        :param multiple_fault_percentage: Default None. If the fault nas_name inject multiple faults for a single inference,
        the percentage of affected parameters
        :param multiple_fault_number: Default None. If the fault nas_name inject multiple faults for a single inference,
        the number of affected parameters
        :return: A tuple formed by : (i) a string containing the formatted time elapsed from the beginning to the end of
        the fault injection campaign, (ii) an integer measuring the average memory occupied (in MB)
        """

        self.skipped_inferences = 0
        self.total_inferences = 0

        total_different_predictions = 0
        total_predictions = 0

        average_memory_occupation = 0
        total_iterations = 1

        # Initialize and create the log and the output folder
        if multiple_fault_percentage is not None:
            multiple_fault_postfix = f'/percentage_{multiple_fault_percentage:.0E}'
        elif multiple_fault_number is not None:
            multiple_fault_postfix = f'/number_{multiple_fault_number}'
        else:
            multiple_fault_postfix = ''

        log_folder = f'{self.__log_folder}/{fault_model}/{multiple_fault_postfix}'
        faulty_output_folder = f'{self.__faulty_output_folder}/{fault_model}/{multiple_fault_postfix}'
        os.makedirs(faulty_output_folder, exist_ok=True)
        os.makedirs(log_folder, exist_ok=True)

        with torch.no_grad():

            # Start measuring the time elapsed
            start_time = time.time()

            # The dict measuring the accuracy of each batch
            accuracy_dict = dict()

            # Cycle all the batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, target = batch
                data = data.to(self.device)

                # The list of the accuracy of the network for each fault
                accuracy_batch_dict = dict()
                accuracy_dict[batch_id] = accuracy_batch_dict

                faulty_prediction_dict = dict()
                batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
                batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

                if fault_dropping or fault_delayed_start:
                    # Move the corresponding ofm to the gpu
                    for smart_module in self.__smart_modules_list:
                        smart_module.load_golden(batch_id=batch_id,
                                                 file_extension=golden_ifm_file_extension)

                # Count how many chunks to create
                if chunk_size is not None:
                    number_of_chunks = math.ceil(fault_list_length/chunk_size)
                    print(f'Total number of chunks: {number_of_chunks}')

                # Restart fault list
                # Read the header
                fault_list_file.seek(0)
                _ = next(fault_list)

                # Inject all the faults in a single batch
                pbar = tqdm(fault_list,
                            total=fault_list_length,
                            colour='green',
                            desc=f'FI on b {batch_id}',
                            ncols=shutil.get_terminal_size().columns)
                for fault_id, fault in enumerate(pbar):

                    # Convert the file to the proper object
                    if 'params' in fault_model:
                        fault = WeightFault(layer_name=fault[1],
                                            tensor_index=make_tuple(fault[2]),
                                            bit=int(fault[-1]))
                    elif 'neuron' in fault_model:
                        fault = NeuronFault(layer_name=str(fault[1]),
                                              layer_index=int(fault[2]),
                                              feature_map_indices=get_list_of_tuples_from_str(fault[3]),
                                              value_list=get_list_from_str(fault[-1]))
                    else:
                        raise AttributeError(f'Unknown fault nas_name {fault_model}')

                    # Update the fault with the correct name
                    smart_module_names = [name for name, module in self.network.named_modules() if isinstance(module, SmartModule)]
                    for smart_module_name in smart_module_names:
                        if '._SmartModule__module' not in fault.layer_name:
                            fault.layer_name = fault.layer_name.replace(smart_module_name, f'{smart_module_name}._SmartModule__module')

                    # Change the description of the progress bar
                    # if fault_dropping and fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ drop & delayed) on b {batch_id}')
                    # elif fault_dropping:
                    #     pbar.set_description(f'FI (w/ drop) on b {batch_id}')
                    # elif fault_delayed_start:
                    #     pbar.set_description(f'FI (w/ delayed) on b {batch_id}')

                    # ------ FAULT  DROPPING ------ #

                    if fault_dropping:
                        # List of all the layer for which it is possible to compare the ofm
                        smart_modules_names = [module.layer_name for module in self.__smart_modules_list]
                        try:
                            fault_layer_index = [fault.layer_name.startswith(smart_module_name) for smart_module_name in smart_modules_names].index(True)

                            # Name of the layers to compare
                            if fault_layer_index < len(smart_modules_names) - 1:
                                smart_modules_to_check = smart_modules_names[fault_layer_index + 1: fault_layer_index + 2]
                            else:
                                smart_modules_to_check = None

                            # # Set which ofm to check during the forward pass. Only check the ofm that come after the fault
                            for smart_module in self.__smart_modules_list:

                                # If the layer needs to be checked
                                if smart_modules_to_check is not None and smart_module.layer_name in smart_modules_to_check:
                                    # Add the comparison for the layer after the fault injection
                                    smart_module.compare_with_golden()
                                else:
                                    # Remove the comparison with golden for all the layer previous to the computation of the
                                    # faulty layer
                                    smart_module.do_not_compare_with_golden()

                        except ValueError:
                            # These are layers that are injectable but not inside any of the smart module
                            pass

                    # ----------------------------- #

                    # ---- FAULT DELAYED START ---- #

                    if fault_delayed_start:

                        # Initialization step. This is also useful if the fault is injected in a non-smart layer, then
                        # starting_layer and starting_module should be None
                        delayed_start_module.starting_layer = None
                        delayed_start_module.starting_module = None

                        # The module where delayed start is enabled
                        if delayed_start_module is None:
                            delayed_start_module = self.network

                        # Get the module corresponding to the faulty layer
                        fault_layer = get_module_by_name(container_module=self.network,
                                                         module_name=fault.layer_name)

                        # Get the first-tier layer containing the module where the fault is injected
                        starting_layer = [children for children in delayed_start_module.children()
                                          if fault_layer in children.modules() and isinstance(children, SmartModule)]

                        # Do this only if the fault is injected inside one of the layer that allow delayed start
                        if len(starting_layer) != 0:
                            assert len(starting_layer) == 1
                            starting_layer = starting_layer[0]

                            delayed_start_module.starting_layer = starting_layer

                            # Get the first smart module inside the starting_layer
                            starting_module = [module for module in starting_layer.modules()
                                               if isinstance(module, SmartModule)]
                            starting_module = starting_module[0]

                            delayed_start_module.starting_module = starting_module

                    # ----------------------------- #

                    # Inject faults
                    if fault_model == 'byzantine_neuron':
                        injected_layer = self.__inject_fault_on_neuron(fault=fault)
                    elif fault_model == 'stuck-at_params':
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault nas_name {fault_model}')


                    # TODO: this class shouldn't manage the search of all the instances of AnalyzableConv2d layers
                    # Set the fault id
                    if save_feature_maps_statistics:
                        for m in self.network.modules():
                            if isinstance(m, AnalyzableConv2d):
                                m.fault_id = fault_id

                    # Reset memory occupation stats
                    torch.cuda.reset_peak_memory_stats()

                    # Run inference on the current batch
                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                         data=data)

                    # Measure the memory occupation
                    memory_occupation = (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved()) // (1024**2)
                    average_memory_occupation = ((total_iterations - 1) * average_memory_occupation + memory_occupation) // total_iterations

                    # If fault prediction is None, the fault had no impact. Use golden predictions
                    if faulty_indices is None:
                        faulty_scores = self.clean_output[batch_id]
                        faulty_indices = batch_clean_prediction_indices

                    # Measure the accuracy of the batch
                    accuracy_batch_dict[fault_id] = float(torch.sum(target.eq(torch.tensor(faulty_indices)))/len(target))

                    # Move the scores to the gpu
                    faulty_scores = faulty_scores.detach().cpu()

                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Store the faulty prediction if the option is set
                    if save_output:
                        # For the exhaustive, save just some minor values
                        if exhaustive:
                            self.faulty_output.append(np.array(faulty_indices))
                        else:
                            self.faulty_output.append(faulty_scores.numpy())

                    if save_output and chunk_size is not None:
                        if fault_id !=0 and fault_id % chunk_size == 0:
                            chunk_id = math.floor(fault_id/chunk_size)
                            print(f'Saving chunk {chunk_id}')
                            np.savez_compressed(f'{faulty_output_folder}/batch_{batch_id}_chunk_{chunk_id}', self.faulty_output)
                            self.faulty_output = list()

                    # Measure the loss in accuracy
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.6f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

                    # Clean the fault
                    if fault_model == 'byzantine_neuron':
                        injected_layer.clean_fault()
                    elif fault_model == 'stuck-at_params':
                        self.weight_fault_injector.restore_golden()
                    else:
                        raise ValueError(f'Invalid fault nas_name {fault_model}')

                    # Increment the iteration count
                    total_iterations += 1

                # Log the accuracy of the batch
                log_filename = f'{log_folder}/batch_{batch_id}.csv'
                with open(log_filename, 'w') as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerows(accuracy_batch_dict.items())

                # Save the output to file if the option is set
                if save_output and chunk_size is None:
                    np.savez_compressed(f'{faulty_output_folder}/batch_{batch_id}', self.faulty_output)
                    self.faulty_output = list()

                # TODO: this class shouldn't manage the search of all the instances of AnalyzableConv2d layers
                # Handle the comparison between golden and faulty
                if save_feature_maps_statistics:
                    output_dir = f'output/masked_analysis/{self.network_name}/batch_{self.loader.batch_size}/{fault_model}'
                    os.makedirs(output_dir, exist_ok=True)

                    data_to_save = {
                        'layer_name': list(),
                        'fault_id': list(),
                        'PSNR': list(),
                        'SSIM': list(),
                        'euclidean_distance': list(),
                        'max_diff': list(),
                        'avg_diff': list()
                    }

                    for m in self.network.modules():
                        if isinstance(m, AnalyzableConv2d):
                            data_to_save['layer_name'] = np.concatenate([data_to_save['layer_name'],
                                                                         m.fault_analysis['layer_name']])
                            data_to_save['fault_id'] = np.concatenate([data_to_save['fault_id'],
                                                                       m.fault_analysis['fault_id']])
                            data_to_save['PSNR'] = np.concatenate([data_to_save['PSNR'],
                                                                   m.fault_analysis['PSNR']])
                            data_to_save['SSIM'] = np.concatenate([data_to_save['SSIM'],
                                                                   m.fault_analysis['SSIM']])
                            data_to_save['euclidean_distance'] = np.concatenate([data_to_save['euclidean_distance'],
                                                                                 m.fault_analysis['euclidean_distance']])
                            data_to_save['max_diff'] = np.concatenate([data_to_save['max_diff'],
                                                                       m.fault_analysis['max_diff']])
                            data_to_save['avg_diff'] = np.concatenate([data_to_save['avg_diff'],
                                                                       m.fault_analysis['avg_diff']])
                            m.initialize_fault_analysis_dict()
                            m.batch_id += 1

                    np.savez(f'{output_dir}/batch_{batch_id}',
                             layer_name=data_to_save['layer_name'],
                             fault_id=data_to_save['fault_id'],
                             SSIM=data_to_save['SSIM'],
                             PSNR=data_to_save['PSNR'],
                             euclidean_distance=data_to_save['euclidean_distance'],
                             max_diff=data_to_save['max_diff'],
                             avg_diff=data_to_save['avg_diff'])

                # End after only one batch if the option is specified
                if first_batch_only:
                    break

                # Remove all the loaded golden output feature map
                if fault_dropping or fault_delayed_start:
                    for smart_module in self.__smart_modules_list:
                        smart_module.unload_golden()


        # Measure the average accuracy
        average_accuracy_dict = dict()
        for fault_id in range(fault_list_length):
            fault_accuracy = np.average([accuracy_batch_dict[fault_id] for _, accuracy_batch_dict in accuracy_dict.items()])
            average_accuracy_dict[fault_id] = float(fault_accuracy)

        # Final log
        log_filename = f'{log_folder}/all_batches.csv'
        with open(log_filename, 'w') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerows(average_accuracy_dict.items())


        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed)), average_memory_occupation


    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data).detach()
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions
            # different_predictions = int(torch.ne(faulty_prediction.values, clean_prediction.values).sum())
            different_predictions = int(torch.ne(faulty_prediction.indices, clean_prediction.indices).sum())

            faulty_prediction_scores = network_output
            faulty_prediction_indices = [int(fault) for fault in faulty_prediction.indices]

        except NoChangeOFMException:
            # If the fault doesn't change the output feature map, then simply say that the fault doesn't worsen the
            # network performances for this batch
            faulty_prediction_scores = None
            faulty_prediction_indices = None
            different_predictions = 0
            self.skipped_inferences += 1

        self.total_inferences += 1

        return faulty_prediction_scores, faulty_prediction_indices, different_predictions

    def __inject_fault_on_weight(self,
                                 fault,
                                 fault_mode='stuck-at') -> None:
        """
        Inject a fault in one of the weight of the network
        :param fault: The fault to inject
        :param fault_mode: Default 'stuck-at'. One of either 'stuck-at' or 'bit-flip'. Which kind of fault nas_name to
        employ
        """

        if fault_mode == 'stuck-at':
            self.weight_fault_injector.inject_stuck_at(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,
                                                       value=fault.value)
        elif fault_mode == 'bit-flip':
            self.weight_fault_injector.inject_bit_flip(layer_name=f'{fault.layer_name}.weight',
                                                       tensor_index=fault.tensor_index,
                                                       bit=fault.bit,)
        else:
            print('FaultInjectionManager: Invalid fault mode')
            quit()


    def __inject_fault_on_neuron(self,
                                 fault: NeuronFault) -> Module:
        """
        Inject a fault in the neuron
        :param fault: The fault to inject
        :return: The injected layer
        """

        # Get the target layer
        layer = fault.layer_index

        # Initialize the mask
        output_fault_mask = torch.zeros(size=self.injectable_modules[layer].output_shape, device=self.device)
        # Initialize the faulty output
        output_fault = torch.zeros(size=self.injectable_modules[layer].output_shape, device=self.device)


        # Set the fault for each value in the feature map indices list
        for feature_map_index, feature_map_value in zip(fault.feature_map_indices, fault.value_list):
            channel = feature_map_index[0]
            height = feature_map_index[1]
            width = feature_map_index[2]
            value = feature_map_value

            # Set values to one for the injected elements
            output_fault_mask[0, channel, height, width] = 1
            output_fault[0, channel, height, width] = value

        # Cast mask to int and move to device
        output_fault_mask = output_fault_mask.int()

        # Inject the fault
        self.injectable_modules[layer].inject_fault(output_fault=output_fault,
                                                    output_fault_mask=output_fault_mask)

        # Return the injected layer
        return self.injectable_modules[layer]
