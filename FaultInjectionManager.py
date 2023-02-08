import os
import shutil
import time
import math
from datetime import timedelta
import copy

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from FaultGenerators.NeurontFault import NeuronFault
from FaultGenerators.WeightFaultInjector import WeightFaultInjector
from masked_analysis.AnalyzableConv2d import AnalyzableConv2d
from models.SmartLayers.utils import NoChangeOFMException

from typing import List, Union

from models.SmartLayers.SmartModule import SmartModule
from models.utils import get_module_by_name


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 smart_modules_list: List[SmartModule],
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None):

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        # The folder where to save the output
        self.__faulty_output_folder = f'output/faulty_output/{self.network_name}/batch_{self.loader.batch_size}'

        # The smart modules in the network
        self.__smart_modules_list = smart_modules_list

        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)

        # The list of injectable module, used only for neuron fault injection
        self.injectable_modules = injectable_modules


    def run_clean_campaign(self):

        pbar = tqdm(self.loader,
                    desc='Clean Inference',
                    colour='green')

        for batch_id, batch in enumerate(pbar):
            data, _ = batch
            data = data.to(self.device)

            self.network(data)


    def run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      fault_dropping: bool = True,
                                      fault_delayed_start: bool = True,
                                      delayed_start_module: Module = None,
                                      first_batch_only: bool = False,
                                      save_output: bool = False,
                                      save_feature_maps_statistics: bool = False) -> (str, int):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_model: The faut model for the injection
        :param fault_list: list of fault to inject. One of ['byzantine_neuron', 'stuckat_params']
        :param fault_dropping: Default True. Whether to drop fault or not
        :param fault_delayed_start: Default True. Whether to start the execution from the layer where the faults are
        injected or not
        :param delayed_start_module: Default None. If specified, the module where delayed start is enable. If
        fault_delayed_start = True and this is set to None, the module where delayed start is enabled is assumed to be
        the network
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
        :param save_output: Default False. Whether to save the output of the network or not
        :param save_feature_maps_statistics: Default False. Whether to save statistics about the feature maps after the
        fault injection
        :return: A tuple formed by : (i) a string containing the formatted time elapsed from the beginning to the end of
        the fault injection campaign, (ii) an integer measuring the average memory occupied (in MB)
        """

        self.skipped_inferences = 0
        self.total_inferences = 0

        total_different_predictions = 0
        total_predictions = 0

        average_memory_occupation = 0
        total_iterations = 1

        with torch.no_grad():

            # Order the fault list to speed up the injection
            # This is also important to avoid differences between a
            fault_list = sorted(fault_list, key=lambda x: x.layer_name)

            # Start measuring the time elapsed
            start_time = time.time()

            # Cycle all the batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, _ = batch
                data = data.to(self.device)

                faulty_prediction_dict = dict()
                batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
                batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

                if fault_dropping or fault_delayed_start:
                    # Move the corresponding ofm to the gpu
                    for smart_module in self.__smart_modules_list:
                        smart_module.load_golden(batch_id=batch_id)

                # Inject all the faults in a single batch
                pbar = tqdm(fault_list,
                            colour='green',
                            desc=f'FI on b {batch_id}',
                            ncols=shutil.get_terminal_size().columns)
                for fault_id, fault in enumerate(pbar):

                    # Change the description of the progress bar
                    if fault_dropping and fault_delayed_start:
                        pbar.set_description(f'FI (w/ drop & delayed) on b {batch_id}')
                    elif fault_dropping:
                        pbar.set_description(f'FI (w/ drop) on b {batch_id}')
                    elif fault_delayed_start:
                        pbar.set_description(f'FI (w/ delayed) on b {batch_id}')

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

                        # Do this only if the fault is injected inside one of the layer that allow delayed start
                        if '._SmartModule__module' in fault.layer_name:

                            # The module where delayed start is enabled
                            if delayed_start_module is None:
                                delayed_start_module = self.network

                            # Get the module corresponding to the faulty layer
                            fault_layer = get_module_by_name(container_module=self.network,
                                                             module_name=fault.layer_name)

                            # Get the first-tier layer containing the module where the fault is injected
                            starting_layer = [children for children in delayed_start_module.children()
                                              if fault_layer in children.modules()]
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
                    elif fault_model == 'stuckat_params':
                        self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')


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

                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Store the faulty prediction if the option is set
                    if save_output:
                        self.faulty_output.append({'fault_id': fault_id,
                                                   'faulty_scores': faulty_scores})

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
                    elif fault_model == 'stuckat_params':
                        self.weight_fault_injector.restore_golden()
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    # Increment the iteration count
                    total_iterations += 1

                # Save the output to file if the option is set
                if save_output:
                    os.makedirs(f'{self.__faulty_output_folder}/{fault_model}', exist_ok=True)
                    np.save(f'{self.__faulty_output_folder}/{fault_model}/batch_{batch_id}', self.faulty_output)
                    self.faulty_output = list()

                # TODO: this class shouldn't manage the search of all the instances of AnalyzableConv2d layers
                # Handle the comparison between golden and faulty
                if save_feature_maps_statistics:
                    output_dir = f'output/masked_analysis/{self.network_name}/batch_{self.loader.batch_size}/{fault_model}'
                    os.makedirs(output_dir, exist_ok=True)

                    data_to_save = list()

                    for m in self.network.modules():
                        if isinstance(m, AnalyzableConv2d):
                            data_to_save += copy.deepcopy(m.fault_analysis)
                            m.fault_analysis = list()
                            m.batch_id += 1

                    np.save(f'{output_dir}/batch_{batch_id}', data_to_save)

                # End after only one batch if the option is specified
                if first_batch_only:
                    break

                # Remove all the loaded golden output feature map
                if fault_dropping or fault_delayed_start:
                    for smart_module in self.__smart_modules_list:
                        smart_module.unload_golden()

        elapsed = math.ceil(time.time() - start_time)

        return str(timedelta(seconds=elapsed)), average_memory_occupation


    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data)
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions
            different_predictions = int(torch.ne(faulty_prediction.values, clean_prediction.values).sum())

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
        :param fault_mode: Default 'stuck-at'. One of either 'stuck-at' or 'bit-flip'. Which kind of fault model to
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
        output_fault_mask = torch.zeros(size=self.injectable_modules[fault.layer_index].output_shape)

        layer = fault.layer_index
        channel = fault.feature_map_index[0]
        height = fault.feature_map_index[1]
        width = fault.feature_map_index[2]
        value = fault.value

        # Set values to one for the injected elements
        output_fault_mask[0, channel, height, width] = 1

        # Cast mask to int and move to device
        output_fault_mask = output_fault_mask.int().to(self.device)

        # Create a random output
        output_fault = torch.ones(size=self.injectable_modules[layer].output_shape, device=self.device).mul(value)

        # Inject the fault
        self.injectable_modules[layer].inject_fault(output_fault=output_fault,
                                                    output_fault_mask=output_fault_mask)

        # Return the injected layer
        return self.injectable_modules[layer]
