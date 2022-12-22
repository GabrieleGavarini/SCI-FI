import shutil
import time
import math
from datetime import timedelta
import re

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from FaultGenerators.WeightFaultInjector import WeightFaultInjector
from models.SmartLayers.utils import NoChangeOFMException
from utils import formatted_print

from typing import List

from models.SmartLayers.SmartModule import SmartModule


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 smart_modules_list: List[SmartModule],
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor):

        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output

        # The network truncated from a starting layer
        self.faulty_network = None

        # The smart modules in the network
        self.__smart_modules_list = smart_modules_list


        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)


    def run_faulty_campaign_on_weight(self,
                                      fault_list: list,
                                      fault_dropping: bool = True,
                                      fault_delayed_start: bool = True,
                                      delayed_start_module: Module = None,
                                      first_batch_only: bool = False) -> (str, int):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_list: list of fault to inject
        :param fault_dropping: Default True. Whether to drop fault or not
        :param fault_delayed_start: Default True. Whether to start the execution from the layer where the faults are
        injected or not
        :param delayed_start_module: Default None. If specified, the module where delayed start is enable. If
        fault_delayed_start = True and this is set to None, the module where delayed start is enabled is assumed to be
        the network
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
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

                    if fault_dropping:
                        # List of all the layer for which it is possible to compare the ofm
                        smart_modules_names = [module.layer_name for module in self.__smart_modules_list]
                        try:
                            fault_layer_index = [fault.layer_name.startswith(smart_module_name) for smart_module_name in smart_modules_names].index(True)
                        except ValueError:
                            # These are layers that are injectable but not inside any of the smart module
                            continue
                        # fault_layer_index = smart_modules_names.index(fault.layer_name.replace('._SmartModule__module', ''))

                        # Set which ofm to check during the forward pass. Only check the ofm that come after the fault
                        for smart_module in self.__smart_modules_list:

                            # Add the comparison for the layer after the fault injection
                            if fault_layer_index < len(smart_modules_names) - 1 and smart_module.layer_name == smart_modules_names[fault_layer_index + 1]:
                                smart_module.compare_with_golden()

                            # Remove the comparison with golden for all the layer previous to the computation of the faulty
                            # layer
                            else:
                                smart_module.do_not_compare_with_golden()

                    if fault_delayed_start:
                        # Do this only if the fault is injected inside one of the layer that allow delayed start
                        if '._SmartModule__module' in fault.layer_name:

                            # The module where delayed start is enabled
                            if delayed_start_module is None:
                                delayed_start_module = self.network

                            # Get the name of the first-tier layer containing the module where the fault is injected
                            # TODO: add the fact that fault.layer_name start with 'delayed_start_module.name + name'
                            starting_layer = [(name, children) for name, children in delayed_start_module.named_children()
                                              # if fault.layer_name.split('_SmartModule__module.')[-1].startswith(name)][0]
                                              # if fault.layer_name.startswith(name)][0]
                                              if name in fault.layer_name][0]

                            delayed_start_module.starting_layer = starting_layer[1]

                            # Select the first smart module inside the faulty first-tier layer
                            delayed_start_module.starting_module = [module for module in self.__smart_modules_list
                                                                    if module in [m for m in delayed_start_module.starting_layer.modules()]][0]
                                                                    # if starting_layer[0] in module.layer_name][0]

                            assert delayed_start_module.starting_module in [m for m in delayed_start_module.starting_layer.modules()]
                        else:
                            # If the fault is injected in a non-smart layer, then starting_layer and starting_module
                            # should be None
                            delayed_start_module.starting_layer = None
                            delayed_start_module.starting_module = None

                    # Inject faults in the weight
                    self.__inject_fault_on_weight(fault, fault_mode='stuck-at')

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
                        faulty_scores = batch_clean_prediction_scores
                        faulty_indices = batch_clean_prediction_indices

                    faulty_prediction_dict[fault_id] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Measure the loss in accuracy
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.4f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

                    # Restore the golden value
                    self.weight_fault_injector.restore_golden()

                    # Increment the iteration count
                    total_iterations += 1

                # Print results to file
                formatted_print(fault_list=fault_list,
                                batch_size=self.loader.batch_size,
                                batch_id=batch_id,
                                network_name=self.network_name,
                                faulty_prediction_dict=faulty_prediction_dict,
                                fault_dropping=fault_dropping,
                                fault_delayed_start=fault_delayed_start)

                # End after only one batch if the option is specified
                if first_batch_only:
                    break

                # Remove all the loaded golden output feature map
                if fault_dropping:
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

            faulty_prediction_scores = [float(fault) for fault in faulty_prediction.values]
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
                                 fault_mode='stuck-at'):
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
