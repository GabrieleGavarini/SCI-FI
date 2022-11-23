import shutil

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm

from FaultGenerators.WeightFaultInjector import WeightFaultInjector
from models.SmartLayers.utils import NoChangeOFMException
from utils import formatted_print

from typing import List

from models.SmartLayers.SmartConv2d import SmartConv2d


class FaultInjectionManager:

    def __init__(self,
                 network: Module,
                 network_name: str,
                 smart_convolutions: List[SmartConv2d],
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

        # The smart convolution layers
        self.__smart_convolutions = smart_convolutions


        # The number of total inferences and the number of skipped inferences
        self.skipped_inferences = 0
        self.total_inferences = 0

        # The weight fault injector
        self.weight_fault_injector = WeightFaultInjector(self.network)


    def run_faulty_campaign_on_weight(self,
                                      fault_list: list,
                                      fault_dropping: bool = True,
                                      first_batch_only: bool = False):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_list: list of fault to inject
        :param fault_dropping: Default True. Whether to drop fault or not
        :param first_batch_only: Default False. Debug parameter, if set run the fault injection campaign on the first
        batch only
        """

        self.skipped_inferences = 0
        self.total_inferences = 0

        total_different_predictions = 0
        total_predictions = 0

        with torch.no_grad():

            # Cycle all the batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, _ = batch
                data = data.to(self.device)

                faulty_prediction_dict = dict()
                batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

                if fault_dropping:
                    # Move the corresponding ofm to the gpu
                    for convolution in self.__smart_convolutions:
                        convolution.load_golden(batch_id=batch_id)

                # Inject all the faults in a single batch
                pbar = tqdm(fault_list, colour='green', desc=f'FI on b {batch_id}', ncols=shutil.get_terminal_size().columns * 2)
                for fault_id, fault in enumerate(pbar):

                    if fault_dropping:
                        # Set which ofm to check during the forward pass. Only check the ofm that come after the fault
                        for convolution in self.__smart_convolutions:
                            convolution.do_not_compare_with_golden()
                            if convolution.layer_name == fault.layer_name:
                                convolution.compare_with_golden()

                        # Change the description of the progress bar
                        pbar.set_description(f'FI (w/ drop) on b {batch_id}')
                        fault.layer_name = f'{fault.layer_name}._SmartConv2d__conv_layer'
                    else:
                        # Correct the layer name in case it has been changed
                        fault.layer_name = fault.layer_name.replace('._SmartConv2d__conv_layer', '')


                    # Inject faults in the weight
                    self.__inject_fault_on_weight(fault, fault_mode='stuck-at')

                    # Run inference on the current batch
                    faulty_prediction, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                             data=data)

                    # If fault prediction is None, the fault had no impact. Use golden predictions
                    if faulty_prediction is None:
                        faulty_prediction = batch_clean_prediction_indices

                    faulty_prediction_dict[fault_id] = faulty_prediction
                    total_different_predictions += different_predictions

                    # Measure the loss in accuracy
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.4f}%',
                                      'Skipped': f'{100*self.skipped_inferences/self.total_inferences:.2f}%'})

                    # Restore the golden value
                    self.weight_fault_injector.restore_golden()

                # Print results to file
                formatted_print(fault_list=fault_list,
                                batch_id=batch_id,
                                network_name=self.network_name,
                                faulty_prediction_dict=faulty_prediction_dict)

                # End after only one batch if the option is specified
                if first_batch_only:
                    break

                # Remove all the loaded golden output feature map
                if fault_dropping:
                    for convolution in self.__smart_convolutions:
                        convolution.unload_golden_ofm()


    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data)
            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions
            different_predictions = int(torch.ne(faulty_prediction.indices, clean_prediction.indices).sum())

            faulty_prediction_indices = [int(fault) for fault in faulty_prediction.indices]

        except NoChangeOFMException:
            # If the fault doesn't change the output feature map, then simply say that the fault doesn't worsen the
            # network performances for this batch
            faulty_prediction_indices = None
            different_predictions = 0
            self.skipped_inferences += 1

        self.total_inferences += 1

        return faulty_prediction_indices, different_predictions

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
