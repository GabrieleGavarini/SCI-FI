import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm


class FaultInjectionExecutor:

    class NoChangeOFMException(Exception):
        """
        Exception thrown when the execution of a faulty neural network doesn't produce any difference between its own
        output feature map and the output feature map of a clean network execution
        """
        pass

    def __init__(self,
                 network: Module,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 ofm: dict):

        self.network = network
        self.loader = loader
        self.device = 'cuda'

        self.ofm = ofm
        self.clean_output = clean_output

        # The ofm currently loaded on the gpu
        self.ofm_gpu = None

        # The network truncated from a starting layer
        self.faulty_network = None

    def run_faulty_campaign_on_weight(self,
                                      fault_list: list,
                                      layer_name: str = None):
        """
        Run a faulty injection campaign for the network. If a layer name is specified, start the computation from that
        layer, loading the input feature maps of the previous layer
        :param fault_list: list of fault to inject
        :param layer_name:
        :return:
        """

        self.__run_complete_inference_on_weight(fault_list=fault_list)

    def __run_complete_inference_on_weight(self,
                                           fault_list: list):

        pbar = tqdm(self.loader, colour='green', desc='Fault injection campaign')

        different_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_id, batch in enumerate(pbar):
                data, _ = batch
                data = data.to(self.device)

                hook_handler = None
                # Inject all the faults
                for fault in fault_list:

                    layer = fault[0]

                    # Load the ofm on gpu
                    keys_to_load = [key for index, key in enumerate(self.ofm.keys())
                                    if index > list(self.ofm.keys()).index(layer)]
                    self.ofm_gpu = {key: values[batch_id].cuda() for key, values in self.ofm.items()
                                    if key in keys_to_load}

                    # For a fault in a layer, add the corresponding hook on that layer.
                    if hook_handler is not None:
                        hook_handler.remove()
                    # TODO: register the hook
                        # TODO: add the possibility to add an hook starting from the layer, and not only for that layer

                    self.__inject_fault_on_weight(fault)
                    different_predictions += self.__run_inference_on_batch(batch_id=batch_id,
                                                                           data=data)
                    total_predictions += len(batch)

                accuracy_loss = 100 * different_predictions / total_predictions
                pbar.set_postfix({'Accuracy_loss': accuracy_loss})

                # TODO: unload ofm from gpu
                print(f'Allocated memory: {torch.cuda.memory_allocated(0)}')
                self.ofm_gpu = None
                print(f'Allocated memory: {torch.cuda.memory_allocated(0)}')

    def __run_inference_on_batch(self,
                                 batch_id: int,
                                 data: torch.Tensor):
        try:
            # Execute the network on the batch
            network_output = self.network(data)

            faulty_prediction = torch.topk(network_output, k=1)
            clean_prediction = torch.topk(self.clean_output[batch_id], k=1)

            # Measure the different predictions
            different_predictions = torch.ne(faulty_prediction.indices, clean_prediction.indices).sum()

        except self.NoChangeOFMException:
            # If the fault doesn't change the output feature map, then simply say that the fault doesn't worsen the
            # network performances for this batch
            different_predictions = 0

        return different_predictions

    def __inject_fault_on_weight(self, fault):
        # TODO: implement this
        return
