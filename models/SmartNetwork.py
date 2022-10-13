from abc import ABC, abstractmethod


class SmartNetwork(ABC):

    @abstractmethod
    def move_to_gpu(self,
                    batch_id: int):
        """
        Move to gpu the ofm of the current batch
        :param batch_id: The id of the batch to move to gpu
        """
        pass

    @abstractmethod
    def set_check_ofm(self,
                      check_ofm_dict: dict):
        """
        Set the check_ofm_dict for the network
        :param check_ofm_dict: The ofm of the network
        """
        pass
