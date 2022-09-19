import torch

class FaultInjectionExecutor():

    def __init__(self, network):

        self.network = network

    def run_faulty_campaign(self,
                            layer_name: str = None):
        """
        Run a faulty injection campaign for the network. If a layer name is specified,
        :param layer_name:
        :return:
        """
        return

    def truncate_network_from(self, layer_name):
        return
