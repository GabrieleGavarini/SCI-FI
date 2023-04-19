from typing import List

class NeuronFault:

    def __init__(self,
                 layer_name: str,
                 layer_index: int,
                 feature_map_indices: List[tuple],
                 value_list: List[float]):

        self.layer_name = layer_name

        self.layer_index = layer_index
        self.feature_map_indices = feature_map_indices
        self.value_list = value_list
