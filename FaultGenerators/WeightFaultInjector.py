import copy
import struct

import torch

from FaultGenerators.utils import convert_index
from modules.utils import get_module_by_name


class WeightFaultInjector:

    def __init__(self,
                 network,
                 dtype: torch.dtype = torch.float):

        self.network = network

        self.layer_name = None
        self.tensor_index = None
        self.unflattened_tensor_index = None
        self.bit = None

        self.golden_value = None


    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit

        # This check is used only for quantized network
        # TODO: this is only valid for quantized fc layers
        if self.layer_name not in self.network.state_dict().keys():
            self.layer_name = self.layer_name.replace('weight', '_packed_params._packed_params')
            self.unflattened_tensor_index = convert_index(self.network.state_dict()[self.layer_name][0].shape, self.tensor_index)
            self.golden_value = copy.deepcopy(self.network.state_dict()[self.layer_name][0][self.unflattened_tensor_index])
        else:
            # Convert the 1d index to the 4d index
            self.unflattened_tensor_index = convert_index(self.network.state_dict()[self.layer_name].shape, self.tensor_index)
            self.golden_value = copy.deepcopy(self.network.state_dict()[self.layer_name][self.unflattened_tensor_index])


        # Manage data type
        if self.golden_value.dtype == torch.float:
            # If the value is not set, then we are doing a bit-flip
            if value is None:
                faulty_value = self.__float32_bit_flip()
            else:
                faulty_value = self.__float32_stuck_at(value)

            # For float, simply copy the values
            self.network.state_dict()[self.layer_name].view(-1)[self.tensor_index] = faulty_value
        elif self.golden_value.dtype == torch.qint8:
            # If the value is not set, then we are doing a bit-flip
            if value is None:
                faulty_value = self.__int_bit_flip(data_type=torch.int8)
            else:
                faulty_value = self.__qint8_stuck_at(value)

            # Get the faulty layer and weights
            faulty_layer = get_module_by_name(self.network,
                                              self.layer_name.replace('.weight', '').replace('._packed_params._packed_params', ''))
            faulty_weights = faulty_layer.weight()

            # Cast the faulty value to the right quantized value
            faulty_value = (faulty_value - self.golden_value.q_zero_point()) * self.golden_value.q_scale()

            # Modify the whole weight tensor
            faulty_weights[self.unflattened_tensor_index] = faulty_value

            # Assign the faulty weights to the faulty layer
            faulty_layer.set_weight_bias(w=faulty_weights, b=faulty_layer.bias())
        else:
            raise NotImplementedError(f'Fault injected not implemented for data represented as {self.golden_value.dtype}')

        self.faulty_value = faulty_value

    def __int_bit_flip(self,
                       data_type: torch.dtype) -> int:
        """
        Injects a bit-flip on data represented as either torch.int8 or torch.int32.

        :param data_type: The data type to represent the data (torch.int8 or torch.int32).
        :return: The value of the bit-flip on the golden value.
        """
        # Check if the data_type is valid
        if data_type != torch.int8 and data_type != torch.int32:
            raise ValueError("Invalid data_type. Supported values: torch.int8 or torch.int32.")

        # Determine the format string based on the data_type parameter
        format_str = '!b' if data_type == torch.int8 else '!i'

        # Determine the format string for fault based on the chosen format_str
        fault_format_str = '!B' if format_str == '!b' else '!I'

        # Pack the golden value using the chosen format string
        golden_binary = struct.pack(format_str, torch.int_repr(self.golden_value))

        # Create the fault representation by packing the bit-flip
        fault = struct.pack(fault_format_str, 1 << self.bit)

        # Perform the bit-flip by XORing the corresponding bytes
        faulty_binary = bytearray()
        for ba, bb in zip(golden_binary, fault):
            faulty_binary.append(ba ^ bb)

        # Unpack the faulted value using the chosen format string
        faulted_value = struct.unpack(format_str, faulty_binary)[0]

        return faulted_value


    def __qint8_stuck_at(self, value):
        """
        Inject a stuck-at on a data represented as torch.qint8
        :param value: The value of the stuck-at bit
        :return: The value of the bit-flip on the golden value
        """
        golden_binary = struct.pack('!b', torch.int_repr(self.golden_value))
        fault = struct.pack('!b', int(2. ** self.bit))

        faulty_binary = list()
        for ba, bb in zip(golden_binary, fault):
            if value == 1:
                faulty_binary.append(ba | bb)
            else:
                faulty_binary.append(ba & (255 - bb))

        faulted_value = struct.unpack('!b', bytes(faulty_binary))[0]
        return faulted_value

    def __float32_bit_flip(self):
        """
        Inject a bit-flip on a data represented as float32
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def __float32_stuck_at(self,
                           value: int):
        """
        Inject a stuck-at fault on a data represented as float32
        :param value: the value to use as stuck-at value
        :return: The value of the bit-flip on the golden value
        """
        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            if value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return faulted_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        if self.golden_value.dtype == torch.float:
            # For float, simply copy the values
            self.network.state_dict()[self.layer_name].view(-1)[self.tensor_index] = self.golden_value
        elif self.golden_value.dtype == torch.qint8:
            # Get the faulty layer
            faulty_layer = get_module_by_name(self.network, self.layer_name.replace('.weight', '').replace('._packed_params._packed_params', ''))

            # Modify the whole weight tensor
            faulty_weights = faulty_layer.weight()
            faulty_weights[self.unflattened_tensor_index] = self.golden_value

            # Assign the faulty weights to the faulty layer
            faulty_layer.set_weight_bias(w=faulty_weights, b=faulty_layer.bias())
        else:
            raise NotImplementedError(f'Fault injected not implemented for data represented as {self.golden_value.dtype}')

    def inject_bit_flip(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit)

    def inject_stuck_at(self,
                        layer_name: str,
                        tensor_index: tuple,
                        bit: int,
                        value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault
        :param value: The stuck-at value to set
        """
        self.__inject_fault(layer_name=layer_name,
                            tensor_index=tensor_index,
                            bit=bit,
                            value=value)
