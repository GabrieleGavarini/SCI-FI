import argparse
from typing import Tuple

import torch
import struct


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # NETWORK
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        choices=['LeNet5', 'LeNet5_MNIST',
                                 'ResNet18', 'ResNet50', 'ResNet50_GTSRB',
                                 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121',
                                 'EfficientNet_B0', 'EfficientNet_B4', 'EfficientNet_B4_GTSRB'])
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')

    # FAULT MODEL
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault model used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])
    parser.add_argument('--fault_percentage', type=float, default=None,
                        help='If the fault model is stuck-at params, how many fault to inject in a single inference in'
                             'percentage of the total number of network\'s neurons')

    parsed_args = parser.parse_args()

    return parsed_args

def int_bit_flip(golden_value,
                 bit) -> Tuple[int, int]:
    """
    Injects a bit-flip on data represented as either torch.int8 or torch.int32.

    :param golden_value: The value to corrupt (torch.int8 or torch.int32).
    :param bit: The bit where to apply the bit-flip.
    :return: The value of the bit-flip on the golden value.
    """
    # Check if the data_type is valid
    if golden_value.dtype != torch.int8 and golden_value.dtype != torch.qint8 and golden_value.dtype != torch.int32:
        raise ValueError("Invalid data_type. Supported values: torch.int8 or torch.int32.")

    # Determine the format string based on the data_type parameter
    format_str = '!b' if golden_value.dtype in [torch.int8, torch.qint8] else '!i'

    # Determine the format string for fault based on the chosen format_str
    fault_format_str = '!B' if format_str == '!b' else '!I'

    # Pack the golden value using the chosen format string
    golden_binary = struct.pack(format_str, torch.int_repr(golden_value))
    faulted_bit = int(''.join('{:0>8b}'.format(c) for c in golden_binary)[len(golden_binary)*8 - 1 - bit])

    # Create the fault representation by packing the bit-flip
    fault = struct.pack(fault_format_str, 1 << bit)

    # Perform the bit-flip by XORing the corresponding bytes
    faulty_binary = bytearray()
    for ba, bb in zip(golden_binary, fault):
        faulty_binary.append(ba ^ bb)

    # Unpack the faulted value using the chosen format string
    faulted_value = struct.unpack(format_str, faulty_binary)[0]

    return faulted_value, faulted_bit


def float_bit_flip(golden_value,
                   bit) -> Tuple[float, int]:
    """
    :param golden_value: The value to corrupt (torch.int8 or torch.int32).
    :param bit: The bit where to apply the bit-flip.
    :return: The value of the bit-flip on the golden value.
    """

    float_list = []
    a = struct.pack('!f', golden_value)
    faulted_bit = int(''.join('{:0>8b}'.format(c) for c in a)[len(a)*8 - 1 - bit])
    b = struct.pack('!I', int(2. ** bit))
    for ba, bb in zip(a, b):
        float_list.append(ba ^ bb)

    faulted_value = struct.unpack('!f', bytes(float_list))[0]

    return faulted_value, faulted_bit
