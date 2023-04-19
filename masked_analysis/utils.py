import argparse


def parse_args():
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description='Save information about the ofm of each layer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault model used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        choices=['LeNet5',
                                 'Smooth_ResNet20',
                                 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121',
                                 'EfficientNet'])
    parser.add_argument('--root_folder', '-r', type=str, default='..',
                        help='The root folder where to look for the output folder')

    parsed_args = parser.parse_args()

    return parsed_args
