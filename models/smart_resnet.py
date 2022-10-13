import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock, ResNet

__all__ = ['ResNet', 'smart_resnet20', 'smart_resnet32', 'smart_resnet44', 'smart_resnet56', 'smart_resnet110', 'smart_resnet1202']

from models.utils import check_difference
from models.SmartNetwork import SmartNetwork


class SmartBasicBlock(BasicBlock):

    expansion = 1

    def __init__(self, ofm_dict, ifm_dict, in_planes, planes, stride=1, option='A', threshold=0):
        super(SmartBasicBlock, self).__init__(in_planes=in_planes,
                                              planes=planes,
                                              stride=stride,
                                              option=option)

        # A dictionary containing all the output feature maps, for all the possible batches
        self.ofm_dict = ofm_dict
        self.ifm_dict = ifm_dict

        # A dictionary containing all the output feature maps for a particular batch. They get loaded when move_to_gpu
        # gets called.
        self.ofm_gpu = None
        self.ifm_gpu = None

        # A dictionary of boolean where each key is a layer name and each value is a boolean value signaling whether a
        # layer should be checked for early stopping or not
        self.check_ofm_dict = None

        # The threshold for two ofm to be considered the same
        self.threshold = threshold

    def move_to_gpu(self, batch_id):
        # TODO: load from a file and not from main memory
        self.ofm_gpu = {key: value[batch_id].cuda() for key, value in self.ofm_dict.items()}
        self.ifm_gpu = {key: value[batch_id].cuda() for key, value in self.ifm_dict.items()}

    def set_check_ofm(self, check_ofm_dict):
        self.check_ofm_dict = check_ofm_dict

    def forward(self, x):
        # First conv output
        out = self.conv1(x)
        check_difference(check_control=self.check_ofm_dict['conv1'],
                         golden=self.ofm_gpu['conv1'],
                         faulty=out,
                         threshold=self.threshold)

        # First batch normalization output
        out = self.bn1(out)
        check_difference(check_control=self.check_ofm_dict['bn1'],
                         golden=self.ofm_gpu['bn1'],
                         faulty=out,
                         threshold=self.threshold)
        out = F.relu(out)

        # Second conv output
        out = self.conv2(out)
        check_difference(check_control=self.check_ofm_dict['conv2'],
                         golden=self.ofm_gpu['conv2'],
                         faulty=out,
                         threshold=self.threshold)

        # Second batch normalization output
        out = self.bn2(out)
        check_difference(check_control=self.check_ofm_dict['bn2'],
                         golden=self.ofm_gpu['bn2'],
                         faulty=out,
                         threshold=self.threshold)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmartResNet(ResNet, SmartNetwork):

    OFM = None

    def __init__(self, block, num_blocks, ofm_dict, ifm_dict, num_classes=10, threshold=0):
        super(SmartResNet, self).__init__(block=BasicBlock, num_blocks=num_blocks, num_classes=num_classes)

        # A dictionary where every key is a layer name and the corresponding value is a bool that specifies whether the
        # output of that layer should be confronted with the golden output
        self.check_ofm_dict = dict()

        # Dict with the input feature map and the output feature maps of each injectable layer. For each layer, the dict
        # contains a list of the ofm for each batch
        # TODO: this should be saved to a file and not kept in main memory
        self.ofm_dict = ofm_dict
        self.ifm_dict = ifm_dict

        # The input feature maps and the output feature maps of the current batch that are loaded in video memory
        self.ofm_gpu = None
        self.ifm_gpu = None

        # ResNet parameters
        self.in_planes = 16

        # The threshold s.t. if |golden - faulty| < threshold => golden == faulty
        self.threshold = threshold

        # ResNet parameters
        self.layer1 = self._make_smart_layer(block, 'layer1', 16, num_blocks[0], stride=1, threshold=self.threshold)
        self.layer2 = self._make_smart_layer(block, 'layer2', 32, num_blocks[1], stride=2, threshold=self.threshold)
        self.layer3 = self._make_smart_layer(block, 'layer3', 64, num_blocks[2], stride=2, threshold=self.threshold)

    @staticmethod
    def __move_layer_to_gpu(layer: torch.nn.Module,
                            batch_id: int):
        """
        Move to gpu all the ofm of the blocks inside a layer for the current batch
        :param layer:  The current layer
        :param batch_id: The id of the batch to move to gpu
        """
        for block in layer.children():
            block.move_to_gpu(batch_id)

    def move_to_gpu(self,
                    batch_id: int):
        """
        Move to gpu the ofm of the current batch
        :param batch_id: The id of the batch to move to gpu
        """
        self.__move_layer_to_gpu(self.layer1, batch_id)
        self.__move_layer_to_gpu(self.layer2, batch_id)
        self.__move_layer_to_gpu(self.layer3, batch_id)

        # TODO: this should load from a file and not from memory
        self.ofm_gpu = {key: value[batch_id].cuda() for key, value in self.ofm_dict.items() if 'layer' not in key}
        self.ifm_gpu = {key: value[batch_id].cuda() for key, value in self.ifm_dict.items() if 'layer' not in key}

    @staticmethod
    def __set_check_ofm_children(layer: torch.nn.Module,
                                 check_ofm_dict: dict):
        """
        Set the check_ofm_dict for all the blocks inside a layer
        :param layer: The current layer
        :param check_ofm_dict: The ofm of the layer
        """
        for block_id, block in enumerate(layer.children()):
            block.set_check_ofm({re.sub(r'[0-9]+.', '', key): value for key, value in check_ofm_dict.items() if re.findall(r'[0-9]+.', key)[0] == f'{block_id}.'})

    def set_check_ofm(self,
                      check_ofm_dict: dict):
        """
        Set the check_ofm_dict for the network
        :param check_ofm_dict: The ofm of the network
        """
        self.__set_check_ofm_children(self.layer1, {key.replace(r'layer1.', ''): value
                                                   for key, value in check_ofm_dict.items() if 'layer1' in key})
        self.__set_check_ofm_children(self.layer2, {key.replace(r'layer2.', ''): value
                                                   for key, value in check_ofm_dict.items() if 'layer2' in key})
        self.__set_check_ofm_children(self.layer3, {key.replace(r'layer3.', ''): value
                                                   for key, value in check_ofm_dict.items() if 'layer3' in key})

        self.check_ofm_dict = check_ofm_dict


    def _make_smart_layer(self, block, layer_name, planes, num_blocks, stride, threshold):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride_id, stride in enumerate(strides):
            ofm_dict = {key.replace(f'{layer_name}.{stride_id}.', ''): value for key, value in self.ofm_dict.items()
                        if f'{layer_name}.{stride_id}.' in key}
            ifm_dict = {key.replace(f'{layer_name}.{stride_id}.', ''): value for key, value in self.ifm_dict.items()
                        if f'{layer_name}.{stride_id}.' in key}
            layers.append(block(ofm_dict, ifm_dict, self.in_planes, planes, stride, threshold=threshold))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # First conv output
        out = self.conv1(x)
        check_difference(check_control=self.check_ofm_dict['conv1'],
                         golden=self.ofm_gpu['conv1'],
                         faulty=out,
                         threshold=self.threshold)

        # First batch normalization output
        out = self.bn1(out)
        check_difference(check_control=self.check_ofm_dict['bn1'],
                         golden=self.ofm_gpu['bn1'],
                         faulty=out,
                         threshold=self.threshold)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def smart_resnet20(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[3, 3, 3],
                       threshold=threshold)


def smart_resnet32(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[5, 5, 5],
                       threshold=threshold)


def smart_resnet44(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[7, 7, 7],
                       threshold=threshold)


def smart_resnet56(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[9, 9, 9],
                       threshold=threshold)


def smart_resnet110(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[18, 18, 18],
                       threshold=threshold)


def smart_resnet1202(ifm_dict, ofm_dict, threshold=0):
    return SmartResNet(block=SmartBasicBlock,
                       ifm_dict=ifm_dict,
                       ofm_dict=ofm_dict,
                       num_blocks=[200, 200, 200],
                       threshold=threshold)
