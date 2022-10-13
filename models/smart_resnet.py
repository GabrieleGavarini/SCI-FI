import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock, ResNet

__all__ = ['ResNet', 'smart_resnet20', 'smart_resnet32', 'smart_resnet44', 'smart_resnet56', 'smart_resnet110', 'smart_resnet1202']

from utils import check_difference


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


class SmartResNet(ResNet):

    OFM = None

    def __init__(self, block, num_blocks, ofm_dict, ifm_dict, num_classes=10, threshold=0):
        super(SmartResNet, self).__init__(block=BasicBlock, num_blocks=num_blocks, num_classes=num_classes)

        self.check_ofm_dict = dict()

        self.ofm_dict = ofm_dict
        self.ifm_dict = ifm_dict

        self.ofm_gpu = None
        self.ifm_gpu = None

        self.in_planes = 16

        self.threshold = threshold

        self.layer1 = self._make_smart_layer(block, 'layer1', 16, num_blocks[0], stride=1, threshold=self.threshold)
        self.layer2 = self._make_smart_layer(block, 'layer2', 32, num_blocks[1], stride=2, threshold=self.threshold)
        self.layer3 = self._make_smart_layer(block, 'layer3', 64, num_blocks[2], stride=2, threshold=self.threshold)

    @staticmethod
    def _move_layer_to_gpu(layer, batch_id):
        for block in layer.children():
            block.move_to_gpu(batch_id)

    def move_to_gpu(self, batch_id):
        self._move_layer_to_gpu(self.layer1, batch_id)
        self._move_layer_to_gpu(self.layer2, batch_id)
        self._move_layer_to_gpu(self.layer3, batch_id)

        self.ofm_gpu = {key: value[batch_id].cuda() for key, value in self.ofm_dict.items() if 'layer' not in key}
        self.ifm_gpu = {key: value[batch_id].cuda() for key, value in self.ifm_dict.items() if 'layer' not in key}

    @staticmethod
    def _set_check_ofm_children(layer, check_ofm_dict):
        for block_id, block in enumerate(layer.children()):
            block.set_check_ofm({re.sub(r'[0-9].', '', key): value for key, value in check_ofm_dict.items() if key[0] == str(block_id)})

    def set_check_ofm(self, check_ofm_dict):
        self._set_check_ofm_children(self.layer1, {key.replace(r'layer1.', ''): value
                                                   for key, value in check_ofm_dict.items() if 'layer1' in key})
        self._set_check_ofm_children(self.layer2, {key.replace(r'layer2.', ''): value
                                                   for key, value in check_ofm_dict.items() if 'layer2' in key})
        self._set_check_ofm_children(self.layer3, {key.replace(r'layer3.', ''): value
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
