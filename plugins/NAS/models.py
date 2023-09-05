from typing import cast
import torch
import torch.nn as nn
from plinio.methods.pit_supernet import PITSuperNetModule
import copy
from typing import Dict, Any, Optional
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, gumbel = False):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                         padding=padding, bias=False)

        self.conv1 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, padding=2 * padding, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, groups=in_channels,
                          padding=padding, stride=stride),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # nn.init.kaiming_normal_(self.conv1.weight)
        it = self.conv1[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv1[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv1[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        # return self.relu(self.bn(x))
        return x


class ResNet8PITSN(torch.nn.Module):
    def __init__(self, gumbel = False):
        super().__init__()

        # Resnet v1 parameters
        self.input_shape = [3, 32, 32]  # default size for cifar10
        self.num_classes = 10  # default class number for cifar10
        self.num_filters = 16  # this should be 64 for an official resnet nas_name

        # Resnet v1 layers

        # First stack
        self.inputblock = ConvBlock(in_channels=3, out_channels=16,
                                    kernel_size=3, stride=1, padding=1, gumbel = gumbel)
        self.convblock1 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=1, padding=1, gumbel = gumbel)

        # self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(16, 16, 3, padding='same'),
                nn.BatchNorm2d(16),
            ),
            nn.Sequential(
                nn.Conv2d(16, 16, 5, padding='same'),
                nn.BatchNorm2d(16),
            ),
            nn.Sequential(
                nn.Conv2d(16, 16, 3, groups=16, padding='same'),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, 1),
                nn.BatchNorm2d(16),
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # nn.init.kaiming_normal_(self.conv1.weight)
        it = self.conv1[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv1[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv1[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second stack
        self.convblock2 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=3, stride=2, padding=1, gumbel = gumbel)

        # self.conv2y = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2y = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                nn.BatchNorm2d(32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 5, padding='same'),
                nn.BatchNorm2d(32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, groups=32, padding='same'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1),
                nn.BatchNorm2d(32)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # nn.init.kaiming_normal_(self.conv2y.weight)
        it = self.conv2y[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv2y[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv2y[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn2 = nn.BatchNorm2d(32)

        self.conv2x = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0)
        self.bn2x = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv2x.weight)

        # Third stack
        self.convblock3 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=3, stride=2, padding=1, gumbel = gumbel)

        # self.conv3y = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3y = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, groups=64, padding='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # nn.init.kaiming_normal_(self.conv3y.weight)
        it = self.conv3y[0].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        it = self.conv3y[1].children()
        nn.init.kaiming_normal_(cast(nn.Conv2d, next(it)).weight)
        children = list(self.conv3y[2].children())
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[0]).weight)
        nn.init.kaiming_normal_(cast(nn.Conv2d, children[3]).weight)
        # self.bn3 = nn.BatchNorm2d(64)

        self.conv3x = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0)
        self.bn3x = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv3x.weight)

        self.avgpool = torch.nn.AvgPool2d(8)

        self.out = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):
        # Input layer
        x = self.inputblock(input)  # [32, 32, 16]

        # First stack
        y = self.convblock1(x)      # [32, 32, 16]
        y = self.conv1(y)
        # y = self.bn1(y)
        x = torch.add(x, y)         # [32, 32, 16]
        x = self.relu(x)

        # Second stack
        y = self.convblock2(x)      # [16, 16, 32]
        y = self.conv2y(y)
        # y = self.bn2(y)
        x = self.conv2x(x)          # [16, 16, 32]
        x = self.bn2x(x)
        x = torch.add(x, y)         # [16, 16, 32]
        x = self.relu(x)

        # Third stack
        y = self.convblock3(x)      # [8, 8, 64]
        y = self.conv3y(y)
        # y = self.bn3(y)
        x = self.conv3x(x)          # [8, 8, 64]
        x = self.bn3x(x)
        x = torch.add(x, y)         # [8, 8, 64]
        x = self.relu(x)

        x = self.avgpool(x)         # [1, 1, 64]
        # x = torch.squeeze(x)        # [64]
        x = torch.flatten(x, 1)
        x = self.out(x)             # [10]

        return x

def _adapt_resnet18_statedict(pretrained_sd, model_sd, skip_inp=False):
    new_dict = copy.deepcopy(model_sd)
    for (item_pretr, item_mdl) in zip(pretrained_sd.items(), new_dict.items()):
        if skip_inp:
            skip_inp = False
            continue
        if 'fc' in item_pretr[0] and 'fc' in item_mdl[0]:
            continue
        new_dict[item_mdl[0]] = item_pretr[1]
    return new_dict

def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None
                        ) -> nn.Module:
    if model_name == 'resnet_18':
        pretrained = model_config.get('pretrained', True)
        state_dict = model_config.get('state_dict', None)
        std_head = model_config.get('std_head', False)
        model = ResNet18(std_head=std_head)
        if pretrained and state_dict is not None:
            new_state_dict = _adapt_resnet18_statedict(
                state_dict, model.state_dict(), skip_inp=True)
            model.load_state_dict(new_state_dict, strict=False)
        return model
    else:
        raise ValueError(f"Unsupported model name {model_name}")

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if inplanes != planes:
            self.conv1 = PITSuperNetModule([
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, padding=1, stride=stride),
                    nn.BatchNorm2d(planes)
                ),
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 5, padding=2, stride=stride),
                    nn.BatchNorm2d(planes)
                ),
                nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, padding=1, stride=stride),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, planes, 1),
                    nn.BatchNorm2d(planes)
                )
            ], gumbel_softmax = True, hard_softmax = True)
        else:
            self.conv1 = PITSuperNetModule([
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, padding=1, stride=stride),
                    nn.BatchNorm2d(planes)
                ),
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 5, padding=2, stride=stride),
                    nn.BatchNorm2d(planes)
                ),
                nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, padding=1, stride=stride),
                    nn.BatchNorm2d(inplanes),
                    nn.ReLU(),
                    nn.Conv2d(inplanes, planes, 1),
                    nn.BatchNorm2d(planes)
                ),
                nn.Identity()
            ], gumbel_softmax = True, hard_softmax = True)
        self.conv2 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(planes, planes, 3, padding='same', stride=1),
                nn.BatchNorm2d(planes)
            ),
            nn.Sequential(
                nn.Conv2d(planes, planes, 5, padding='same', stride=1),
                nn.BatchNorm2d(planes)
            ),
            nn.Sequential(
                nn.Conv2d(planes, planes, 3, groups=planes, padding='same', stride=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(),
                nn.Conv2d(planes, planes, 1),
                nn.BatchNorm2d(planes)
            ),
            nn.Identity()
        ], gumbel_softmax = True, hard_softmax = True)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1,
                                        stride=stride, bias=False)
            self.bn_ds = nn.BatchNorm2d(planes)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.bn_ds(residual)
        out = out + residual
        # return out
        return F.relu(out)


class Backbone18(nn.Module):
    def __init__(self, std_head):
        super(Backbone18, self).__init__()
        self.std_head = std_head
        if std_head:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bb_1_0 = BasicBlock(64, 64, stride=1)
        self.bb_1_1 = BasicBlock(64, 64, stride=1)
        self.bb_2_0 = BasicBlock(64, 128, stride=2)
        self.bb_2_1 = BasicBlock(128, 128, stride=1)
        self.bb_3_0 = BasicBlock(128, 256, stride=2)
        self.bb_3_1 = BasicBlock(256, 256, stride=1)
        self.bb_4_0 = BasicBlock(256, 512, stride=2)
        self.bb_4_1 = BasicBlock(512, 512, stride=1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        if std_head:
            self.avg_pool = nn.AvgPool2d(kernel_size=(7,7))
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=(8,8))

    def forward(self, x):
        if self.std_head:
            x = self.max_pool(x)
        x = self.bb_1_0(x)
        x = self.bb_1_1(x)
        x = self.bb_2_0(x)
        x = self.bb_2_1(x)
        x = self.bb_3_0(x)
        x = self.bb_3_1(x)
        x = self.bb_4_0(x)
        out = self.bb_4_1(x)
        out = self.avg_pool(out)
        return out


class ResNet18PITSN(nn.Module):
    def __init__(self, input_size=64, num_classes=200, std_head=False, **kwargs):
        super(ResNet18PITSN, self).__init__()

        # ResNet18 parameters
        self.input_shape = [3, input_size, input_size]
        self.num_classes = 200
        self.inplanes = 64

        # Model
        if std_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False,
                                   padding=3)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False,
                                   padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.backbone = Backbone18(std_head=std_head)

        # Initialize bn and conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # Final classifier
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class DSCnnSN(torch.nn.Module):
    def __init__(self, gumbel = False):
        super().__init__()

        # Model layers

        # Input pure conv2d
        self.inputlayer = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1))
        self.bn = nn.BatchNorm2d(64, momentum=0.99)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        # First layer of separable depthwise conv2d
        # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
        '''
        self.depthwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint1 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # self.bn11 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu11 = nn.ReLU()

        # Second layer of separable depthwise conv2d
        '''
        self.depthwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint2 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # self.bn21 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu21 = nn.ReLU()

        # Third layer of separable depthwise conv2d
        '''
        self.depthwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint3 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # self.bn31 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu31 = nn.ReLU()

        # Fourth layer of separable depthwise conv2d
        '''
        self.depthwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pointwise4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint4 = PITSuperNetModule([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64, momentum=0.99),
                nn.ReLU()
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)
        # self.bn41 = nn.BatchNorm2d(64, momentum=0.99)
        # self.relu41 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.4)
        self.avgpool = torch.nn.AvgPool2d((25, 5))
        self.out = nn.Linear(64, 12)

    def forward(self, input):

        # Input pure conv2d
        x = self.inputlayer(input)
        x = self.dropout1(self.relu(self.bn(x)))

        # First layer of separable depthwise conv2d
        # x = self.depthwise1(x)
        # x = self.pointwise1(x)
        x = self.depthpoint1(x)
        # x = self.relu11(self.bn11(x))

        # Second layer of separable depthwise conv2d
        # x = self.depthwise2(x)
        # x = self.pointwise2(x)
        x = self.depthpoint2(x)
        # x = self.relu21(self.bn21(x))

        # Third layer of separable depthwise conv2d
        # x = self.depthwise3(x)
        # x = self.pointwise3(x)
        x = self.depthpoint3(x)
        # x = self.relu31(self.bn31(x))

        # Fourth layer of separable depthwise conv2d
        # x = self.depthwise4(x)
        # x = self.pointwise4(x)
        x = self.depthpoint4(x)
        # x = self.relu41(self.bn41(x))

        x = self.dropout2(x)
        x = self.avgpool(x)
        # x = torch.squeeze(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.out(x)

        return x


class ConvBlock_MN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, kernel_size=3, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False,
                               groups=groups)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class MobileNetSN(torch.nn.Module):
    def __init__(self, gumbel = False):
        super().__init__()

        # MobileNet v1 parameters
        self.input_shape = [3, 96, 96]  # default size for coco dataset
        self.num_classes = 2  # binary classification: person or non person
        self.num_filters = 8

        # MobileNet v1 layers

        # 1st layer
        self.inputblock = nn.Conv2d(3,
                               8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False,
                               groups=1)
        nn.init.kaiming_normal_(self.inputblock.weight)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()


        # 2nd layer
        self.depthpoint2 = PITSuperNetModule([
            ConvBlock_MN(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
                          groups=8),
                ConvBlock_MN(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 3d layer
        '''
        self.depthwise3 = ConvBlock(in_channels=16, out_channels=16,
                                    kernel_size=3, stride=2, padding=1, groups=16)
        self.pointwise3 = ConvBlock(in_channels=16, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint3 = PITSuperNetModule([
            ConvBlock_MN(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            ConvBlock_MN(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock_MN(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1,
                          groups=16),
                ConvBlock_MN(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 4th layer
        '''
        self.depthwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise4 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=1, stride=1, padding=0)
        '''

        self.depthpoint4 = PITSuperNetModule([
            ConvBlock_MN(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                          groups=32),
                ConvBlock_MN(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 5h layer
        '''
        self.depthwise5 = ConvBlock(in_channels=32, out_channels=32,
                                    kernel_size=3, stride=2, padding=1, groups=32)
        self.pointwise5 = ConvBlock(in_channels=32, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint5 = PITSuperNetModule([
            ConvBlock_MN(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            ConvBlock_MN(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock_MN(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                          groups=32),
                ConvBlock_MN(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 6th layer
        '''
        self.depthwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=1, padding=1, groups=64)
        self.pointwise6 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint6 = PITSuperNetModule([
            ConvBlock_MN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                          groups=64),
                ConvBlock_MN(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 7th layer
        '''
        self.depthwise7 = ConvBlock(in_channels=64, out_channels=64,
                                    kernel_size=3, stride=2, padding=1, groups=64)
        self.pointwise7 = ConvBlock(in_channels=64, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint7 = PITSuperNetModule([
            ConvBlock_MN(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            ConvBlock_MN(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock_MN(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1,
                          groups=64),
                ConvBlock_MN(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 8th layer
        '''
        self.depthwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise8 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint8 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 9th layer
        '''
        self.depthwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise9 = ConvBlock(in_channels=128, out_channels=128,
                                    kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint9 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 10th layer
        '''
        self.depthwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise10 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint10 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 11th layer
        '''
        self.depthwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise11 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint11 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 12th layer
        '''
        self.depthwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=1, padding=1, groups=128)
        self.pointwise12 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint12 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 13th layer
        '''
        self.depthwise13 = ConvBlock(in_channels=128, out_channels=128,
                                     kernel_size=3, stride=2, padding=1, groups=128)
        self.pointwise13 = ConvBlock(in_channels=128, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint13 = PITSuperNetModule([
            ConvBlock_MN(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            ConvBlock_MN(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=2),
            nn.Sequential(
                ConvBlock_MN(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                          groups=128),
                ConvBlock_MN(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
            ),
            # nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        # 14th layer
        '''
        self.depthwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=3, stride=1, padding=1, groups=256)
        self.pointwise14 = ConvBlock(in_channels=256, out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        '''
        self.depthpoint14 = PITSuperNetModule([
            ConvBlock_MN(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            ConvBlock_MN(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding='same'),
            nn.Sequential(
                ConvBlock_MN(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1,
                          groups=256),
                ConvBlock_MN(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
            ),
            nn.Identity()
        ], gumbel_softmax = gumbel, hard_softmax = gumbel)

        self.avgpool = torch.nn.AvgPool2d(3)

        self.out = nn.Linear(256, 2)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, input):

        # Input tensor shape        # [96, 96,  3]

        # 1st layer
        x = self.relu(self.bn(self.inputblock(input)))  # [48, 48,  8]

        # 2nd layer
        # x = self.depthwise2(x)      # [48, 48,  8]
        # x = self.pointwise2(x)      # [48, 48, 16]
        x = self.depthpoint2(x)

        # 3rd layer
        # x = self.depthwise3(x)      # [24, 24, 16]
        # x = self.pointwise3(x)      # [24, 24, 32]
        x = self.depthpoint3(x)

        # 4th layer
        # x = self.depthwise4(x)      # [24, 24, 32]
        # x = self.pointwise4(x)      # [24, 24, 32]
        x = self.depthpoint4(x)

        # 5th layer
        # x = self.depthwise5(x)      # [12, 12, 32]
        # x = self.pointwise5(x)      # [12, 12, 64]
        x = self.depthpoint5(x)

        # 6th layer
        # x = self.depthwise6(x)      # [12, 12, 64]
        # x = self.pointwise6(x)      # [12, 12, 64]
        x = self.depthpoint6(x)

        # 7th layer
        # x = self.depthwise7(x)      # [ 6,  6, 64]
        # x = self.pointwise7(x)      # [ 6,  6, 128]
        x = self.depthpoint7(x)

        # 8th layer
        # x = self.depthwise8(x)      # [ 6,  6, 128]
        # x = self.pointwise8(x)      # [ 6,  6, 128]
        x = self.depthpoint8(x)

        # 9th layer
        # x = self.depthwise9(x)      # [ 6,  6, 128]
        # x = self.pointwise9(x)      # [ 6,  6, 128]
        x = self.depthpoint9(x)

        # 10th layer
        # x = self.depthwise10(x)     # [ 6,  6, 128]
        # x = self.pointwise10(x)     # [ 6,  6, 128]
        x = self.depthpoint10(x)

        # 11th layer
        # x = self.depthwise11(x)     # [ 6,  6, 128]
        # x = self.pointwise11(x)     # [ 6,  6, 128]
        x = self.depthpoint11(x)

        # 12th layer
        # x = self.depthwise12(x)     # [ 6,  6, 128]
        # x = self.pointwise12(x)     # [ 6,  6, 128]
        x = self.depthpoint12(x)

        # 13th layer
        # x = self.depthwise13(x)     # [ 3,  3, 128]
        # x = self.pointwise13(x)     # [ 3,  3, 256]
        x = self.depthpoint13(x)

        # 14th layer
        # x = self.depthwise14(x)     # [ 3,  3, 256]
        # x = self.pointwise14(x)     # [ 3,  3, 256]
        x = self.depthpoint14(x)

        x = self.avgpool(x)         # [ 1,  1, 256]
        x = torch.flatten(x, start_dim=1)   # [256]
        x = self.out(x)             # [2]

        return x
