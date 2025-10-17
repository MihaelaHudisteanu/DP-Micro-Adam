import torch.nn as nn
import math


__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56']

NUM_CLASSES = 10

def norm(num_channels, groups=16):
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)  

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = norm(planes) 
        self.relu1 = nn.ReLU(inplace=False) 
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = norm(planes) 
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.gn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        self.layer_gates = []
        for layer in range(3):
            self.layer_gates.append([])  
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.gn1 = norm(self.inplanes) 
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model
