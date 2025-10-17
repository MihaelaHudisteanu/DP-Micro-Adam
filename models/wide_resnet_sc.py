#code taken from https://github.com/meliketoy/wide-resnet.pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import math


class WSConv2d1(nn.Conv2d):
    def reset_parameters(self):
        fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        nn.init.normal_(self.weight, 0.0, fan_in ** -0.5)   # var = 1/fan_in
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = self.weight
        mean = w.mean(dim=(1,2,3), keepdim=True)
        var  = w.var(dim=(1,2,3), unbiased=False, keepdim=True)
        w = (w - mean) / torch.sqrt(var + 1e-5)

        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        w = w * (fan_in ** -0.5)  # sqrt(1/fan_in) 

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.gain = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Fan-in scaled initialization to match JAX VarianceScaling(1.0)
        fan_in = in_channels * kernel_size * kernel_size
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))

    def forward(self, x, eps=1e-4):
        w = self.weight
        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        var = w.var(dim=(1, 2, 3), unbiased=False, keepdim=True)

        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        

        scale = torch.rsqrt(torch.clamp(var * fan_in, min=eps)) * self.gain[:, None, None, None]
        shift = mean * scale
        w_standardized = w * scale - shift

        return F.conv2d(x, w_standardized, self.bias, stride=self.stride, 
                       padding=self.padding, dilation=self.dilation, groups=self.groups)

def norm(num_channels, groups=16):
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)

def conv3x3(in_planes, out_planes, stride=1):
    return WSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    return WSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, groups=16):
        super(wide_basic, self).__init__()
        self.bn1 = norm(in_planes, groups=groups)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = norm(planes, groups)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ReLU(inplace=False),
                norm(in_planes, groups=groups),
                conv1x1(in_planes, planes, stride=stride),
            )

        # add scale norm after residual addition if enabled
        self.scale_norm = norm(planes, groups=groups)

    def forward(self, x):
        out = F.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)

        out = F.relu(out)  
        out = self.bn2(out)
        out = self.conv2(out)

        out = out + self.shortcut(x)

        out = self.scale_norm(out)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, groups=16, scale_norm=False):
        super().__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1, groups=groups)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2, groups=groups)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2, groups=groups)

        self.bn1 = norm(nStages[3], groups=groups)
        self.linear = nn.Linear(nStages[3], num_classes)

        # init linear
        nn.init.normal_(self.linear.weight, 0.0, self.linear.in_features ** -0.5) 
        if self.linear.bias is not None: 
            nn.init.zeros_(self.linear.bias)

    def _wide_layer(self, block, planes, num_blocks, stride, groups=16, scale_norm=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.bn1(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def wide_resnet_16_4_sc(**kwargs):
    model = Wide_ResNet(16, 4, **kwargs)
    return model

def wide_resnet_40_4_sc(**kwargs):
    model = Wide_ResNet(40, 4, **kwargs)
    return model    
