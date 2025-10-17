import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WSConv2d(nn.Conv2d):
    def reset_parameters(self):
        nn.init.normal_(self.weight, 0.0, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        mean = w.mean(dim=(1,2,3), keepdim=True)
        var  = w.var(dim=(1,2,3), unbiased=False, keepdim=True)
        w = (w - mean) / torch.sqrt(var + 1e-5) 

        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        w = w * (fan_in ** -0.5)  # sqrt(1/fan_in) 

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1):
    return WSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return WSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class ScaledReLU(nn.Module):
    def __init__(self, gamma=1.713958859):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        return self.gamma * F.relu(x, inplace=False)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class SqueezeExcite(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        hidden = max(1, int(round(channels * se_ratio)))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.avg(x)
        h = F.relu(self.fc1(h), inplace=False)
        h = torch.sigmoid(self.fc2(h))
        return h

class NFResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, *,
                 bottleneck_ratio=0.25,
                 alpha=0.2,
                 beta=1.0,
                 stochdepth_rate=0.0,
                 use_se=False,
                 se_ratio=0.25,
                 activation=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.width = int(out_ch * bottleneck_ratio)
        self.stride = stride
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.act   = activation if activation is not None else ScaledReLU()

        # residual path (all WS)
        self.conv0 = conv1x1(in_ch, self.width, stride=1)
        self.conv1 = conv3x3(self.width, self.width, stride=self.stride)
        self.conv2 = conv1x1(self.width, out_ch, stride=1)

        # projection on shortcut if channels/stride change
        self.use_projection = (self.stride > 1) or (in_ch != out_ch)
        self.conv_shortcut = conv1x1(in_ch, out_ch, stride=self.stride) if self.use_projection else None

        # SE (optional)
        self.use_se = bool(use_se)
        self.se = SqueezeExcite(out_ch, se_ratio) if self.use_se else None

        # stochastic depth
        self.drop_path = DropPath(stochdepth_rate) if (0.0 < stochdepth_rate < 1.0) else nn.Identity()

        # SkipInit scalar
        self.skip_gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.act(x) * self.beta

        # shortcut from the pre-activated path (matches JAX)
        shortcut = self.conv_shortcut(out) if self.use_projection else x

        # residual
        out = self.conv0(out)
        out = self.conv1(self.act(out))
        out = self.conv2(self.act(out))

        if self.use_se:
            gate = self.se(out)
            out = 2.0 * gate * out

        out = self.drop_path(out)
        out = out * self.skip_gain
        out = out * self.alpha + shortcut
        return out

NF_VARIANTS = {
    'ResNet50':  [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
    'ResNet200': [3, 24, 36, 3],
    'ResNet288': [24, 24, 24, 24],
    'ResNet600': [50, 50, 50, 50],
}

class NF_ResNet(nn.Module):
    def __init__(self, num_classes,
                 variant='ResNet50',
                 width=4,
                 alpha=0.2,
                 stochdepth_rate=0.1,
                 drop_rate=0.0,
                 use_se=False,
                 se_ratio=0.25,
                 activation=None):
        super().__init__()
        assert variant in NF_VARIANTS, f"Unknown variant {variant}"
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.drop_rate = float(drop_rate)
        self.act = activation if activation is not None else ScaledReLU()

        # ---- stem ----
        ch = int(16 * width)
        self.conv1 = WSConv2d(3, ch, kernel_size=7, stride=2, padding=3, bias=False)  # SAME 7x7
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- stages ----
        stage_widths = [64*width, 128*width, 256*width, 512*width]
        stage_depths = NF_VARIANTS[variant]
        stage_strides = [1, 2, 2, 2]

        blocks = []
        expected_std = 1.0
        num_blocks_total = sum(stage_depths)
        idx = 0
        in_ch = ch

        for out_ch, depth, stride in zip(stage_widths, stage_depths, stage_strides):
            for i in range(depth):
                beta = 1.0 / expected_std
                sd_rate = stochdepth_rate * (idx / num_blocks_total)
                bstride = stride if i == 0 else 1

                block = NFResBlock(in_ch, out_ch,
                                   stride=bstride,
                                   bottleneck_ratio=0.25,
                                   alpha=self.alpha,
                                   beta=beta,
                                   stochdepth_rate=sd_rate,
                                   use_se=use_se,
                                   se_ratio=se_ratio,
                                   activation=self.act)
                blocks.append(block)
                in_ch = out_ch
                idx += 1

                # reset expected_std at start of each stage, then grow once
                if i == 0:
                    expected_std = 1.0
                expected_std = math.sqrt(expected_std*expected_std + self.alpha*self.alpha)

        self.blocks = nn.Sequential(*blocks)

        # ---- head ----
        self.dropout = nn.Dropout(p=self.drop_rate) if self.drop_rate > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

        # init FC only; WSConv2d handles its own init
        nn.init.normal_(self.fc.weight, 0.0, 0.01)   # match hk.initializers.RandomNormal(0.01)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # stem
        out = self.conv1(x)
        out = self.pool1(out)

        # body
        out = self.blocks(out)

        # head
        out = self.act(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def nf_resnet_50(num_classes,
                 width=4,
                 alpha=0.2,
                 stochdepth_rate=0.0,
                 drop_rate=0.0,
                 use_se=False,
                 se_ratio=0.25):
    model = NF_ResNet(num_classes=num_classes,
                      variant='ResNet50',
                      width=width,
                      alpha=alpha,
                      stochdepth_rate=stochdepth_rate,
                      drop_rate=drop_rate,
                      use_se=use_se,
                      se_ratio=se_ratio)
    return model

def nf_resnet_101(num_classes, **kwargs):
    return NF_ResNet(num_classes=num_classes, variant='ResNet101', **kwargs)

def nf_resnet_152(num_classes, **kwargs):
    return NF_ResNet(num_classes=num_classes, variant='ResNet152', **kwargs)
