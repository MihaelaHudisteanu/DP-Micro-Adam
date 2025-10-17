import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperedSigmoid(nn.Module):
    def __init__(self, scale=1.58, inverse_temp=3.0, offset=0.71):
        super().__init__()
        self.scale = scale
        self.inverse_temp = inverse_temp
        self.offset = offset

    def forward(self, x):
        return self.scale * torch.sigmoid(self.inverse_temp * x) - self.offset

class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], unbiased=False, keepdim=True)
        weight = (weight - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class CNN5(nn.Module):
    def __init__(self, num_classes=10, activation='tanh', normalization='none', weight_standardization=False):
        super().__init__()

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0)
        elif activation == "tempered_sigmoid":
            self.activation = TemperedSigmoid()

        conv = WSConv2d if weight_standardization else nn.Conv2d

        def norm_layer(num_channels):
            if normalization == "none":
                return nn.Identity()
            elif normalization == "group_norm":
                return nn.GroupNorm(16, num_channels) 


        self.conv1 = conv(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = norm_layer(32)

        self.conv2 = conv(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(64)

        self.conv3 = conv(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = norm_layer(128)

        self.conv4 = conv(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = norm_layer(256)

        self.conv5 = conv(256, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.norm1(x)

        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.norm2(x)

        x = self.activation(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.norm3(x)

        x = self.activation(self.conv4(x))
        x = self.norm4(x)

        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, 1)  
        x = x.view(x.size(0), -1)     

        return x

class CNN5_Papernot(nn.Module):
    def __init__(self, activation_fn = nn.Tanh):
        super().__init__()
        self.act = activation_fn()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
    
        #self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) #see https://arxiv.org/pdf/2011.11660
        #self.conv8 = nn.Conv2d(256, 10, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool1(x)

        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool2(x)

        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        #x = self.act(self.conv7(x))
        #x = self.conv8(x) 

        #x = x.mean(dim=(2,3))
        return x
    
class CNN2_Papernot(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32, 32) 
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       
        x = self.pool1(x)       

        x = F.relu(self.conv2(x))     
        x = self.pool2(x)       

        # Flatten
        x = x.view(x.size(0), -1)   

        x = F.relu(self.fc1(x))
        x = self.fc2(x)               

        return x