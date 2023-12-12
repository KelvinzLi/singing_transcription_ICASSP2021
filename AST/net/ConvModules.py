import torch
from torch import nn

class ConvLayer1d(nn.Module):
    def __init__(self, in_features, out_features, *conv_args, activation=nn.ReLU(), conv_builder=nn.Conv1d, **conv_kwargs):
        super().__init__()
        self.conv = conv_builder(in_features, out_features, *conv_args, **conv_kwargs)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_features, out_features, *conv_args, mid_features=None, **conv_kwargs):
        super().__init__()
        
        if mid_features is None:
            mid_features = in_features
        
        self.depthwise = nn.Conv1d(in_features, mid_features, *conv_args, groups=in_features, **conv_kwargs)
        self.pointwise = nn.Conv1d(mid_features, out_features, kernel_size=1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
class ConvLayer2d(nn.Module):
    def __init__(self, in_features, out_features, *conv_args, activation=nn.ReLU(), conv_builder=nn.Conv2d, **conv_kwargs):
        super().__init__()
        self.conv = conv_builder(in_features, out_features, *conv_args, **conv_kwargs)
        self.bn = nn.BatchNorm2d(out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, **conv_kwargs):
        super().__init__()
        
        if mid_features is None:
            mid_features = in_features
        
        self.depthwise = nn.Conv2d(in_features, in_features, *conv_args, groups=in_features, **conv_kwargs)
        self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
class ResConnect(nn.Module):
    def __init__(self, layer, shortcut=nn.Identity(), activation=nn.ReLU()):
        super().__init__()
        self.layer = layer
        self.activation = activation
        self.shortcut = shortcut
        
    def forward(self, x):
        x = self.layer(x) + self.shortcut(x)
        x = self.activation(x)
        return x
    
class ConcatConnect(nn.Module):
    def __init__(self, layer, shortcut=nn.Identity(), activation=nn.Identity(), dim=1):
        super().__init__()
        self.layer = layer
        self.activation = activation
        self.shortcut = shortcut
        self.dim = dim
        
    def forward(self, x):
        x = torch.cat((self.layer(x), self.shortcut(x)), dim = self.dim)
        x = self.activation(x)
        return x