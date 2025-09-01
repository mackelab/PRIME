#%%
"""Core utilities for neural network models."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_numpy(data):
    """Convert data to NumPy array if it's a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif data is None:
        return None
    else:
        try:
            return np.array(data)
        except Exception as e:
            raise TypeError(f"Data of type {type(data)} could not be converted to NumPy array. Error: {e}")


class Expression(nn.Module):
    """Compute given expression on forward pass."""
    
    def __init__(self, expression_fn):
        super().__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


def squeeze_final_output(x):
    """Remove empty dimensions at end."""
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _glorot_weight_zero_bias(model):
    """Initialize parameters with Glorot uniform and zero bias."""
    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            if "BatchNorm" not in module.__class__.__name__ and module.weight.ndim > 1:
                nn.init.xavier_uniform_(module.weight, gain=1)
            elif "BatchNorm" in module.__class__.__name__:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Ensure4d(nn.Module):
    """Ensure input has 4 dimensions."""
    
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with max norm constraint."""
    
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with max norm constraint."""
    
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    """Conv1d layer with max norm constraint."""
    
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None and self.max_norm > 0:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=1, maxnorm=self.max_norm)
        return super().forward(x)


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution with automatic padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                        padding=0, dilation=dilation, groups=groups, bias=bias)
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        x = F.pad(input, (self.__padding, 0))
        return super().forward(x)


def _calculate_conv1d_out_len(input_len, kernel_size, stride=1, padding=0, dilation=1):
    """Calculate output length of Conv1d layer."""
    return math.floor((input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def _calculate_pool1d_out_len(input_len, kernel_size, stride=1, padding=0, dilation=1):
    """Calculate output length of Pool1d layer."""
    if stride is None:
        stride = kernel_size
    return math.floor((input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)