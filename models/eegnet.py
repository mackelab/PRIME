#%%
"""EEGNet v4 implementation from Lawhern et al. 2018."""

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .utils import (Conv2dWithConstraint, Ensure4d, Expression, _calculate_pool1d_out_len,
                   _glorot_weight_zero_bias, squeeze_final_output)


class EEGNetv4(nn.Module):
    """EEGNet v4 from Lawhern et al. 2018."""
    def __init__(self, n_chans, n_outputs, n_times, kernel_length=64, dilation1=1,
                 pool1_kernel=4, pool1_stride=4, sep_kernel=16, pool2_kernel=8, pool2_stride=8,
                 final_conv_length="auto", pool_mode="mean", F1=16, D=2, F2=32, 
                 drop_prob=0.25, add_log_softmax=False):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        if F2 != F1 * D:
            print(f"Warning: EEGNetv4 F2 ({F2}) usually equals F1*D ({F1*D}). Using provided F2.")

        if final_conv_length == "auto":
            if n_times is None:
                raise ValueError("n_times must be specified if final_conv_length is 'auto'")
            
            current_l = n_times
            current_l = _calculate_pool1d_out_len(current_l, pool1_kernel, pool1_stride)
            current_l = _calculate_pool1d_out_len(current_l, pool2_kernel, pool2_stride)
            self.final_conv_length_calculated = max(1, int(current_l))
        else:
            self.final_conv_length_calculated = final_conv_length

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.ensuredims = Ensure4d()
        self.dimshuffle = Rearrange("batch ch t 1 -> batch 1 ch t")

        # Block 1: Temporal and spatial convolution
        padding_temporal = (dilation1 * (kernel_length - 1)) // 2
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), stride=1, bias=False,
                                      padding=(0, padding_temporal), dilation=(1, dilation1))
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3)
        
        self.conv_spatial = Conv2dWithConstraint(F1, F1 * D, (self.n_chans, 1), max_norm=1,
                                               stride=1, bias=False, groups=F1, padding=(0, 0))
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3)
        self.elu_1 = Expression(F.elu)
        self.pool_1 = pool_class(kernel_size=(1, pool1_kernel), stride=(1, pool1_stride))
        self.drop_1 = nn.Dropout(p=drop_prob)

        # Block 2: Separable convolution
        padding_separable = sep_kernel // 2
        self.conv_separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel), stride=1, bias=False,
                                            groups=F1 * D, padding=(0, padding_separable), dilation=(1, 1))
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, (1, 1), stride=1, bias=False, padding=(0, 0))
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = Expression(F.elu)
        self.pool_2 = pool_class(kernel_size=(1, pool2_kernel), stride=(1, pool2_stride))
        self.drop_2 = nn.Dropout(p=drop_prob)

        # Classifier
        self.conv_classifier = nn.Conv2d(F2, self.n_outputs, (1, self.final_conv_length_calculated), bias=True)

        if add_log_softmax:
            raise ValueError("add_log_softmax=True is not suitable for BCEWithLogitsLoss. Set to False.")
        self.logsoftmax = nn.Identity()
        self.squeeze = Expression(squeeze_final_output)

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.ensuredims(x)
        x = self.dimshuffle(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        x = self.conv_classifier(x)
        x = self.logsoftmax(x)
        x = self.squeeze(x)
        return x