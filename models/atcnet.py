#%%
"""ATCNet implementation adapted from Altaheri et al. 2023.
MODIFIED to be compatible with an input length of 50 samples.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .utils import (CausalConv1d, Conv2dWithConstraint, LinearWithConstraint, _glorot_weight_zero_bias)


class _ConvBlock(nn.Module):
    """Convolutional block for feature extraction."""
    
    def __init__(self, F1=16, kernel_length=64, dilation1=1, pool_length=8, pool_stride1=8,
                 D=2, in_channels=22, dropout=0.3):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.n_chans_for_spat_conv = in_channels

        self.rearrange_input = Rearrange("b c seq -> b 1 c seq")
        
        # Temporal convolution with dilation
        padding_temporal = (dilation1 * (kernel_length - 1)) // 2
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, padding_temporal),
                                     bias=False, dilation=(1, dilation1))
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)

        # Spatial convolution
        self.spat_conv = Conv2dWithConstraint(F1, F1 * D, (in_channels, 1), bias=False, groups=F1, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride1))
        self.drop1 = nn.Dropout(dropout)

        # MODIFIED: Second convolution block with parameters adjusted for shorter sequences.
        # Original kernel size was 16, padding 8.
        self.conv = nn.Conv2d(F1 * D, F1 * D, (1, 8), padding=(0, 4), bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nn.ELU()
        # MODIFIED: Pooling layer adjusted for shorter sequences.
        # Original pool size was (1, 8).
        self.pool2 = nn.AvgPool2d((1, 4), stride=(1, 2))
        self.drop2 = nn.Dropout(dropout)

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spat_conv(x)
        x = self.bn2(x)
        x = self.nonlinearity1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv(x)
        x = self.bn3(x)
        x = self.nonlinearity2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x


class _AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    
    def __init__(self, d_model, key_dim=8, n_head=2, dropout=0.5):
        super().__init__()
        self.n_head = n_head
        
        self.w_qs = nn.Linear(d_model, n_head * key_dim)
        self.w_ks = nn.Linear(d_model, n_head * key_dim)
        self.w_vs = nn.Linear(d_model, n_head * key_dim)
        self.fc = nn.Linear(n_head * key_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        q = rearrange(self.w_qs(x), "b l (head k) -> head b l k", head=self.n_head)
        k = rearrange(self.w_ks(x), "b t (head k) -> head b t k", head=self.n_head)
        v = rearrange(self.w_vs(x), "b t (head v) -> head b t v", head=self.n_head)
        
        attn = torch.einsum("hblk, hbtk -> hblt", [q, k]) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=3)
        
        output = torch.einsum("hblt,hbtv->hblv", [attn, v])
        output = rearrange(output, "head b l v -> b l (head v)")
        output = self.dropout(self.fc(output))
        output = output + residual
        return output


class TCNBlock(nn.Module):
    """Temporal convolutional network block with residual connection."""
    
    def __init__(self, kernel_length=4, n_filters=32, dilation=1, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_filters, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_filters, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)
        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        x = self.nonlinearity3(input + x)
        return x


class TCN(nn.Module):
    """Temporal convolutional network with multiple dilated blocks."""
    
    def __init__(self, depth=2, kernel_length=4, n_filters=32, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2**i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ATCBlock(nn.Module):
    """Combined attention and temporal convolution block."""
    
    def __init__(self, d_model=32, key_dim=8, n_head=2, dropout_attn=0.3, tcn_depth=2,
                 kernel_length=4, dropout_tcn=0.3, n_classes=4):
        super().__init__()
        self.attention_block = _AttentionBlock(d_model, key_dim, n_head, dropout_attn)
        self.rearrange = Rearrange("b seq c -> b c seq")
        self.tcn = TCN(tcn_depth, kernel_length, d_model, dropout_tcn)
        self.linear = LinearWithConstraint(d_model, n_classes, max_norm=0.25)

    def forward(self, x):
        x = self.attention_block(x)
        x = self.rearrange(x)
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x


class ATCNet(nn.Module):
    """ATCNet adapted from Altaheri et al. 2023."""

    def __init__(self, F1=16, 
                 # MODIFIED: Default ConvBlock parameters adjusted for shorter input sequences.
                 kernel_length_conv=32, dilation1_conv=1, pool_length=4, pool_stride1=4,
                 D=2, n_chans=22, n_outputs=4, n_times=None, dropout_conv=0.3, key_dim=8, n_head=2,
                 dropout_attn=0.5, tcn_depth=2, kernel_length_tcn=4, dropout_tcn=0.3, n_windows=5, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs
        self.n_windows = n_windows
        internal_d_model = F1 * D

        self.conv_block = _ConvBlock(F1=F1, kernel_length=kernel_length_conv, dilation1=dilation1_conv,
                                   pool_length=pool_length, pool_stride1=pool_stride1, D=D,
                                   in_channels=n_chans, dropout=dropout_conv)
        self.rearrange = Rearrange("b c 1 seq -> b seq c")
        self.atc_blocks = nn.ModuleList([
            ATCBlock(internal_d_model, key_dim, n_head, dropout_attn, tcn_depth,
                    kernel_length_tcn, dropout_tcn, n_outputs)
            for _ in range(n_windows)
        ])

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.conv_block(x)
        x = self.rearrange(x)

        bs, seq_len, _ = x.shape
        blk_output = torch.zeros(bs, self.n_outputs, dtype=x.dtype, device=x.device)
        
        if seq_len < self.n_windows:
            raise ValueError(f"Sequence length {seq_len} is too short for {self.n_windows} windows.")

        for i, blk in enumerate(self.atc_blocks):
            window_data = x[:, i : (seq_len - self.n_windows + i + 1), :]
            blk_output = blk_output + blk(window_data)

        blk_output = blk_output / self.n_windows
        return blk_output
    

# #############################################################################
if __name__ == '__main__':
    # --- Example parameters ---
    N_CHANS = 60
    N_TIMES = 50
    N_OUTPUTS = 4  # Example: 4 classes
    BATCH_SIZE = 16 # Example: batch size of 16 trials

    # --- Create a dummy input tensor ---
    # Shape: (batch_size, channels, time_samples)
    dummy_input = torch.randn(BATCH_SIZE, N_CHANS, N_TIMES)
    print(f"Input tensor shape: {dummy_input.shape}\n")

    # --- Test Modified ATCNet ---
    print("--- Testing ATCNet ---")
    try:
        # Instantiate the model with parameters for your data
        # Note: We pass other relevant params like kernel_length for the sub-modules
        atc_model = ATCNet(
            n_chans=N_CHANS,
            n_outputs=N_OUTPUTS,
            n_times=N_TIMES,
            kernel_length=32, # for _ConvBlock
            pool_length=4,    # for _ConvBlock
            pool_stride1=4,    # for _ConvBlock
            n_windows=3,      # for ATCNet main
            key_dim=8,        # for _AttentionBlock
            n_head=2,         # for _AttentionBlock
            tcn_depth=2,      # for TCN
            dropout=0.3
        )
        
        # Perform a forward pass
        output_atc = atc_model(dummy_input)

        print(f"ATCNet instantiated successfully.")
        print(f"Output tensor shape: {output_atc.shape}")
        assert output_atc.shape == (BATCH_SIZE, N_OUTPUTS)
        print("Test PASSED: Output shape is correct.\n")

    except Exception as e:
        print(f"Test FAILED for ATCNet: {e}\n")
# %%
