
#%%
'''PRIME architecture and ablations for TMS-EEG classification'''
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch
from typing import Optional, List
from .utils import (Conv2dWithConstraint, Ensure4d, Expression, _calculate_pool1d_out_len,
                   _glorot_weight_zero_bias, squeeze_final_output)

from .s4 import FFTConv


class _S4Block(nn.Module):
    """
    An adapted S4 block using a bidirectional S4 layer, residual connection, and a feedforward network.
    """
    def __init__(
        self,
        features: int,        # The number of input and output features
        fft_l_max: int,
        drop_prob: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()

        # S4 Layer is now bidirectional and assumes input is (B, C, L)
        self.s4_layer = FFTConv(
            d_model=features,
            l_max=fft_l_max,
            bidirectional=True, # The key feature from your revised block
        )

        # Standard feedforward path
        self.pointwise = nn.Conv1d(
            features, features, kernel_size=1, bias=use_bias
        )
        self.norm = nn.BatchNorm1d(features)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Input x: (B, C, L)
        residual = x

        # S4 layer operates directly on the (B, C, L) input
        x_s4, _ = self.s4_layer(x)

        # FFN processes the S4 output
        x_ffn = self.pointwise(x_s4)
        x_ffn = self.norm(x_ffn)
        x_ffn = self.activation(x_ffn)
        x_ffn = self.dropout(x_ffn)

        # Add the residual connection
        return x_ffn + residual


#  PRIME ARCHITECTURE
class PRIME(nn.Module):
    """
    A hybrid architecture that fuses a spatial-temporal frontend with an S4 core.
    """
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        # --- Frontend Params ---
        n_filters_time: int = 25,
        filter_time_length: int = 10,
        n_filters_spat: int = 25,
        pool_time_length: int = 2,
        pool_time_stride: int = 2,
        # --- S4 Core & Regularization ---
        drop_prob: float = 0.4,
        # --- Other ---
        use_bias: bool = False,
    ):
        super().__init__()

        # --- Block 1: Spatial-Temporal Frontend ---
        self.conv_block1 = nn.Sequential(
            # Reshape and permute for 2D Convolutions
            Rearrange("b c t -> b 1 t c"),
            # Temporal convolution
            nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=use_bias,
                      padding='same'),
            # Spatial convolution
            Conv2dWithConstraint(n_filters_time, n_filters_spat, (1, n_chans),
                                 max_norm=2.0, bias=use_bias),
            nn.BatchNorm2d(n_filters_spat),
            nn.ELU(),
            # Pooling reduces the temporal dimension
            nn.MaxPool2d((pool_time_length, 1), stride=(pool_time_stride, 1)),
            nn.Dropout(p=drop_prob),
            # Reshape for the S4 block
            Rearrange("b f t 1 -> b f t")
        )

        # Calculate the sequence length and features entering the S4 block
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_chans, n_times)
            out = self.conv_block1(dummy_input)
            s4_features = out.shape[1]
            s4_input_length = out.shape[2]
            print(f"PRIME: Sequence length entering S4 is {s4_input_length}")
            print(f"PRIME: Features entering S4 is {s4_features}")

        n_s4_blocks = 1 # Hyperparameter to tune 
        self.s4_core = nn.Sequential(
            *[_S4Block(
                features=s4_features,
                fft_l_max=s4_input_length,
                drop_prob=drop_prob,
                use_bias=use_bias
            ) for _ in range(n_s4_blocks)]
        )

        # --- Block 3: Classifier ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(s4_features, n_outputs)
        )

        _glorot_weight_zero_bias(self)

    def forward(self, x):
        # Input x: (B, C, T)
        x = self.conv_block1(x)
        #x = self.s4_block(x)
        x = self.s4_core(x)
        x = self.classifier(x)
        return x

# Ablations: shared building blocks
def _make_conv_frontend(n_chans: int, drop_prob: float) -> nn.Sequential:
    """DeepConvNet style front-end = (B,C,T) ➜ (B,F,T)"""
    return nn.Sequential(
        Rearrange("b c t -> b 1 t c"),
        nn.Conv2d(1, 25, (10, 1), padding="same", bias=False),
        Conv2dWithConstraint(25, 25, (1, n_chans), max_norm=2.0, bias=False),
        nn.BatchNorm2d(25),
        nn.ELU(),
        nn.MaxPool2d((2, 1), stride=(2, 1)),
        nn.Dropout(drop_prob),
        Rearrange("b f t 1 -> b f t"),
    )


def _make_fc_classifier(in_features: int, n_outputs: int) -> nn.Sequential:
    """(B,F,T) ➜ (B, n_outputs) via global pooling"""
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(in_features, n_outputs),
    )


class _SimpleConvStack(nn.Module):
    """
    A simple stack of non-residual, non-dilated Conv1d blocks.
    """
    def __init__(self, channels: int, depth: int = 3, drop_prob: float = 0.4):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding="same", # Keeps sequence length the same
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.ELU(),
                nn.Dropout(drop_prob),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return self.layers(x) # No residual connection


#  Ablation 1 ─ No S4  (pure spatial-temporal front-end + FC head)
class Ablation_NoS4(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        drop_prob: float = 0.4,
    ):
        super().__init__()
        self.front_end = _make_conv_frontend(n_chans, drop_prob)

        # probe shapes
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            feat_dim = self.front_end(dummy).shape[1]

        self.classifier = _make_fc_classifier(feat_dim, n_outputs)
        _glorot_weight_zero_bias(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)
        return self.classifier(x)



#  Ablation 2 ─ Conv stack instead of S4
class Ablation_ConvInsteadOfS4(nn.Module):
    """
    Control #2: replace S4 with a Conv1d stack.
    """
    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        drop_prob: float = 0.4,
        conv_depth: int = 3,
    ):
        super().__init__()
        self.front_end = _make_conv_frontend(n_chans, drop_prob)

        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            feat_dim = self.front_end(dummy).shape[1]

        self.conv_core = _SimpleConvStack(
            channels=feat_dim, depth=conv_depth, drop_prob=drop_prob
        )
        # -------------------------

        self.classifier = _make_fc_classifier(feat_dim, n_outputs)
        _glorot_weight_zero_bias(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_end(x)
        x = self.conv_core(x)
        return self.classifier(x)

# Quick smoke-test 
if __name__ == "__main__":
    B, C, T = 16, 60, 50
    NUM_OUT = 1
    dummy = torch.randn(B, C, T)

    for cls in (Ablation_NoS4, Ablation_ConvInsteadOfS4, PRIME):
        net = cls(n_chans=C, n_outputs=NUM_OUT, n_times=T)
        out = net(dummy)
        print(f"{cls.__name__:>30}: out --> {list(out.shape)}  |  params = {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")


