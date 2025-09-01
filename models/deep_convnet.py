#%%
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .utils import Conv2dWithConstraint, _glorot_weight_zero_bias

class DeepConvNet(nn.Module):
    """
    DeepConvNet from Schirrmeister et al. 2017, modified to be more robust to shorter signal lengths
    by using smaller default kernel sizes and less aggressive pooling.
    """
    def __init__(self, n_chans, n_outputs, n_times, n_filters_time=25, filter_time_length=10,
                 dilation1=1, n_filters_spat=25,
                 # MODIFICATION: Changed default kernel sizes for later blocks from 10 to 5
                 n_filters_2=50, filter_length_2=5,
                 n_filters_3=100, filter_length_3=5,
                 n_filters_4=200, filter_length_4=5,
                 # MODIFICATION: Changed default pooling from 3 to 2 to be less aggressive
                 pool_time_length=2, pool_time_stride=2,
                 drop_prob=0.5, final_conv_length="auto", **kwargs):
        super().__init__()

        if final_conv_length == "auto":
            self.final_conv_length_calculated = self._get_final_conv_length(
                n_times, filter_time_length, dilation1, filter_length_2,
                filter_length_3, filter_length_4, pool_time_length, pool_time_stride)
        else:
            self.final_conv_length_calculated = final_conv_length

        # Block 1: Temporal + Spatial convolution
        padding_temporal = (dilation1 * (filter_time_length - 1)) // 2
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=False,
                     padding=(padding_temporal, 0), dilation=(dilation1, 1)),
            Conv2dWithConstraint(n_filters_time, n_filters_spat, (1, n_chans),
                               max_norm=2.0, bias=False),
            nn.BatchNorm2d(n_filters_spat, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((pool_time_length, 1), stride=(pool_time_stride, 1)),
            nn.Dropout(p=drop_prob)
        )

        # Subsequent blocks
        self.conv_block2 = self._create_conv_block(n_filters_spat, n_filters_2, filter_length_2, drop_prob)
        self.conv_block3 = self._create_conv_block(n_filters_2, n_filters_3, filter_length_3, drop_prob)
        self.conv_block4 = self._create_conv_block(n_filters_3, n_filters_4, filter_length_4, drop_prob)

        # Final classifier
        self.final_conv = nn.Conv2d(n_filters_4, n_outputs, (self.final_conv_length_calculated, 1), bias=True)

        self.to_b_1_c_t = Rearrange("b c t -> b 1 c t")

        _glorot_weight_zero_bias(self)

    def _create_conv_block(self, in_filters, out_filters, kernel_length, dropout):
        """Create standard convolution block."""
        # MODIFICATION: Changed pooling from (3,1) to (2,1) to be less aggressive
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, (kernel_length, 1), bias=False,
                     padding=((kernel_length - 1) // 2, 0)),
            nn.BatchNorm2d(out_filters, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.Dropout(p=dropout)
        )

    def _get_final_conv_length(self, n_times, f1, d1, f2, f3, f4, p1_k, p1_s):
        """Calculate final convolution length after all layers."""
        def conv_len(l_in, kernel, stride=1, dilation=1, padding=0):
            if l_in < kernel:
                # This is a critical check to prevent errors in conv layers
                # if the input is smaller than the kernel.
                return 0
            return math.floor(((l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1)

        def pool_len(l_in, kernel, stride, padding=0):
            return math.floor(((l_in + 2 * padding - kernel) / stride) + 1)

        len_ = n_times
        # Block 1
        padding1 = (d1 * (f1 - 1)) // 2
        len_ = conv_len(len_, f1, stride=1, dilation=d1, padding=padding1)
        len_ = pool_len(len_, p1_k, p1_s)

        # Blocks 2-4 with now less aggressive pooling (handled in _create_conv_block)
        pool_kernel_size, pool_stride = 2, 2
        for f in [f2, f3, f4]:
            padding = (f - 1) // 2
            len_ = conv_len(len_, f, padding=padding)
            len_ = pool_len(len_, pool_kernel_size, pool_stride)

        # Apply the same safeguard as EEGNet to prevent a kernel size of 0
        return max(1, int(len_))

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.to_b_1_c_t(x)
        x = x.permute(0, 1, 3, 2)  # -> (B, 1, T, C)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.final_conv(x)
        logits = x.squeeze(-1).squeeze(-1)
        return logits

# =============================================================================
# Test Runner
# =============================================================================
if __name__ == "__main__":
    signal_lengths_to_test = [50, 100, 200, 300, 400, 500]
    
    # Common parameters for the model
    batch_size = 4
    n_chans = 60
    n_outputs = 1

    print("--- Testing DeepConvNet with Various Signal Lengths (Robust Version) ---")
    
    for n_times in signal_lengths_to_test:
        print(f"\n--- Testing with signal length: {n_times} ---")
        try:
            # 1. Create a dummy input tensor
            dummy_input = torch.randn(batch_size, n_chans, n_times)
            
            # 2. Instantiate the model
            #    The model itself will calculate the final conv length inside __init__
            model = DeepConvNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
            
            # 3. Print the calculated kernel size for the final layer
            final_len = model.final_conv_length_calculated
            print(f"Calculated final convolution length: {final_len}")
            
            # 4. Perform a forward pass
            output = model(dummy_input)
            
            # 5. Print the output shape to confirm success
            print(f"Input shape:  {list(dummy_input.shape)}")
            print(f"Output shape: {list(output.shape)}")
            print("✅ Test PASSED")
            
        except Exception as e:
            print(f"❌ Test FAILED for signal length {n_times}")
            print(f"Error: {e}")