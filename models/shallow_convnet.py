#%%
"""ShallowConvNet implementation adapted from Schirrmeister et al. 2017."""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from .utils import _glorot_weight_zero_bias

class Square(nn.Module):
    """Square activation function."""
    
    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):
    """Log activation with numerical stability."""
    
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """
    Modified ShallowConvNet from Schirrmeister et al. 2017.
    Adjusted for shorter input time lengths.
    """
    def __init__(self, n_chans, n_outputs, n_times=50, cnn_temporal_kernels=40,
                 cnn_temporal_kernelsize=(13, 1), cnn_spatial_kernels=40,
                 cnn_poolsize=(5, 1), cnn_poolstride=(2, 1), cnn_pool_type="avg", dropout=0.5):
        super().__init__()
        if cnn_spatial_kernels != cnn_temporal_kernels:
            print("Warning: Spatial and Temporal kernels are different. Setting them to be the same.")
            cnn_spatial_kernels = cnn_temporal_kernels

        # Input adapter
        self.input_adapter = Rearrange("b c t -> b 1 t c")

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, cnn_temporal_kernels, cnn_temporal_kernelsize, bias=False),
            nn.Conv2d(cnn_temporal_kernels, cnn_spatial_kernels, (1, n_chans), bias=False),
            nn.BatchNorm2d(cnn_spatial_kernels, momentum=0.1, affine=True),
            Square(),
            nn.AvgPool2d(cnn_poolsize, stride=cnn_poolstride) if cnn_pool_type == "avg"
                else nn.MaxPool2d(cnn_poolsize, stride=cnn_poolstride),
            Log(),
            nn.Dropout(p=dropout)
        )

        # Calculate dense layer input size automatically
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_chans, n_times)
            adapted_dummy_input = self.input_adapter(dummy_input)
            dummy_output = self.conv_module(adapted_dummy_input)
            dense_input_size = self._num_flat_features(dummy_output)

        # Dense classification layer
        self.dense_module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_input_size, n_outputs)
        )

        _glorot_weight_zero_bias(self)

    def _num_flat_features(self, x):
        """Calculate number of flattened features."""
        size = x.size()[1:]
        return np.prod(size)

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.input_adapter(x)
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x

# #############################################################################
if __name__ == '__main__':
    # --- Example parameters ---
    N_CHANS = 60
    N_TIMES = 300
    N_OUTPUTS = 2  # Example: 4 classes
    BATCH_SIZE = 16 # Example: batch size of 16 trials

    # --- Create a dummy input tensor ---
    # Shape: (batch_size, channels, time_samples)
    dummy_input = torch.randn(BATCH_SIZE, N_CHANS, N_TIMES)
    print(f"Input tensor shape: {dummy_input.shape}\n")

    # --- Test Modified ShallowConvNet ---
    print("--- Testing ShallowConvNet ---")
    try:
        # Instantiate the model with example parameters
        shallow_model = ShallowConvNet(n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES)
        
        # Perform a forward pass
        output_shallow = shallow_model(dummy_input)
        
        print(f"ShallowConvNet instantiated successfully.")
        print(f"Output tensor shape: {output_shallow.shape}")
        assert output_shallow.shape == (BATCH_SIZE, N_OUTPUTS)
        print("Test PASSED: Output shape is correct.\n")
        
    except Exception as e:
        print(f"Test FAILED for ShallowConvNet: {e}\n")
# %%
