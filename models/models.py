#%%
"""Model registry and imports."""
from .eegnet import EEGNetv4
from .shallow_convnet import ShallowConvNet
from .deep_convnet import DeepConvNet
from .atcnet import ATCNet
from .deep_tepnet import PRIME, Ablation_NoS4, Ablation_ConvInsteadOfS4

MODEL_CLASS_MAP = {
    "EEGNetv4": EEGNetv4,
    "ShallowConvNet": ShallowConvNet,
    "DeepConvNet": DeepConvNet,
    "ATCNet": ATCNet,
    "PRIME": PRIME,
    "Ablation_NoS4": Ablation_NoS4,
    "Ablation_ConvInsteadOfS4": Ablation_ConvInsteadOfS4,
}

PREFIX_MAP = {
    "EEGNetv4": "eegnet_",
    "ShallowConvNet": "shallow_",
    "DeepConvNet": "deep_",
    "ATCNet": "atcnet_",
    "PRIME": "prime_",
    "Ablation_NoS4": "ablation_nos4_",
    "Ablation_ConvInsteadOfS4": "ablation_conv_",
}