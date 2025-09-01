#%%
"""Centralized model builder for neural network models."""

import inspect
import logging
from typing import Any, Dict
import torch
import torch.nn as nn
from models.models import MODEL_CLASS_MAP


def build_model(model_name: str, n_channels: int, n_times: int, n_outputs: int,
                device: torch.device, model_specific_args: Dict[str, Any],
                target_type: str = "classification") -> Any:
    """Build neural network model with specified parameters."""
    
    if model_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CLASS_MAP.keys())}")
    
    ModelClass = MODEL_CLASS_MAP[model_name]

    if n_times is None or n_times <= 0:
        raise ValueError(f"Invalid n_times ({n_times}) for model building.")

    # Determine output size
    actual_n_outputs = n_outputs

    # Prepare constructor arguments
    temp_constructor_args = {}
    temp_constructor_args.update(model_specific_args)
    
    # Add common arguments
    if "n_chans" not in temp_constructor_args:
        temp_constructor_args["n_chans"] = n_channels
    if "n_channels" not in temp_constructor_args:
        temp_constructor_args["n_channels"] = n_channels
    if "n_outputs" not in temp_constructor_args:
        temp_constructor_args["n_outputs"] = actual_n_outputs
    if "n_times" not in temp_constructor_args:
        temp_constructor_args["n_times"] = n_times

    # Add device for PyTorch-based models if needed
    if hasattr(ModelClass, "_torch_device") and "device" not in temp_constructor_args:
        temp_constructor_args["device"] = device

    # Filter arguments based on model's __init__ signature
    final_constructor_args = {}
    if inspect.isclass(ModelClass):
        if not hasattr(ModelClass, "__init__"):
            raise TypeError(f"ModelClass {ModelClass.__name__} missing __init__ method.")
        
        sig = inspect.signature(ModelClass.__init__)
        expected_params = set(sig.parameters.keys())
        
        for k, v in temp_constructor_args.items():
            if k in expected_params:
                final_constructor_args[k] = v
            else:
                logging.debug(f"Filtering out argument '{k}' for model {model_name}")
    else:
        logging.warning(f"ModelClass for {model_name} is not a class. Using all arguments.")
        final_constructor_args = temp_constructor_args

    # Instantiate model
    try:
        logging.info(f"Building model {model_name} with args: {final_constructor_args}")
        model = ModelClass(**final_constructor_args)
    except TypeError as e:
        logging.error(f"Failed to instantiate {model_name}")
        logging.error(f"Constructor args attempted: {final_constructor_args}")
        if inspect.isclass(ModelClass) and hasattr(ModelClass, "__init__"):
            init_sig = inspect.signature(ModelClass.__init__)
            expected_params = set(init_sig.parameters.keys())
            expected_params.discard("self")
            logging.error(f"Expected __init__ params: {expected_params}")
            
            provided_keys = set(final_constructor_args.keys())
            missing_required = []
            for p_name, p_obj in init_sig.parameters.items():
                if (p_name != "self" and p_obj.default == inspect.Parameter.empty and 
                    p_name not in provided_keys):
                    missing_required.append(p_name)
            
            if missing_required:
                logging.error(f"Missing required params: {missing_required}")
            
            unexpected_params = provided_keys - expected_params
            if unexpected_params:
                logging.error(f"Unexpected params: {unexpected_params}")
        
        logging.error(f"Original error: {e}")
        raise

    # Handle device placement
    if isinstance(model, nn.Module):
        model.to(device)
        logging.info(f"Moved model {model_name} to device: {device}")
    elif hasattr(model, "_torch_device"):
        model_device = getattr(model, "_torch_device", "NotReported")
        logging.info(f"Model {model_name} device: {model_device}")
    else:
        logging.info(f"Model {model_name} assumed CPU or self-managed device")

    return model