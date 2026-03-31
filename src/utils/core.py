"""
Core utilities: seeding, GPU memory management, and environment checks.
"""

import gc
import random
import subprocess
import torch
import numpy as np
from transformers import PreTrainedModel


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """

    # Taken from
    # https://github.com/huggingface/trl/blob/b4899b29d246ff656ba736198a7730f9e96aa73f/trl/core.py#L233
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def claim_memory():
    """Force a Python GC cycle and clear the CUDA memory cache."""
    gc.collect()
    torch.cuda.empty_cache()

# https://github.com/huggingface/transformers/issues/28188
def supports_flash_attention():
    """Check if the current GPU used supports FlashAttention."""

    print("cuda is available", torch.cuda.is_available())
    
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    major, minor = torch.cuda.get_device_capability(DEVICE)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90

def print_cuda_usage():
    """Print current free and total GPU memory in GiB."""
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    print("free", free, "total", total)



def check_gcc():
    """Print whether GCC is available on the system and its version string."""
    try:
        # Run the `gcc --version` command
        result = subprocess.run(["gcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the command ran successfully
        if result.returncode == 0:
            print("GCC is installed.")
            print(result.stdout.splitlines()[0])  # Print the first line of the version output
        else:
            print("GCC is not installed.")
    except FileNotFoundError:
        print("GCC is not available on your system.")



def get_gpu_memory_in_gb(device_id):
    """
    Given a GPU device ID, returns the total memory in GB as an integer.
    
    Parameters:
        device_id (int): The ID of the GPU device.

    Returns:
        int: Total memory of the GPU in GB.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid device ID. Available device IDs: 0 to {torch.cuda.device_count() - 1}")
    
    # Get total memory in bytes
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    
    # Convert to GB and return as an integer
    return int(total_memory / (1024**3))


def count_transformers_models():
    """
    Count live HuggingFace ``PreTrainedModel`` instances in Python memory.

    Useful for debugging memory leaks between training stages.

    Returns:
        tuple[int, list]: ``(count, model_list)`` where ``model_list`` is the
        list of ``PreTrainedModel`` objects currently reachable by the GC.
    """
    objects = gc.get_objects()
    transformers_models = [obj for obj in objects if isinstance(obj, PreTrainedModel)]
    return len(transformers_models), transformers_models

