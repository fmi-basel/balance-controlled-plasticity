import logging
import jax
import re

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("incontrol.utils")


def select_device(device, gpu_id) -> None:
    """
    Selects device to run on.
    """
    if device == 'cpu':
        logger.info("Using CPU...")
        device = jax.devices('cpu')[0]
        jax.config.update('jax_platform_name', 'cpu')
        
    elif device == 'gpu':
        
        num_gpus = jax.devices('gpu')
        assert gpu_id < len(num_gpus), f"GPU {gpu_id} not available. Only {len(num_gpus)} GPUs available."

        logger.info(f"Using GPU {gpu_id}...")
        device = jax.devices('gpu')[gpu_id]
        jax.config.update('jax_platform_name', 'gpu')
        
    else:
        raise ValueError("Unknown device. Must be 'cpu' or 'gpu'.")
        
    jax.config.update("jax_default_device", device)
    return device


def make_weight_decay_mask(pytree):
    """
    Mask for weight decay (hidden layer weights only)
    """
    def apply_mask(subtree, parent_key=None):
        # If the current subtree is a dictionary, iterate through its keys
        if isinstance(subtree, dict):
            masked_subtree = {}
            for key, value in subtree.items():
                # If the parent key matches "hidden_{}", apply the mask (True) to all children
                if re.match(r"hidden_\d+", str(parent_key)):
                    masked_subtree[key] = True
                else:
                    # Recursively process the child subtrees
                    masked_subtree[key] = apply_mask(value, key)
            return masked_subtree
        # For non-dict items, return False (no weight decay)
        return False

    # Start recursive mask application from the root of the tree
    return apply_mask(pytree)

