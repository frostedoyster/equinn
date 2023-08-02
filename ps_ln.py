import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap

def normalize_ps(ps):
    new_keys = []
    new_blocks = []
    for key, block in ps.items():
        new_keys.append(key.values)
        values = block.values
        mean = torch.mean(values, dim=-1, keepdim=True)
        centered_values = values - mean
        variance = torch.mean(centered_values**2, dim=-1, keepdim=True)
        new_values = centered_values / torch.sqrt(variance)
        new_blocks.append(
            TensorBlock(
                values = new_values,
                samples = block.samples,
                components = block.components,
                properties = block.properties
            )
        )
    return TensorMap(
        keys = Labels(
            names = ("a_i",),
            values = np.array(new_keys),
        ), 
        blocks = new_blocks
    )
