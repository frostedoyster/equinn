import numpy as np
import torch
from equinn.stringify import stringify
from equistore import Labels, TensorBlock, TensorMap


class LinearContractionBlock(torch.nn.Module):

    def __init__(self, sizes) -> None:
        super().__init__()

        self.contraction_weights = torch.nn.ModuleDict(
            {
                str(stringify(key)): torch.nn.Linear(size[0], size[1], bias=False) for key, size in sizes.items()
            }
        )

    def forward(self, map_in):

        keys_out = []
        blocks_out = []
        for key_in, block_in in map_in:
            block_out_values = self.contraction_weights[stringify(key_in)](block_in.values)
            block_out = TensorBlock(
                values = block_out_values,
                samples = block_in.samples,
                components = block_in.components,
                properties = Labels(
                    names = ["mixed"],
                    values = np.arange(block_out_values.shape[-1]).reshape((-1, 1))
                )
            )
            keys_out.append(list(key_in))
            blocks_out.append(block_out)

        map_out = TensorMap(
            keys = Labels(
                names = map_in.keys.names,
                values = np.array(keys_out)
            ),
            blocks = blocks_out
        )

        return map_out

