import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap


class CGIteration(torch.nn.Module):

    def __init__(self, cg, all_species, L_max=None) -> None:
        super().__init__()
        self.cg = cg
        self.all_species = all_species
        self.L_max = L_max if L_max is not None else 100

    def initialize(self, ghost1, ghost2, mode="full"):

        ghost_out = {}
        selected_features = {}
        for (a1, l1, s1), q_max_1 in ghost1.items():
            for (a2, l2, s2), q_max_2 in ghost2.items():
                if (a1 != a2): continue
                for L in range(abs(l1-l2), min(l1+l2, self.L_max)+1):
                    S = s1 * s2 * (-1)**(l1+l2+L)
                    if (a1, L, S) not in ghost_out: ghost_out[(a1, L, S)] = 0
                    ghost_out[(a1, L, S)] += q_max_1 * q_max_2
                    selected_features[(l1, s1, l2, s2, L, S)] = torch.LongTensor(
                        [[q1, q2] for q1 in range(q_max_1) for q2 in range(q_max_2)]
                    )

        self.selections = selected_features

        return ghost_out

    def forward(self, tmap1, tmap2):

        block_dictionary = {}
        for a_i in self.all_species:
            for (l1, s1, l2, s2, L, S), selected_features in self.selections.items():
                if (a_i, L, S) not in block_dictionary: block_dictionary[(a_i, L, S)] = []
                block_dictionary[(a_i, L, S)].append(
                    self.cg.combine(
                        tmap1.block(a_i=a_i, lam=l1, sigma=s1),
                        tmap2.block(a_i=a_i, lam=l2, sigma=s2),
                        L,
                        selected_features
                    )
                )
        
        keys_out = []
        blocks_out = []
        for key, block in block_dictionary.items():
            L = key[1]
            block = torch.cat(block, dim=-1)
            blocks_out.append(
                TensorBlock(
                    values = block,
                    samples = tmap1.block(a_i=key[0], lam=0, sigma=1).samples,
                    components = [Labels(
                        names = ("m",),
                        values = np.arange(-L, L+1, dtype=np.int32).reshape(2*L+1, 1)
                    )],
                    properties = Labels(
                        names = ["mixed"],
                        values = np.arange(block.shape[-1]).reshape((-1, 1))
                    )
                )
            )
            keys_out.append(list(key))

        tmap_out = TensorMap(
            keys = Labels(
                names = ["a_i", "lam", "sigma"],
                values = np.array(keys_out)
            ),
            blocks = blocks_out
        )

        return tmap_out

