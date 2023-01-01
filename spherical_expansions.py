import numpy as np
import torch
import ase
from equistore import TensorMap, Labels, TensorBlock

from angular_basis import AngularBasis
from radial_basis import RadialBasis


class SphericalExpansion(torch.nn.Module):

    def __init__(self, hypers, all_species) -> None:
        super().__init__()

        self.hypers = hypers
        self.all_species = all_species
        self.vector_expansion = VectorExpansion(hypers)

    def forward(self, structures):
        return 0


class VectorExpansion(torch.nn.Module):

    def __init__(self, hypers) -> None:
        super().__init__()

        self.hypers = hypers
        self.spherical_harmonics_calculator = AngularBasis(hypers["lmax"])
        self.radial_basis_calculator = RadialBasis(hypers["radial basis"])

        # self.mlps = ...  # One for each l?

    def forward(self, structures):

        cutoff_radius = self.hypers["cutoff radius"]
        cartesian_vectors = get_cartesian_vectors(structures, cutoff_radius)

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)

        r = torch.sqrt(
            (bare_cartesian_vectors**2)
            .sum(dim=-1)
        )

        radial_basis = self.radial_basis_calculator(r)

        cos_theta = bare_cartesian_vectors[:, 2]/r
        phi = torch.atan2(bare_cartesian_vectors[:, 1], bare_cartesian_vectors[:, 0])

        spherical_harmonics = self.spherical_harmonics_calculator(cos_theta, phi)

        # Use broadcasting semantics to get the products in equistore shape
        vector_expansion_blocks = []
        for l, (radial_basis_l, spherical_harmonics_l) in enumerate(zip(radial_basis, spherical_harmonics)):
            vector_expansion_l = radial_basis_l.unsqueeze(dim = 1) * spherical_harmonics_l.unsqueeze(dim = 2)
            n_max_l = vector_expansion_l.shape[2]
            vector_expansion_blocks.append(
                TensorBlock(
                    values = vector_expansion_l,
                    samples = cartesian_vectors.samples,
                    components = [Labels(
                        names = ("m",),
                        values = np.arange(-l, l+1, dtype=np.int32).reshape(2*l+1, 1)
                    )],
                    properties = Labels(
                        names = ("n",),
                        values = np.arange(0, n_max_l, dtype=np.int32).reshape(n_max_l, 1)
                    )
                )   
            )

        l_max = len(vector_expansion_blocks) - 1
        vector_expansion_tmap = TensorMap(
            keys = Labels(
                names = ("l",),
                values = np.arange(0, l_max+1, dtype=np.int32).reshape(l_max+1, 1),
            ),
            blocks = vector_expansion_blocks
        )

        return vector_expansion_tmap


def get_cartesian_vectors(structures, cutoff_radius):

    labels = []
    vectors = []

    for structure_index, structure in enumerate(structures):
        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(structure, cutoff_radius) 
        positions = torch.tensor(structure.positions, dtype=torch.get_default_dtype())
        cell = structure.cell
        species = structure.get_atomic_numbers()

        for center, neighbor, unit_cell_shift_vector in zip(centers, neighbors, unit_cell_shift_vectors):
            vector = positions[neighbor] - positions[center] + torch.tensor(unit_cell_shift_vector.dot(cell), dtype=torch.get_default_dtype())

            vectors.append(vector)
            labels.append([structure_index, center, neighbor, species[center], species[neighbor]])

    vectors = torch.stack(vectors)
    
    block = TensorBlock(
        values = vectors.unsqueeze(dim=-1),
        samples = Labels(
            names = ["structure", "center", "neighbor", "species_center", "species_neighbor"],
            values = np.array(labels)
        ),
        components = [
            Labels(
                names = ["cartesian_dimension"],
                values = np.array([-1, 0, 1]).reshape((-1, 1))
            )
        ],
        properties = Labels.single()
    )

    return block 


def get_neighbor_list(structure, cutoff_radius):

    centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=structure.pbc,
        cell=structure.cell,
        positions=structure.positions,
        cutoff=cutoff_radius,
        self_interaction=True,
        use_scaled_positions=False,
    )

    pairs_to_throw = np.logical_and(centers == neighbors, np.all(unit_cell_shift_vectors == 0, axis=1))
    pairs_to_keep = np.logical_not(pairs_to_throw)

    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    centers = torch.LongTensor(centers)
    neighbors = torch.LongTensor(neighbors)

    return centers, neighbors, unit_cell_shift_vectors
