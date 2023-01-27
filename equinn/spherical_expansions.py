import numpy as np
import torch
import ase
from ase.neighborlist import primitive_neighbor_list
from equistore import TensorMap, Labels, TensorBlock

from .angular_basis import AngularBasis
from .radial_basis import RadialBasis


class SphericalExpansion(torch.nn.Module):

    def __init__(self, hypers, all_species) -> None:
        super().__init__()

        self.hypers = hypers
        self.all_species = all_species
        self.vector_expansion_calculator = VectorExpansion(hypers)

    def forward(self, structures):
        
        expanded_vectors = self.vector_expansion_calculator(structures)
        samples_metadata = expanded_vectors.block(l=0).samples

        s_metadata = samples_metadata["structure"]
        i_metadata = samples_metadata["center"]
        ai_metadata = samples_metadata["species_center"]
        aj_metadata = samples_metadata["species_neighbor"]

        n_species = len(self.all_species)
        species_to_index = {atomic_number : i_species for i_species, atomic_number in enumerate(self.all_species)}

        s_i_metadata = np.stack([s_metadata, i_metadata], axis=-1)
        unique_s_i_indices, s_i_unique_to_metadata, s_i_metadata_to_unique = np.unique(s_i_metadata, axis=0, return_index=True, return_inverse=True)

        aj_shifts = np.array([species_to_index[aj_index] for aj_index in aj_metadata])
        density_indices = torch.LongTensor(s_i_metadata_to_unique*n_species+aj_shifts)

        l_max = self.hypers["l_max"]
        n_centers = len(unique_s_i_indices)
        densities = []
        for l in range(l_max+1):
            expanded_vectors_l = expanded_vectors.block(l=l).values
            densities_l = torch.zeros(
                (n_centers*n_species, expanded_vectors_l.shape[1], expanded_vectors_l.shape[2]), 
                dtype = torch.get_default_dtype(),
                device = expanded_vectors_l.device
            )
            densities_l.index_add_(dim=0, index=density_indices.to(expanded_vectors_l.device), source=expanded_vectors_l)
            densities_l = densities_l.reshape((n_centers, n_species, 2*l+1, -1)).swapaxes(1, 2).reshape((n_centers, 2*l+1, -1))
            densities.append(densities_l)

        ai_new_indices = torch.tensor(ai_metadata[s_i_unique_to_metadata])
        labels = []
        blocks = []
        for l in range(l_max+1):
            densities_l = densities[l]
            for a_i in self.all_species:
                where_ai = np.where(ai_new_indices == a_i)[0]
                densities_ai_l = densities_l[where_ai]
                labels.append([a_i, l, 1])
                blocks.append(
                    TensorBlock(
                        values = densities_ai_l,
                        samples = Labels(
                            names = ["structure", "center"],
                            values = unique_s_i_indices[where_ai]
                        ),
                        components = expanded_vectors.block(l=l).components,
                        properties = Labels(
                            names = ["a1", "n1", "l1"],
                            values = np.concatenate(
                                [np.stack([
                                    a_j*np.ones_like(expanded_vectors.block(l=l).properties["n"]), 
                                    expanded_vectors.block(l=l).properties["n"],
                                    l*np.ones_like(expanded_vectors.block(l=l).properties["n"])
                                ], axis=1) for a_j in self.all_species],
                                axis = 0
                            )
                        )
                    )
                )

        spherical_expansion = TensorMap(
            keys = Labels(
                names = ["a_i", "lam", "sigma"],
                values = np.array(labels)
            ),
            blocks = blocks
        )

        return spherical_expansion


class VectorExpansion(torch.nn.Module):

    def __init__(self, hypers) -> None:
        super().__init__()

        self.hypers = hypers
        self.spherical_harmonics_calculator = AngularBasis(hypers["l_max"])
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
        
        x = bare_cartesian_vectors[:, 0]
        y = bare_cartesian_vectors[:, 1]
        z = bare_cartesian_vectors[:, 2]

        spherical_harmonics = self.spherical_harmonics_calculator(x, y, z, r)

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

    for structure_index in range(structures.n_structures):

        where_selected_structure = np.where(structures.structure_indices == structure_index)[0]

        centers, neighbors, unit_cell_shift_vectors = get_neighbor_list(
            structures.positions.detach().cpu().clone().numpy()[where_selected_structure], 
            structures.pbcs[structure_index], 
            structures.cells[structure_index], 
            cutoff_radius) 
        
        positions = structures.positions[torch.LongTensor(where_selected_structure)]
        cell = torch.tensor(np.array(structures.cells[structure_index]), dtype=torch.get_default_dtype())
        species = structures.atomic_species[structure_index]

        structure_vectors = positions[neighbors] - positions[centers] + (unit_cell_shift_vectors @ cell).to(positions.device)  # Warning: it works but in a weird way when there is no cell
        vectors.append(structure_vectors)
        labels.append(
            np.stack([
            np.array([structure_index]*len(centers)), 
            centers.numpy(), 
            neighbors.numpy(), 
            species[centers], 
            species[neighbors]], axis=-1))

    vectors = torch.cat(vectors, dim=0)
    labels = np.concatenate(labels, axis=0)
    
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


def get_neighbor_list(positions, pbc, cell, cutoff_radius):

    centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
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
    unit_cell_shift_vectors = torch.tensor(unit_cell_shift_vectors, dtype=torch.get_default_dtype())

    return centers, neighbors, unit_cell_shift_vectors
