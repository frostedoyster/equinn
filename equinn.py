import argparse
import json
import numpy as np
import torch
from equinn.dataset import get_dataset_slices
from equinn.spherical_expansions import SphericalExpansion
from equinn.forces import compute_forces
from equinn.structures import Structures
from equinn.error_measures import get_mae, get_rmse, get_sse
from equinn.conversions import get_conversions
from equinn.power_spectrum import PowerSpectrum
from equinn.contractions import LinearContractionBlock

def run_fit(**parameters):

    torch.set_default_dtype(torch.float64)

    # Unpack options
    random_seed = parameters["random seed"]
    energy_conversion = parameters["energy conversion"]
    force_conversion = parameters["force conversion"]
    target_key = parameters["target key"]
    dataset_path = parameters["dataset path"]
    do_forces = parameters["do forces"]
    force_weight = parameters["force weight"]
    n_test = parameters["n_test"]
    n_train = parameters["n_train"]
    r_cut = parameters["r_cut"]
    optimizer_name = parameters["optimizer"]

    np.random.seed(random_seed)
    print(f"Random seed: {random_seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    conversions = get_conversions()
    energy_conversion_factor = conversions[energy_conversion]
    if do_forces:
        force_conversion_factor = conversions[force_conversion]

    if "rmd17" in dataset_path:
        train_slice = str(0) + ":" + str(n_train)
        test_slice = str(0) + ":" + str(n_test)
    else:
        test_slice = str(0) + ":" + str(n_test)
        train_slice = str(n_test) + ":" + str(n_test+n_train)
    
    train_structures, test_structures = get_dataset_slices(dataset_path, train_slice, test_slice)

    n_max = [9, 8, 7, 7, 6, 5]
    l_max = len(n_max) - 1
    n_max_rs = [8]

    hypers = {
        "cutoff radius": r_cut,
        "radial basis": {
            "cutoff radius": r_cut,
            "mode": "full bessel",
            "kind": "first",
            "l_max": l_max,
            "n_max": n_max
        },
        "l_max": l_max
    }

    hypers_rs = {
        "cutoff radius": r_cut,
        "radial basis": {
            "cutoff radius": r_cut,
            "mode": "full bessel",
            "kind": "second",
            "l_max": 0,
            "n_max": n_max_rs
        },
        "l_max": 0
    }

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
    print(f"All species: {all_species}")


    class Model(torch.nn.Module):

        def __init__(self, hypers_rs, hypers, n_feat, all_species, do_forces) -> None:
            super().__init__()
            self.all_species = all_species
            self.radial_spectrum_calculator = SphericalExpansion(hypers_rs, all_species)
            self.radial_spectrum_contractor = LinearContractionBlock(
                {(a_i, 0, 1): (8*len(self.all_species), 19) for a_i in self.all_species}
            )
            self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
            self.power_spectrum_calculator = PowerSpectrum(all_species)
            self.additive = torch.nn.Parameter(torch.tensor([0.0]))
            self.radial_spectrum_model = torch.nn.ModuleDict({
                str(a_i): torch.nn.Linear(19, 1, bias=False) for a_i in self.all_species
            })
            """
            self.power_spectrum_model = torch.nn.ModuleDict({
                str(a_i): torch.nn.Linear(n_feat[1], 1, bias=False) for a_i in self.all_species
            })
            """
            self.do_forces = do_forces

        def forward(self, structures, is_training=True):

            #print("Transforming structures")
            structures = Structures(structures)
            structures.to(device)
            energies = torch.zeros((structures.n_structures,), device=device, dtype=torch.get_default_dtype())

            if self.do_forces:
                structures.positions.requires_grad = True

            #print("Calculating RS")
            radial_spectrum = self.radial_spectrum_calculator(structures)
            radial_spectrum = self.radial_spectrum_contractor(radial_spectrum)
            """
            spherical_expansion = self.spherical_expansion_calculator(structures)
            power_spectrum = self.power_spectrum_calculator(spherical_expansion, spherical_expansion)
            """

            #print("Calculating energies")
            atomic_energies = []
            structure_indices = []
            for a_i in self.all_species:
                block = radial_spectrum.block(a_i=a_i, lam=0, sigma=1)
                features = block.values.squeeze(dim=1)
                structure_indices.append(block.samples["structure"])
                atomic_energies.append(
                    self.radial_spectrum_model[str(a_i)](features).squeeze(dim=-1)
                )
            atomic_energies_rs = torch.concat(atomic_energies)
            structure_indices = torch.LongTensor(np.concatenate(structure_indices))
            
            energies.index_add_(dim=0, index=structure_indices.to(device), source=atomic_energies_rs)
            energies += self.additive

            """
            atomic_energies = []
            for a_i in self.all_species:
                block = power_spectrum.block(a_i=a_i)
                features = block.values.squeeze(dim=1)
                atomic_energies.append(
                    self.power_spectrum_model[str(a_i)](features).squeeze(dim=-1)
                )
            
            atomic_energies_ps = torch.concat(atomic_energies)
            energies.index_add_(dim=0, index=structure_indices, source=atomic_energies_ps)
            """

            #print("Computing forces by backpropagation")
            if self.do_forces:
                forces = compute_forces(energies, structures.positions, is_training=is_training)
            else:
                forces = None  # Or zero-dimensional tensor?

            return energies, forces

        def train_epoch(self, data_loader, force_weight):
            
            if optimizer_name == "Adam":
                total_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    predicted_energies, predicted_forces = model(batch)
                    energies = torch.tensor([structure.info[target_key] for structure in batch])*energy_conversion_factor - avg
                    energies = energies.to(device)
                    loss = get_sse(predicted_energies, energies)
                    if do_forces:
                        forces = torch.tensor(np.concatenate([structure.get_forces() for structure in batch], axis=0))*force_conversion_factor
                        forces = forces.to(device)
                        loss += force_weight * get_sse(predicted_forces, forces)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            else:
                def closure():
                    optimizer.zero_grad()
                    for train_structures in data_loader:
                        batch.to(device)
                        predicted_energies, predicted_forces = model(train_structures)
                        energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor - avg
                        energies = energies.to(device)
                        loss = get_sse(predicted_energies, energies)
                        if do_forces:
                            forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
                            forces = forces.to(device)
                            loss += force_weight * get_sse(predicted_forces, forces)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                total_loss = loss.item()

            return total_loss

        # def print_state()... Would print loss, train errors, validation errors, test errors, ...


    n_feat = [
        n_max_rs[0]*len(all_species),
        sum([n_max[l]**2 * len(all_species)**2 for l in range(l_max+1)])
    ]
    model = Model(hypers_rs, hypers, n_feat, all_species, do_forces=do_forces).to(device)
    print(model)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1) 
        batch_size = 10
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
        batch_size = n_train

    data_loader = torch.utils.data.DataLoader(train_structures, batch_size=batch_size, shuffle=True, collate_fn=(lambda x: x))

    avg = torch.mean(torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor)



    predicted_train_energies, predicted_train_forces = model(train_structures, is_training=False)
    predicted_test_energies, predicted_test_forces = model(test_structures, is_training=False)
    train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor - avg
    train_energies = train_energies.to(device)
    test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor - avg
    test_energies = test_energies.to(device)
    if do_forces:
        train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
        train_forces = train_forces.to(device)
        test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
        test_forces = test_forces.to(device)
    print()
    print(f"Before training")
    print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
    if do_forces:
        print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")


    #with torch.autograd.set_detect_anomaly(True):
    for epoch in range(50):

        #force_weight = 1.0 if epoch < 10 else 1e-3
        
        total_loss = model.train_epoch(data_loader, force_weight)

        predicted_train_energies, predicted_train_forces = model(train_structures, is_training=False)
        predicted_test_energies, predicted_test_forces = model(test_structures, is_training=False)
        train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor - avg
        train_energies = train_energies.to(device)
        test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor - avg
        test_energies = test_energies.to(device)
        if do_forces:
            train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
            train_forces = train_forces.to(device)
            test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
            test_forces = test_forces.to(device)

        print()
        print(f"Epoch number {epoch}, Total loss: {total_loss}")
        print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
        if do_forces:
            print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")




    import ase
    import matplotlib.pyplot as plt
    atomic_species_strings = {number: name for name, number in ase.data.atomic_numbers.items()}
    for a_i in all_species:
        for a_j in all_species:
            r_array = np.linspace(0.2, 5.0, 100)
            structures = [
                ase.Atoms(atomic_species_strings[a_i] + atomic_species_strings[a_j], positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]))
                for r in r_array
            ]
            energies, _ = model(structures, is_training=False)
            energies = energies.detach().numpy()
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(r_array, energies)
            fig.savefig(atomic_species_strings[a_i] + atomic_species_strings[a_j] + ".pdf")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )
    args = parser.parse_args()
    parameter_dict = json.load(open(args.parameters, "r"))
    run_fit(**parameter_dict)
