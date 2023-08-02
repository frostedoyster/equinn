import numpy as np
import torch
from dataset import get_dataset_slices
from torch_spex.forces import compute_forces
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, TransformerProperty, collate_nl
from torch_spex.spherical_expansions import SphericalExpansion
from power_spectrum import PowerSpectrum
from torch_spex.normalize import get_average_number_of_neighbors, normalize_true, normalize_false
import equistore
from ps_ln import normalize_ps

# Conversions

def get_conversions():
    
    conversions = {}
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177
    conversions["NO_CONVERSION"] = 1.0

    return conversions

# Error measures
def get_mae(first, second):
    return torch.mean(torch.abs(first - second))
def get_rmse(first, second):
    return torch.sqrt(torch.mean((first - second)**2))
def get_sse(first, second):
    return torch.sum((first - second)**2)

# Unpack options
torch.set_default_dtype(torch.float64)
random_seed = 123123
energy_conversion = "NO_CONVERSION"
force_conversion = "NO_CONVERSION"
target_key = "energy"
dataset_path = "datasets/alchemical.xyz"
do_forces = True
force_weight = 10.0
n_test = 200
n_train = 2000
r_cut = 6.0
optimizer_name = "Adam"

np.random.seed(random_seed)
torch.manual_seed(random_seed)
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

#######################################
print("common pseudo 4 pseudo correct normalization higher Emax layernorm 10.0 rcut 6.0 scale 3.0")
#######################################

n_pseudo = 4
normalize = True
print("normalize", normalize)
hypers = {
    "alchemical": n_pseudo,
    "cutoff radius": r_cut,
    "radial basis": {
        "type": "physical",
        "cost_trade_off": False,
        "scale": 3.0,
        "r_cut": r_cut,
        "E_max": 500,
        "mlp": True
    }
}
if not normalize:
    normalize_func = normalize_false
else:
    hypers["normalize"] = get_average_number_of_neighbors(train_structures, r_cut)
    print(hypers["normalize"])
    normalize_func = normalize_true

average_number_of_atoms = sum([structure.get_atomic_numbers().shape[0] for structure in train_structures])/len(train_structures)
print("Average number of atoms per structure:", average_number_of_atoms)

all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
print(f"All species: {all_species}")


class Model(torch.nn.Module):

    def __init__(self, hypers, all_species, do_forces) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species, device=device)
        n_max = self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        print(n_max)
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l]**2 * n_pseudo**2 for l in range(l_max+1)])
        self.ps_calculator = PowerSpectrum(l_max, all_species)
        self.combination_matrix = self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix
        self.all_species_labels = equistore.Labels(
            names = ["a_i"],
            values = all_species[:, None]
        )
        self.nu2_model = torch.nn.ModuleDict({
            str(alpha_i): torch.nn.Sequential(
                normalize_func("linear_no_bias", torch.nn.Linear(n_feat, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 256, bias=False)),
                normalize_func("activation", torch.nn.SiLU()),
                normalize_func("linear_no_bias", torch.nn.Linear(256, 1, bias=False))
            ) for alpha_i in range(n_pseudo)
        })
        # """
        self.do_forces = do_forces
        # self.zero_body_energies = torch.nn.Parameter(torch.zeros(len(all_species)))

    def forward(self, structures, is_training=True):

        n_structures = len(structures["positions"])
        energies = torch.zeros((n_structures,), device=device, dtype=torch.get_default_dtype())

        if self.do_forces:
            for structure_positions in structures["positions"]:
                structure_positions.requires_grad = True

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(**structures)
        ps = self.ps_calculator(spherical_expansion)
        if normalize: ps = normalize_ps(ps)

        # print("Calculating energies")
        self._apply_layer(energies, ps, self.nu2_model)
        if normalize: energies = energies / average_number_of_atoms
        # print("Final", torch.mean(energies), get_2_mom(energies))
        # energies += comp @ self.zero_body_energies

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(energies, structures["positions"], is_training=is_training)
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces


    def predict_epoch(self, data_loader):
        
        predicted_energies = []
        predicted_forces = []
        for batch in data_loader:
            batch.pop("energies")
            batch.pop("forces")
            predicted_energies_batch, predicted_forces_batch = model(batch, is_training=False)
            predicted_energies.append(predicted_energies_batch)
            predicted_forces.extend(predicted_forces_batch)  # the predicted forces for the batch are themselves a list

        predicted_energies = torch.concatenate(predicted_energies, dim=0)
        predicted_forces = torch.concatenate(predicted_forces, dim=0)
        return predicted_energies, predicted_forces


    def train_epoch(self, data_loader, force_weight):
        
        if optimizer_name == "Adam":
            total_loss = 0.0
            for batch in data_loader:
                energies = batch.pop("energies")
                forces = batch.pop("forces")
                optimizer.zero_grad()
                predicted_energies, predicted_forces = model(batch)

                loss = get_sse(predicted_energies, energies)
                if do_forces:
                    forces = forces.to(device)
                    predicted_forces = torch.concatenate(predicted_forces)
                    loss += force_weight * get_sse(predicted_forces, forces)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            def closure():
                optimizer.zero_grad()
                total_loss = 0.0
                for batch in data_loader:
                    energies = batch.pop("energies")
                    forces = batch.pop("forces")
                    predicted_energies, predicted_forces = model(batch)

                    loss = get_sse(predicted_energies, energies)
                    if do_forces:
                        forces = forces.to(device)
                        predicted_forces = torch.concatenate(predicted_forces)
                        loss += force_weight * get_sse(predicted_forces, forces)
                    loss.backward()
                    total_loss += loss.item()
                print(total_loss)
                return total_loss

            total_loss = optimizer.step(closure)
        return total_loss

    def _apply_layer(self, energies, tmap, layer):
        atomic_energies = []
        structure_indices = []
        # print(tmap.block(0).values)
        tmap = tmap.keys_to_samples("a_i")
        block = tmap.block()
        # print(block.values)
        samples = block.samples
        one_hot_ai = torch.tensor(
            equistore.one_hot(samples, self.all_species_labels),
            dtype = torch.get_default_dtype(),
            device = block.values.device
        )
        pseudo_species_weights = self.combination_matrix(one_hot_ai)
        features = block.values.squeeze(dim=1)
        #print("features", torch.mean(features), get_2_mom(features))
        embedded_features = features[:, :, None] * pseudo_species_weights[:, None, :]
        atomic_energies = torch.zeros((block.values.shape[0],), dtype=torch.get_default_dtype(), device=block.values.device)
        for alpha_i in range(n_pseudo):
            atomic_energies += layer[str(alpha_i)](embedded_features[:, :, alpha_i]).squeeze(dim=-1)
            #print("individual", torch.mean(layer[str(alpha_i)](embedded_features[:, :, alpha_i]).squeeze(dim=-1)), get_2_mom(layer[str(alpha_i)](embedded_features[:, :, alpha_i]).squeeze(dim=-1)))
        if normalize:
            atomic_energies = atomic_energies / np.sqrt(n_pseudo)
        #print("total", torch.mean(atomic_energies), get_2_mom(atomic_energies))
        structure_indices = torch.LongTensor(block.samples["structure"].copy())
        energies.index_add_(dim=0, index=structure_indices.to(device), source=atomic_energies)
        # print("in", torch.mean(energies), get_2_mom(energies))
        # THIS IN-PLACE MODIFICATION HAS TO CHANGE!

    # def print_state()... Would print loss, train errors, validation errors, test errors, ...

model = Model(hypers, all_species, do_forces=do_forces).to(device)
# print(model)

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    batch_size = 8  # Batch for training speed
else:
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", history_size=128)
    batch_size = 16  # Batch for memory

from torch_spex.normalize import get_2_mom
train_energies = torch.tensor([structure.info[target_key] for structure in train_structures])*energy_conversion_factor
train_energies = train_energies.to(device)
test_energies = torch.tensor([structure.info[target_key] for structure in test_structures])*energy_conversion_factor
test_energies = test_energies.to(device)

# Linear fit for one-body energies:
import rascaline
import equistore
center_species_labels = equistore.Labels(
    names = ["species_center"],
    values = np.array(all_species).reshape(-1, 1)
)
comp_calculator = rascaline.AtomicComposition(per_structure=True)
train_comp = comp_calculator.compute(train_structures)
train_comp = train_comp.keys_to_properties(center_species_labels)
train_comp = torch.tensor(train_comp.block().values).to(device)

c_comp = torch.linalg.solve(train_comp.T @ train_comp, train_comp.T @ train_energies)
model.energy_shifts = c_comp

test_comp = comp_calculator.compute(test_structures)
test_comp = test_comp.keys_to_properties(center_species_labels)
test_comp = torch.tensor(test_comp.block().values).to(device)

train_energies_rescaled = train_energies - train_comp @ model.energy_shifts
train_uncentered_std = torch.sqrt(get_2_mom(train_energies_rescaled))

if do_forces:
    train_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in train_structures], axis=0))*force_conversion_factor
    train_forces = train_forces.to(device)
    test_forces = torch.tensor(np.concatenate([structure.get_forces() for structure in test_structures], axis=0))*force_conversion_factor
    test_forces = test_forces.to(device)


print("Precomputing neighborlists")

def get_composition(frame):
    comp = comp_calculator.compute([frame])
    comp = comp.keys_to_properties(center_species_labels)
    comp = torch.tensor(comp.block().values).to(device)
    return comp[0]

transformers = [
    TransformerNeighborList(cutoff=hypers["cutoff radius"], device=device),
    TransformerProperty("energies", lambda frame: (torch.tensor([frame.info["energy"]], dtype=torch.get_default_dtype(), device=device)-get_composition(frame)@c_comp)/train_uncentered_std),
]
if do_forces: transformers.append(TransformerProperty("forces", lambda frame: torch.tensor(frame.get_forces(), dtype=torch.get_default_dtype(), device=device)/train_uncentered_std))

predict_train_dataset = InMemoryDataset(train_structures, transformers)
predict_test_dataset = InMemoryDataset(test_structures, transformers)
train_dataset = InMemoryDataset(train_structures, transformers)  # avoid sharing tensors between different dataloaders

predict_train_data_loader = torch.utils.data.DataLoader(predict_train_dataset, batch_size=32, shuffle=False, collate_fn=collate_nl)
predict_test_data_loader = torch.utils.data.DataLoader(predict_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_nl)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_nl)

print("Finished neighborlists")


for epoch in range(1000):

    predicted_train_energies, predicted_train_forces = model.predict_epoch(predict_train_data_loader)
    predicted_test_energies, predicted_test_forces = model.predict_epoch(predict_test_data_loader)
    predicted_train_energies *= train_uncentered_std
    predicted_test_energies *= train_uncentered_std
    predicted_train_forces *= train_uncentered_std
    predicted_test_forces *= train_uncentered_std
    predicted_train_energies += train_comp @ c_comp
    predicted_test_energies += test_comp @ c_comp

    print()
    if do_forces:
        print(f"Epoch number {epoch}, Total loss: {get_sse(predicted_train_energies, train_energies)+force_weight*get_sse(predicted_train_forces, train_forces)}, due to energies: {get_sse(predicted_train_energies, train_energies)}, due to forces: {force_weight*get_sse(predicted_train_forces, train_forces)}")
    else:
        print(f"Epoch number {epoch}, Total loss: {get_sse(predicted_train_energies, train_energies)}, due to energies: {get_sse(predicted_train_energies, train_energies)}")
    
    print(f"Energy errors: Train RMSE: {get_rmse(predicted_train_energies, train_energies)}, Train MAE: {get_mae(predicted_train_energies, train_energies)}, Test RMSE: {get_rmse(predicted_test_energies, test_energies)}, Test MAE: {get_mae(predicted_test_energies, test_energies)}")
    if do_forces:
        print(f"Force errors: Train RMSE: {get_rmse(predicted_train_forces, train_forces)}, Train MAE: {get_mae(predicted_train_forces, train_forces)}, Test RMSE: {get_rmse(predicted_test_forces, test_forces)}, Test MAE: {get_mae(predicted_test_forces, test_forces)}")

    _ = model.train_epoch(train_data_loader, force_weight)
