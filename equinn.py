import argparse
import json
import numpy as np
import torch
import ase
from dataset import get_dataset_slices
from spherical_expansions import VectorExpansion, SphericalExpansion

def run_fit(parameters):

    # Load parameters
    param_dict = json.load(open(parameters, "r"))
    RANDOM_SEED = param_dict["random seed"]
    ENERGY_CONVERSION = param_dict["energy conversion"]
    TARGET_KEY = param_dict["target key"]
    DATASET_PATH = param_dict["dataset path"]
    n_test = param_dict["n_test"]
    n_train = param_dict["n_train"]
    r_cut = param_dict["r_cut"]

    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calculating features on {device}")

    conversions = {}
    conversions["HARTREE_TO_EV"] = 27.211386245988
    conversions["HARTREE_TO_KCAL_MOL"] = 627.509608030593
    conversions["EV_TO_KCAL_MOL"] = conversions["HARTREE_TO_KCAL_MOL"]/conversions["HARTREE_TO_EV"]
    conversions["KCAL_MOL_TO_MEV"] = 0.0433641153087705*1000.0
    conversions["METHANE_FORCE"] = conversions["HARTREE_TO_KCAL_MOL"]/0.529177

    CONVERSION_FACTOR = conversions[ENERGY_CONVERSION]

    if "rmd17" in DATASET_PATH:
        train_slice = str(0) + ":" + str(n_train)
        test_slice = str(0) + ":" + str(n_test)
    else:
        test_slice = str(0) + ":" + str(n_test)
        train_slice = str(n_test) + ":" + str(n_test+n_train)
    
    train_structures, test_structures = get_dataset_slices(DATASET_PATH, train_slice, test_slice)

    n_max = [12, 5, 4, 3]
    l_max = len(n_max) - 1

    hypers = {
        "cutoff radius": r_cut,
        "radial basis": {
            "cutoff radius": r_cut,
            "mode": "full bessel",
            "l_max": l_max,
            "n_max": n_max
        },
        "l_max": l_max
    }

    all_species = np.sort(np.unique(np.concatenate([train_structure.numbers for train_structure in train_structures] + [test_structure.numbers for test_structure in test_structures])))
    print(f"All species: {all_species}")

    print("Testing vector expansion")
    vector_expansion_calculator = VectorExpansion(hypers)
    tmap = vector_expansion_calculator(train_structures[:1000])

    print("Testing spherical expansion")
    spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
    tmap = spherical_expansion_calculator(train_structures[:1000])

    class Model(torch.nn.Module):

        def __init__(self, hypers, n_feat, all_species) -> None:
            super().__init__()
            self.all_species = all_species
            self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
            self.linear = torch.nn.ModuleDict({
                str(a_i): torch.nn.Linear(n_feat, 1) for a_i in self.all_species
            })

        def forward(self, structures):
            spherical_expansion = self.spherical_expansion_calculator(structures)
            atomic_energies = []
            structure_indices = []
            for a_i in self.all_species:
                block = spherical_expansion.block(a_i=a_i, l=0)
                features = block.values.squeeze(dim=1)
                structure_indices.append(block.samples["structure"])
                atomic_energies.append(
                    self.linear[str(a_i)](features).squeeze(dim=-1)
                )
            atomic_energies = torch.concat(atomic_energies)
            structure_indices = torch.LongTensor(np.concatenate(structure_indices))
            energies = torch.zeros((len(structures),))
            energies.index_add_(dim=0, index=structure_indices, source=atomic_energies)

            return energies

    model = Model(hypers, n_max[0]*len(all_species), all_species)
    data_loader = torch.utils.data.DataLoader(train_structures, batch_size=10, shuffle=True, collate_fn=(lambda x: x))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1) 

    avg = torch.mean(torch.tensor([structure.info["energy"] for structure in train_structures])*627.5)

    for epoch in range(100):
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            predicted_energies = model(batch)
            energies = torch.tensor([structure.info["energy"] for structure in batch])*627.5 - avg
            loss = torch.sum((energies-predicted_energies)**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(np.sqrt(total_loss/n_train))
    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="?"
    )

    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    args = parser.parse_args()
    run_fit(args.parameters)
