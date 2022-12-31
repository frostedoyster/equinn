import argparse
import json
import numpy as np
import torch
import ase
from dataset import get_dataset_slices

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

    structure = train_structures[0]

    centers, neighbors, unit_cell_shift_vectors = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=structure.pbc,
        cell=structure.cell,
        positions=structure.positions,
        cutoff=r_cut,
        self_interaction=True,
        use_scaled_positions=False,
    )

    pairs_to_throw = np.logical_and(centers == neighbors, np.all(unit_cell_shift_vectors == 0, axis=1))
    pairs_to_keep = np.logical_not(pairs_to_throw)
    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    for center, neighbor, vector in zip(centers, neighbors, unit_cell_shift_vectors):
        print(center, neighbor, vector)

    positions = torch.tensor(structure.positions, dtype=torch.get_default_dtype())
    





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