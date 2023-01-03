import torch

def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training=True
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,
        inputs=positions,
        grad_outputs=torch.ones_like(energy),  # This works because positions don't contribute to energies of other structures
        retain_graph=training,
        create_graph=training,
    )[0]
    return -1 * gradient


if __name__ == "__main__":
    
    positions = torch.tensor(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    energies = torch.sum(positions, dim=0)

    forces = compute_forces(energies, positions)
    print(forces)
