import torch
import spherical_bessel
from spherical_bessel_utils import get_spherical_bessel_zeros, get_LE_normalization_factors


class RadialBasis(torch.nn.Module):

    def __init__(self, hypers) -> None:
        super().__init__()

        self.l_max = hypers["l_max"]
        self.spherical_bessel_zeros = get_spherical_bessel_zeros(max(hypers["n_max"]), hypers["l_max"])
        self.radial_transform = (lambda x: x)
        self.a = hypers["cutoff radius"]
        self.n_max = hypers["n_max"]
        self.mode = hypers["mode"]

        # Precompute normalization factors
        self.normalization_factors = get_LE_normalization_factors(
            SphericalBesselFirstKind.apply, 
            self.a, 
            max(self.n_max), 
            self.l_max, 
            self.spherical_bessel_zeros
        )

    def forward(self, r):

        x = self.radial_transform(r)

        radial_basis = []
        for l in range(self.l_max+1):
            l_eff = (0 if self.mode == "single bessel" else l)
            l_block = []
            for n in range(self.n_max[l]):
                R_nl = self.normalization_factors[n, l] * SphericalBesselFirstKind.apply(
                    l_eff, 
                    self.spherical_bessel_zeros[n, l_eff] * x / self.a  # Careful: needs to be n, 0 if using naive Bessel basis
                )
                l_block.append(R_nl)
            radial_basis.append(torch.stack(l_block, dim = -1))

        return radial_basis


class SphericalBesselFirstKind(torch.autograd.Function):

    @staticmethod
    def forward(ctx, l, x):

        assert(x.is_cpu)
        assert(x.is_contiguous())
        output = spherical_bessel.first_kind_forward(l, x)
        ctx.save_for_backward(*[l, x])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        raise NotImplementedError
