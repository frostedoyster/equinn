import torch
import spherical_bessel
from spherical_bessel_zeros import get_spherical_bessel_zeros

class RadialBasis(torch.nn.Module):

    def __init__(self, hypers) -> None:
        super().__init__()

        self.hypers = hypers
        self.spherical_bessel_zeros = get_spherical_bessel_zeros(hypers["nmax"], hypers["lmax"])
        # self.normalization_factors = .........
        self.radial_transform = (lambda x: x)
        self.a = hypers["cutoff radius"]

        if hypers["mode"] == "single bessel":
            self.n_max = [hypers["nmax"]] * (hypers["lmax"] + 1)


    def forward(self, r):

        x = self.radial_transform(r)
        l_max = self.hypers["lmax"]
        n_max = self.n_max

        radial_basis = []
        for l in range(l_max+1):
            l_block = []
            for n in range(self.n_max[l]):
                # Need normalization...
                R_nl = SphericalBesselFirstKind.apply(
                    l, 
                    self.spherical_bessel_zeros[n, l] * x / self.a  # Careful: needs to be n, 0 if using naive Bessel basis
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

