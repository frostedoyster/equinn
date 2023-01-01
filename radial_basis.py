import torch
import spherical_bessel

class RadialBasis(torch.nn.Module):

    def __init__(self, hypers) -> None:
        super().__init__()

        self.spherical_bessel_zeros = ...
        self.normalization_factors = ...
        self.radial_transform = ...
        self.a = 

        ...
        # Also contains any transformation of the r variable into x!

    def forward(r):

        x = self.radial_transform(r)

        radial_basis = []
        for l in range(l_max+1):
            l_block = []
            for n in range(n_max[l]):
                R_nl = SphericalBesselFirstKind.apply(
                    l=l, 
                    x = self.spherical_bessel_zeros[n, l] * x / self.a
                )


class SphericalBesselFirstKind(torch.autograd.Function):

    @staticmethod
    def forward(ctx, l, x):

        assert(x.device == "cpu")
        output = spherical_bessel.first_kind_forward(l, x)
        ctx.save_for_backward(*[l, x])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        
        return None

