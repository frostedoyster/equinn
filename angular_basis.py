import torch
import spherical_harmonics


class AngularBasis(torch.nn.Module):

    def __init__(self, l_max) -> None:
        super().__init__()
        
        self.l_max = l_max

    def forward(self, cos_theta, phi):

        spherical_harmonics = SphericalHarmonics.apply(self.l_max, cos_theta, phi)
        return spherical_harmonics


class SphericalHarmonics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, l_max, cos_theta, phi):

        assert(cos_theta.is_cpu)
        assert(cos_theta.is_contiguous())
        assert(phi.is_cpu)
        assert(phi.is_contiguous())
        assert(len(cos_theta.shape) == 1)
        assert(cos_theta.shape == phi.shape)
        
        output = spherical_harmonics.forward(l_max, cos_theta, phi)
        ctx.l_max = l_max
        ctx.save_for_backward(*[cos_theta, phi])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        raise NotImplementedError
