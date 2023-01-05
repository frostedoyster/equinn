import torch
import spherical_harmonics
import warnings


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
        assert(phi.is_cpu)
        assert(len(cos_theta.shape) == 1)
        assert(cos_theta.shape == phi.shape)

        output = spherical_harmonics.forward(l_max, cos_theta, phi)
        ctx.l_max = l_max
        ctx.save_for_backward(*[cos_theta, phi])
        return (*output,)  # Return as a tuple, otherwise backward won't be called

    @staticmethod
    def backward(ctx, *d_loss_d_output):

        l_max = ctx.l_max
        cos_theta, phi = ctx.saved_tensors

        # Print a warning when the cartesian vector is very close to the z axis.
        # This could trigger numerical instabilities in the backpropagation
        # through the spherical harmonics:
        if torch.any(cos_theta > 0.999999) or torch.any(cos_theta < -0.999999):
            warnings.warn("A vector happens to be very close to the z axis. Numerical instabilities could ensue during backward pass.", category=UserWarning)
            # Uncomment below to see the effects...

        d_output_d_cos_theta, d_output_d_phi = spherical_harmonics.backward(l_max, cos_theta, phi)

        d_loss_d_cos_theta = torch.sum(
            torch.cat([d_loss_d_output_single_l * d_output_d_cos_theta_single_l 
            for d_loss_d_output_single_l, d_output_d_cos_theta_single_l in zip(d_loss_d_output, d_output_d_cos_theta)], dim = 1),
            dim = 1
        )

        d_loss_d_phi = torch.sum(
            torch.cat([d_loss_d_output_single_l * d_output_d_phi_single_l 
            for d_loss_d_output_single_l, d_output_d_phi_single_l in zip(d_loss_d_output, d_output_d_phi)], dim = 1),
            dim = 1
        )

        """
        if torch.any(cos_theta > 0.999999):
            where = torch.where(cos_theta > 0.999999)
            print(d_loss_d_phi[where])  # Should be very small. Will be multiplied by very large dphi/dx and/or dphi/dy...
        
        if torch.any(cos_theta < -0.999999):
            where = torch.where(cos_theta < -0.999999)
            print(d_loss_d_phi[where])  # Should be very small. Will be multiplied by very large dphi/dx and/or dphi/dy...
        """

        return None, d_loss_d_cos_theta, d_loss_d_phi
