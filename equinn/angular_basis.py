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
        return (*output,)  # Return as a tuple, otherwise backward won't be called

    @staticmethod
    def backward(ctx, *d_loss_d_output):

        l_max = ctx.l_max
        cos_theta, phi = ctx.saved_tensors

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

        #if torch.any(cos_theta > 0.9999):  # Careful also when it's close to -1
            #print("NAN TIME")
            #print(d_loss_d_cos_theta[torch.where(cos_theta > 0.9999)])
        #d_loss_d_cos_theta[torch.where(cos_theta > 0.99)] = 0.0
        #d_loss_d_cos_theta[torch.where(cos_theta < -0.99)] = 0.0
        """
        if torch.any(cos_theta > 0.9999):
            print(d_loss_d_phi[torch.where(cos_theta > 0.9999)])
        if torch.any(cos_theta < -0.9999):
            print(d_loss_d_phi[torch.where(cos_theta < -0.9999)])
        """

        return None, d_loss_d_cos_theta, d_loss_d_phi
