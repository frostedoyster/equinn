import numpy as np
import torch
import warnings


class AngularBasis(torch.nn.Module):

    def __init__(self, l_max) -> None:
        super().__init__()
        
        self.l_max = l_max

    def forward(self, cos_theta, phi):

        sh = spherical_harmonics(self.l_max, cos_theta, phi)
        return sh

    
@torch.jit.script
def factorial(n):
    return torch.exp(torch.lgamma(n+1))


@torch.jit.script
def associated_legendre_polynomials(l_max: int, x):
    p = []
    for l in range(l_max+1):
        p.append(
            torch.empty((l+1, x.shape[0]), dtype = x.dtype, device = x.device)
        )

    p[0][0] = 1.0
    for m in range(1, l_max+1):
        p[m][m] = -(2*m-1)*torch.sqrt((1.0-x)*(1.0+x))*p[m-1][m-1].clone()
    for m in range(l_max):
        p[m+1][m] = (2*m+1)*x*p[m][m].clone()
    for m in range(l_max-1):
        for l in range(m+2, l_max+1):
            p[l][m] = ((2*l-1)*x*p[l-1][m].clone()-(l+m-1)*p[l-2][m].clone())/(l-m)

    p = [p_l.swapaxes(0, 1) for p_l in p]
    return p


@torch.jit.script
def spherical_harmonics(l_max: int, cos_theta, phi):

    sqrt_2 = torch.sqrt(torch.tensor([2.0], device=cos_theta.device, dtype=cos_theta.dtype))
    pi = 2.0 * torch.acos(torch.zeros(1))

    Plm = associated_legendre_polynomials(l_max, cos_theta)
    phi = phi.unsqueeze(dim=-1)
    one_over_sqrt_2 = (1/sqrt_2).repeat(phi.shape)

    output = []
    for l in range(l_max+1):
        m = torch.LongTensor(list(range(-l, l+1)))
        abs_m = torch.abs(m)
        Phi = torch.cat([
            torch.sin(abs_m[:l]*phi),
            one_over_sqrt_2,
            torch.cos(abs_m[l+1:2*l+1]*phi)
        ], dim=-1)
        output.append(
            torch.pow(-1, m) * sqrt_2
            * torch.sqrt((2*l+1)/(4*pi)*factorial(l-abs_m)/factorial(l+abs_m))
            * Plm[l][:, abs_m]
            * Phi
        )

    return output

        

