import numpy as np
import scipy
import torch
from scipy import arange, pi, sqrt, zeros
from scipy.special import jv
from scipy.optimize import brentq
from scipy.integrate import quadrature


def get_LE_normalization_factors(f, a, n_max, l_max, z_nl):

    N_nl = np.zeros((n_max, l_max+1))

    for l in range(l_max+1):
        for n in range(n_max):
            integral, _ = scipy.integrate.quadrature(lambda x: (f(l, torch.tensor(x))**2).numpy() * x**2, 0.0, z_nl[n, l], maxiter=100)
            N_nl[n, l] = (1.0/z_nl[n, l]**3 * integral)**(-0.5) * a**(-1.5)

    return N_nl


def get_spherical_bessel_zeros(n_big, l_big):

    z_ln = Jn_zeros(l_big, n_big)
    z_nl = z_ln.T

    return z_nl


def Jn(r, n):
    return (sqrt(pi/(2*r))*jv(n+0.5,r))

    
def Jn_zeros(n, nt):
  zerosj = zeros((n+1, nt), dtype=np.float64)
  zerosj[0] = arange(1,nt+1)*pi
  points = arange(1,nt+n+1)*pi
  racines = zeros(nt+n, dtype=np.float64)
  for i in range(1,n+1):
    for j in range(nt+n-i):
      foo = brentq(Jn, points[j], points[j+1], (i,))
      racines[j] = foo
    points = racines
    zerosj[i][:nt] = racines[:nt]
  return (zerosj)
