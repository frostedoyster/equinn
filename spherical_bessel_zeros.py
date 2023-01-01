import numpy as np
from scipy import arange, pi, sqrt, zeros
from scipy.special import jv
from scipy.optimize import brentq

def get_laplacian_eigenvalues(n_big, l_big):

    z_ln = Jn_zeros(l_big, n_big)
    z_nl = z_ln.T

    return z_nl**2


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
