import numpy as np
import torch
import wigners
from collections import namedtuple
import sparse_accumulation

SparseCGTensor = namedtuple("SparseCGTensor", "m1 m2 M cg")

class ClebschGordanReal:
    def __init__(self, algorithm = "fast cg"):
        self._cg = {}
        self.algorithm = algorithm


    def _add(self, l1, l2, L):
        # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

        if self._cg is None: 
            print("Trying to add CGs when not initialized... exiting")
            exit()

        if (l1, l2, L) in self._cg: 
            print("Trying to add CGs that are already present... exiting")
            exit()

        maxx = max(l1, max(l2, L))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )

        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
            real_cg.shape
        )
        real_cg = real_cg.swapaxes(0, 1)

        real_cg = real_cg @ c2r[L].T

        if (l1 + l2 + L) % 2 == 0:
            rcg = np.real(real_cg)
        else:
            rcg = np.imag(real_cg)

        # Note that this construction already provides "aligned" CG 
        # coefficients ready to be used in the sparse accumulation package:

        m1_array = [] 
        m2_array = []
        M_array = []
        cg_array = []

        for M in range(2 * L + 1):
            cg_nonzero = np.where(np.abs(rcg[:,:,M]) > 1e-15)
            m1_array.append(cg_nonzero[0])
            m2_array.append(cg_nonzero[1])
            M_array.append(np.array([M]*len(cg_nonzero[0])))
            cg_array.append(rcg[cg_nonzero[0], cg_nonzero[1], M])

        m1_array = torch.LongTensor(np.concatenate(m1_array))
        m2_array = torch.LongTensor(np.concatenate(m2_array))
        M_array = torch.LongTensor(np.concatenate(M_array))
        cg_array = torch.tensor(np.concatenate(cg_array)).type(torch.get_default_dtype())

        new_cg = SparseCGTensor(m1_array, m2_array, M_array, cg_array)
            
        self._cg[(l1, l2, L)] = new_cg

    
    def combine(self, block_nu, block_1, L, selected_features):

        lam = (block_nu.values.shape[1] - 1) // 2
        l = (block_1.values.shape[1] - 1) // 2

        if (lam, l, L) not in self._cg:
            self._add(lam, l, L)

        sparse_cg_tensor = self._cg[(lam, l, L)]

        mu_array = sparse_cg_tensor.m1.to(block_nu.values.device)
        m_array = sparse_cg_tensor.m2.to(block_nu.values.device)
        M_array = sparse_cg_tensor.M.to(block_nu.values.device)
        cg_array = sparse_cg_tensor.cg.to(block_nu.values.device)

        if self.algorithm == "fast cg":

            new_values = sparse_accumulation.accumulate(
                block_nu.values[:, :, selected_features[:, 0]].swapaxes(1, 2).contiguous(),
                block_1.values[:, :, selected_features[:, 1]].swapaxes(1, 2).contiguous(), 
                M_array, 
                2*L+1, 
                mu_array, 
                m_array, 
                cg_array
            ).swapaxes(1, 2)

        else:  # Python loop algorithm

            new_values = torch.zeros((block_nu.values.shape[0], 2*L+1, selected_features.shape[0]))

            for mu, m, M, cg_coefficient in zip(sparse_cg_tensor.m1, sparse_cg_tensor.m2, sparse_cg_tensor.M, sparse_cg_tensor.cg):
                new_values[:, M, :] += cg_coefficient * block_nu.values[:, mu, selected_features[:, 0]] * block_1.values[:, m, selected_features[:, 1]]

        return new_values


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)
