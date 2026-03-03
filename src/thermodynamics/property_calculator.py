from typing import Tuple

import numpy as np

from src.config.libloader import xp, cuda_is_available
from src.thermodynamics.model_properties import ModelProperties


class PropertyCalculator:

    @staticmethod
    def get_n(F, properties: ModelProperties):
        return xp.einsum('ijkl->i', F[properties.bc.n_ghost:-properties.bc.n_ghost]) * properties.dV

    @staticmethod
    def get_u1(F, n, properties: ModelProperties):
        u_num = xp.einsum('ijkl,j->i', F[properties.bc.n_ghost:-properties.bc.n_ghost], properties.xi) * properties.dV
        return u_num / xp.maximum(n, 1e-15)

    @staticmethod
    def get_T(F, n, u, properties: ModelProperties):
        E = xp.einsum('ijkl, jkl -> i', F[properties.bc.n_ghost:-properties.bc.n_ghost], properties.XI_SQUARE) * properties.dV
        return (2 / 3) * (E / xp.maximum(n, 1e-15) - u ** 2)

    @staticmethod
    def get_q(F, u, properties: ModelProperties):
        xi1_shifted = properties.XI1[None, :, :, :] - u[:, None, None, None]
        q = 0.5 * xp.einsum('ijkl, ijkl -> i', xi1_shifted * properties.XI_SQUARE, F[properties.bc.n_ghost:-properties.bc.n_ghost]) * properties.dV
        return q

    @staticmethod
    def get_macros(F, properties: ModelProperties):
        n = PropertyCalculator.get_n(F, properties)
        u = PropertyCalculator.get_u1(F, n, properties)
        T = PropertyCalculator.get_T(F, n, u, properties)
        q = PropertyCalculator.get_q(F, u, properties)
        return n, u, T, q

    @staticmethod
    def get_solution_macros(F, properties: ModelProperties) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, u, T, q = PropertyCalculator.get_macros(F, properties)
        if cuda_is_available:
            return xp.asnumpy(n), xp.asnumpy(u), xp.asnumpy(T), xp.asnumpy(q)
        else:
            return n, u, T, q

    @staticmethod
    def get_mu(T, properties: ModelProperties):
        return T**properties.w

    @staticmethod
    def get_nu(n, T, properties: ModelProperties):
        return n * T ** (1 - properties.w) * 0.9 / properties.Kn

    @staticmethod
    def get_fS(F, properties: ModelProperties):
        n, u, T, q = PropertyCalculator.get_macros(F, properties)

        n_4d = n[:, None, None, None]
        u_4d = u[:, None, None, None]
        T_4d = xp.maximum(T[:, None, None, None], 1e-12)
        q_4d = q[:, None, None, None]

        tx = (properties.XI1[None] - u_4d) / xp.sqrt(T_4d)
        ty = properties.XI2[None] / xp.sqrt(T_4d)
        tz = properties.XI3[None] / xp.sqrt(T_4d)
        c_sq = tx * tx + ty * ty + tz * tz
        e2 = xp.exp(-c_sq)
        M = e2

        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV
        fM = n_4d * M / Z
        S = 2 * q_4d / (xp.maximum(n_4d, 1e-12) * T_4d ** 1.5)

        shakhov_correction = (4 / 5) * (1 - properties.Pr) * S * tx * (c_sq - 2.5)
        fS = fM * (1 + shakhov_correction)
        return fS

    @staticmethod
    def get_J(F, properties: ModelProperties):
        n, u, T, q = PropertyCalculator.get_macros(F, properties)

        n_4d = n[:, None, None, None]
        u_4d = u[:, None, None, None]
        T_4d = xp.maximum(T[:, None, None, None], 1e-12)
        q_4d = q[:, None, None, None]

        tx = (properties.XI1[None] - u_4d) / xp.sqrt(T_4d)
        ty = properties.XI2[None] / xp.sqrt(T_4d)
        tz = properties.XI3[None] / xp.sqrt(T_4d)
        c_sq = tx * tx + ty * ty + tz * tz
        e2 = xp.exp(-c_sq)
        M = e2

        #Z = (2 * xp.pi * T_4d) ** 1.5
        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV
        fM = n_4d * M / Z
        S = 2 * q_4d / (xp.maximum(n_4d, 1e-12) * T_4d ** 1.5)

        shakhov_correction = (4 / 5) * (1 - properties.Pr) * S * tx * (c_sq - 2.5)
        fS = fM * (1 + shakhov_correction)

        nu =  PropertyCalculator.get_nu(xp.maximum(n, 1e-12), xp.maximum(T, 1e-12), properties)
        nu_4d = nu.reshape(-1, 1, 1, 1)

        J = nu_4d * (fS - F[1:-1])
        return J
