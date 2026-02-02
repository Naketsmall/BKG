from typing import Tuple

from src.boundary_condition import BoundaryCondition
from src.config.libloader import xp, cuda_is_available

import numpy as np

class ModelProperties:
    def __init__(self, config: dict, bc: BoundaryCondition):

        self.x = xp.linspace(config['X_LEFT'], config['X_RIGHT'], config['n_x'])
        self.xi = xp.linspace(config['XI_LEFT'], config['XI_RIGHT'], config['n_xi'])
        self.xi_cell_size = self.xi[1] - self.xi[0]
        self.dV = (self.xi[1] - self.xi[0]) ** 3
        self.h = self.x[1] - self.x[0]
        self.XI1 = self.xi[:, None, None]
        self.XI2 = self.xi[None, :, None]
        self.XI3 = self.xi[None, None, :]
        self.XI_SQUARE = (self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2)

        self.Kn = config['Kn']
        self.Pr = config['Pr']
        self.w = config['w']

        self.bc = bc


class ModelState:

    def __init__(self, properties: ModelProperties, config: dict):
        self.F = None
        self.init_conditions(properties, config)

    def init_conditions(self, properties: ModelProperties, config: dict):

        N = len(properties.x)

        n = xp.zeros((len(properties.x) + 1), dtype=xp.float64)
        n[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_N'](
            (properties.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        u = xp.zeros((len(properties.x) + 1), dtype=xp.float64)
        u[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_U'](
            (properties.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        T = xp.zeros((len(properties.x) + 1), dtype=xp.float64)
        T[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_T'](
            (properties.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        self.F = xp.zeros((len(properties.x) + 1, len(properties.xi), len(properties.xi), len(properties.xi)))
        self.F[properties.bc.n_ghost:-properties.bc.n_ghost, :, :, :] = self.init_F_vectorized(n, u, T, properties, properties.bc.n_ghost)

    @staticmethod
    def init_F_vectorized(n, u, T, properties: ModelProperties, n_ghost):
        """
        Инициализирует 4D [x, xi1, xi2, xi3] пространство локальным максвеллианом с параметрами (n, u, T)
        :param n_ghost колмчество граничных ячеек.
        Z вычисляется дискретно как сумма, а не аналитически
        """

        n_4d = n[n_ghost:-n_ghost, None, None, None]
        u_4d = u[n_ghost:-n_ghost, None, None, None]
        T_4d = T[n_ghost:-n_ghost, None, None, None]

        v_sq = (properties.XI1 - u_4d) ** 2 + properties.XI2 ** 2 + properties.XI3 ** 2

        M = xp.exp(-v_sq / T_4d)

        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV
        F = n_4d * M / Z

        return F

    def get_F(self):
        return self.F

    def set_F(self, F):
        if F.shape != self.F.shape:
            raise ValueError(f"Shape mismatch error: F.shape: {F.shape} != self.F.shape: {self.F.shape}")
        self.F = F


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


class ShakhovSolver:

    def __init__(self, state: ModelState, properties: ModelProperties,
                 solver, prop_calc=PropertyCalculator):
        self.state = state
        self.props = properties
        self.solver = solver
        self.prop_calc = prop_calc

    def calculate(self, CFL, t_max):
        t_cur = 0
        while t_cur < t_max:
            print(f"calculation: {t_cur} / {t_max}")
            n, u, T, q = self.prop_calc.get_macros(self.state.F, self.props)
            tau = min(CFL * self.props.h / xp.max(xp.abs(self.props.xi)),
                      max(t_max - t_cur, 1e-15))
            """tau = xp.min([CFL * self.props.h / xp.max(xp.abs(self.props.xi)),
                      1 / xp.max(self.prop_calc.get_nu(n, T, self.props)),
                      xp.max([t_max - t_cur, 1e-15])])"""
            self.solver.calculate_layer(self.state.F, t_cur, tau, self.props, self.prop_calc)
            t_cur += tau
