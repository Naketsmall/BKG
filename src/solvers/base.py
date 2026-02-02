from src.boundary_condition import BoundaryCondition
from src.config.libloader import xp, cuda_is_available
from abc import ABC, abstractmethod

from src.thermodynamics import ModelProperties


def minmod(F):
    F_left = xp.roll(F, shift=1, axis=0)
    F_right = xp.roll(F, shift=-1, axis=0)

    du_minus = F - F_left
    du_plus = F_right - F

    sigma = xp.zeros_like(F)
    mask = (du_minus * du_plus) > 0
    sigma[mask] = xp.sign(du_minus[mask]) * xp.minimum(
        xp.abs(du_minus[mask]),
        xp.abs(du_plus[mask])
    )

    sigma[0] = 0
    sigma[-1] = 0

    return sigma


def W_god(u_l, u_r, coef_per):
    return xp.where(coef_per >= 0, u_l, u_r)


class Solver(ABC):
    __name__ = "Solver"

    def get_name(self):
        return self.__name__

    @abstractmethod
    def _step(self, F, t, tau, properties: ModelProperties, prop_calc):
        pass

    @abstractmethod
    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        pass

    @staticmethod
    def _calculate_collisions(F, tau, properties: ModelProperties, prop_calc):
        fS = prop_calc.get_fS(F, properties)
        n, u, T, q = prop_calc.get_macros(F, properties)
        nu4d = prop_calc.get_nu(n, T, properties)[:, None, None, None]
        F[properties.bc.n_ghost:-properties.bc.n_ghost] = \
            F[properties.bc.n_ghost:-properties.bc.n_ghost] * \
            xp.exp(-nu4d * tau) + fS * (-xp.expm1(-nu4d * tau))
