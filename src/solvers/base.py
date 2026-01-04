from src.config.libloader import xp, cuda_is_available
from abc import ABC, abstractmethod

from src.thermodynamics import ModelProperties


def PBC(F):
    F[0] = F[-2]
    F[-1] = F[1]

def ZBC(F):
    F[0] = F[1]
    F[-1] = F[-2]

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


def W_god(u_l, u_r, coef_per=1):
    return xp.where(coef_per >= 0, u_l, u_r)


class Solver(ABC):
    @abstractmethod
    def _step(self, F, h, tau, coef_per=1):
        pass

    @abstractmethod
    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        pass
