import numpy as np
from abc import ABC, abstractmethod


def PBC(F):
    F[0] = F[-2]
    F[-1] = F[1]

def ZBC(F):
    F[0] = F[1]
    F[-1] = F[-2]

def minmod(F):
    F_left = np.roll(F, shift=1, axis=0)
    F_right = np.roll(F, shift=-1, axis=0)

    du_minus = F - F_left
    du_plus = F_right - F

    sigma = np.zeros_like(F)
    mask = (du_minus * du_plus) > 0
    sigma[mask] = np.sign(du_minus[mask]) * np.minimum(
        np.abs(du_minus[mask]),
        np.abs(du_plus[mask])
    )

    sigma[0] = 0
    sigma[-1] = 0

    return sigma


def W_god(u_l, u_r, coef_per=1):
    return 0.5 * np.sign(coef_per) * ((1 + np.sign(coef_per)) * u_l + (-1 + np.sign(coef_per)) * u_r)


class Solver(ABC):
    @abstractmethod
    def step(self, F, h, tau, coef_per=1, J=None):
        pass


class SolverGodunov(Solver):
    def step(self, F, h, tau, coef_per=1, J=None):
        ZBC(F)
        F_r = F
        F_l = F
        Ws = W_god(F_l[:-1], F_r[1:], coef_per)
        F[1:-1] += coef_per * tau / h * (Ws[:-1] - Ws[1:])
        if J is not None:
            F[1:-1] += tau * J
        return F

class SolverKolgan(Solver):
    def step(self, F, h, tau, coef_per=1, J=None):
        ZBC(F)
        sigma = minmod(F)
        F_r = F - 0.5 * sigma
        F_l = F + 0.5 * sigma
        Ws = W_god(F_l[:-1], F_r[1:], coef_per)
        F[1:-1] += coef_per * tau / h * (Ws[:-1] - Ws[1:])
        if J is not None:
            F[1:-1] += tau * J
        return F

class SolverRK(SolverKolgan):
    def step(self, F, h, tau, coef_per=1, J=None):
        F_pred = super().step(F.copy(), h, tau, coef_per, J)
        F_corr = super().step(F_pred, h, tau, coef_per, J)
        F_new = 0.5 * (F + F_corr)
        return F_new





class SolverL3(Solver):
    pass