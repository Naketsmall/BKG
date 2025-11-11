from src.solvers.base import *


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
