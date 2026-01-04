from src.solvers.base import *


class SolverKolgan(Solver):
    def _step(self, F, h, tau, coef_per=1):
        ZBC(F)
        sigma = minmod(F)
        F_r = F - 0.5 * sigma
        F_l = F + 0.5 * sigma
        Ws = W_god(F_l[:-1], F_r[1:], coef_per)
        F[1:-1] += coef_per * tau / h * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F[:, j, :, :], properties.h, tau, xi_v)
        F[1:-1] += prop_calc.get_J(F, properties) * tau