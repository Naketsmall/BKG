from src.solvers.base import *
from src.config.libloader import xp


class SolverGodunov(Solver):
    def _step(self, F, h, tau, coef_per=1):
        ZBC(F)
        F_r = F
        F_l = F
        Ws = W_god(F_l[:-1], F_r[1:], coef_per)
        F[1:-1] += coef_per * tau / h * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F[:, j, :, :], properties.h, tau, xi_v)
        # F[1:-1] += prop_calc.get_J(F, properties) * tau

        super()._calculate_collisions(F, tau, properties, prop_calc)
        #(1 - xp.exp(-nu4d * tau))
        #return F

