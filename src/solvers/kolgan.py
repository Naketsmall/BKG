from src.solvers.base import *


class SolverKolgan(Solver):
    __name__ = "Kolgan"

    def _step(self, F, h, tau, bc, xi):
        bc.apply(F)
        sigma = minmod(F)
        F_r = F - 0.5 * sigma
        F_l = F + 0.5 * sigma
        xi = xi[None, :, None, None]
        Ws = W_god(F_l[:-1], F_r[1:], xi)
        F[1:-1] += (tau / h) * xi * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        self._step(F, properties.h, tau, properties.bc, properties.xi)
        super()._calculate_collisions(F, tau, properties, prop_calc)

