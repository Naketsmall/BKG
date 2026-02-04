from src.solvers.base import *


class SolverKolgan(Solver):
    __name__ = "Kolgan"

    def _step(self, F, t, tau, properties: ModelProperties, prop_calc):
        properties.bc.apply(F, t, properties, prop_calc)
        sigma = minmod(F)
        F_r = F - 0.5 * sigma
        F_l = F + 0.5 * sigma
        xi = properties.xi[None, :, None, None]
        Ws = W_god(F_l[:-1], F_r[1:], xi)
        dx = properties.mesh.get_dx()[:, None, None, None]
        F[1:-1] += (tau / dx) * xi * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        self._step(F, t, tau, properties, prop_calc)
        super()._calculate_collisions(F, tau, properties, prop_calc)

