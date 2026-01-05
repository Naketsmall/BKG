from src.solvers.base import *
from src.config.libloader import xp


class SolverGodunov(Solver):

    def _step(self, F, h, tau, bc, xi):
        bc.apply(F)

        F_l = F[:-1]
        F_r = F[1:]

        xi = xi[None, :, None, None]
        Ws = W_god(F_l, F_r, xi)

        F[1:-1] += (tau / h) * xi * (Ws[:-1] - Ws[1:])

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        self._step(F, properties.h, tau, properties.bc, properties.xi)
        super()._calculate_collisions(F, tau, properties, prop_calc)

