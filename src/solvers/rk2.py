from src.solvers.kolgan import SolverKolgan
from src.thermodynamics import ModelProperties


class SolverRK(SolverKolgan):

    def _step(self, F, h, tau, bc, xi):
        super()._step(F, h, tau, bc, xi)

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        F0 = F.copy()
        F1 = F
        self._step(F1, properties.h, tau, properties.bc, properties.xi)
        self._step(F1, properties.h, tau, properties.bc, properties.xi)
        F[:] = 0.5 * (F0 + F1)
        super()._calculate_collisions(F, tau, properties, prop_calc)
