from src.solvers.kolgan import SolverKolgan
from src.thermodynamics import ModelProperties


class SolverRK(SolverKolgan):
    __name__ = "Kolgan+RK2"

    def _step(self, F, t, tau, properties: ModelProperties, prop_calc):
        super()._step(F, t, tau, properties, prop_calc)

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        F0 = F.copy()
        F1 = F
        self._step(F1, t, tau, properties, prop_calc)
        self._step(F1, t, tau, properties, prop_calc)
        F[:] = 0.5 * (F0 + F1)
        super()._calculate_collisions(F, tau, properties, prop_calc)
