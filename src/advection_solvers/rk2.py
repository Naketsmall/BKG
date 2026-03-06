from src.advection_solvers.kolgan import SolverKolgan
from src.config.libloader import xp
from src.thermodynamics.model_properties import ModelProperties


class SolverRK(SolverKolgan):
    __name__ = "Kolgan+RK2"

    def __init__(self):
        super().__init__()
        self._rk_allocated = False

    def _alloc_rk(self, F):
        self._F0 = xp.zeros_like(F)
        self._rk_allocated = True

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):

        if not self._rk_allocated:
            self._alloc_rk(F)

        F0 = self._F0
        F0[:] = F

        self._step(F, t, tau, properties, prop_calc)
        self._step(F, t + tau, tau, properties, prop_calc)

        F[:] = 0.5 * (F0 + F)

        super()._calculate_collisions(F, tau, properties, prop_calc)