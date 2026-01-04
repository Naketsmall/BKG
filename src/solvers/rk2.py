from src.solvers.kolgan import SolverKolgan
from src.thermodynamics import ModelProperties


class SolverRK(SolverKolgan):

    def _step(self, F, h, tau, coef_per=1):
        super()._step(F, h, tau, coef_per)

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        F0 = F.copy()
        F1 = F
        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F1[:, j, :, :], properties.h, tau, xi_v)

        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F1[:, j, :, :], properties.h, tau, xi_v)
        F[:] = 0.5 * (F0 + F1)

        super()._calculate_collisions(F, tau, properties, prop_calc)
