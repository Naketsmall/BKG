from src.solvers.kolgan import SolverKolgan
from src.thermodynamics import ModelProperties


class SolverRK(SolverKolgan):
    def _step(self, F, h, tau, coef_per=1, v_right_part=None):
        F = super()._step(F, h, tau, coef_per, v_right_part)
        return F

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        F1 = F.copy()
        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F1[:, j, :, :], properties.h, tau, xi_v)
        F1[1:-1] += prop_calc.get_J(F1, properties) * tau
        F2 = F1.copy()
        for j in range(len(properties.xi)):
            xi_v = properties.xi[j]
            self._step(F2[:, j, :, :], properties.h, tau, xi_v)
        F2[1:-1] += prop_calc.get_J(F2, properties) * tau
        F = (F + F2)/2
