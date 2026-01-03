from src.solvers.base import *


class SolverKolgan(Solver):
    def _step(self, F, h, tau, coef_per=1, v_right_part=None):
        ZBC(F)
        sigma = minmod(F)
        F_r = F - 0.5 * sigma
        F_l = F + 0.5 * sigma
        Ws = W_god(F_l[:-1], F_r[1:], coef_per)
        F[1:-1] += coef_per * tau / h * (Ws[:-1] - Ws[1:])
        if v_right_part is not None:
            F[1:-1] += tau * v_right_part
        return F

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        pass
