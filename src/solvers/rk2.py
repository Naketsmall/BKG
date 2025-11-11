from src.solvers.kolgan import SolverKolgan


class SolverRK(SolverKolgan):
    def step(self, F, h, tau, coef_per=1, J=None):
        F_pred = super().step(F.copy(), h, tau, coef_per, J)
        F_corr = super().step(F_pred, h, tau, coef_per, J)
        F_new = 0.5 * (F + F_corr)
        return F_new