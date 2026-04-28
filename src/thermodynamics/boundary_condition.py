from abc import ABC, abstractmethod
from src.config.libloader import xp


class BoundaryCondition(ABC):
    def __init__(self, n_ghost):
        self.n_ghost = n_ghost # Количество клеток с ГУ с одной стороны

    @abstractmethod
    def apply(self, F, t, properties, prop_calc):
        pass


class PeriodicBoundaryCondition(BoundaryCondition):

    def apply(self, F, t, properties, prop_calc):
        F[: self.n_ghost] = F[-2*self.n_ghost : -self.n_ghost]
        F[-self.n_ghost :] = F[self.n_ghost : 2*self.n_ghost]

class ZeroGradBoundaryCondition(BoundaryCondition):

    def apply(self, F, t, properties, prop_calc):
        F[: self.n_ghost] = F[self.n_ghost]
        F[-self.n_ghost:] = F[-self.n_ghost-1]

class EvapCondBoundaryCondition(BoundaryCondition):
    def __init__(self, n_ghost, n_f, T_f):
        super().__init__(n_ghost)
        self.n_f = n_f
        self.T_f = T_f

    def apply(self, F, t, properties, prop_calc):

        n_t = self.n_f(t)
        T_t = self.T_f(t)

        n = xp.full(self.n_ghost, n_t)
        u = xp.zeros(self.n_ghost)
        T = xp.full(self.n_ghost, T_t)

        Fw = EvapCondBoundaryCondition.init_F_vectorized(
            n, u, T, properties, n_ghost=0
        )

        F[-self.n_ghost:] = F[-self.n_ghost - 1]
        F[:self.n_ghost] = Fw

    @staticmethod
    def init_F_vectorized(n, u, T, properties, n_ghost):
        N = n.shape[0]

        n_loc = n[n_ghost:N - n_ghost]
        u_loc = u[n_ghost:N - n_ghost]
        T_loc = T[n_ghost:N - n_ghost]

        xi = properties.xi
        dxi = properties.xi_cell_size


        M1 = xp.exp(-(xi[None, :] - u_loc[:, None]) ** 2 / T_loc[:, None])
        M2 = xp.exp(-(xi[None, :] ** 2) / T_loc[:, None])
        M3 = xp.exp(-(xi[None, :] ** 2) / T_loc[:, None])


        Z1 = xp.sum(M1, axis=1) * dxi
        Z2 = xp.sum(M2, axis=1) * dxi
        Z3 = xp.sum(M3, axis=1) * dxi

        Z = Z1 * Z2 * Z3

        F = (
                n_loc[:, None, None, None]
                * M1[:, :, None, None]
                * M2[:, None, :, None]
                * M3[:, None, None, :]
                / Z[:, None, None, None]
        )

        return F


