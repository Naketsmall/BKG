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
        """
        Инициализирует 4D [x, xi1, xi2, xi3] пространство локальным максвеллианом с параметрами (n, u, T)
        :param n_ghost колмчество граничных ячеек.
        Z вычисляется дискретно как сумма, а не аналитически
        """
        N = n.shape[0]

        n_4d = n[n_ghost:N-n_ghost, None, None, None]
        u_4d = u[n_ghost:N-n_ghost, None, None, None]
        T_4d = T[n_ghost:N-n_ghost, None, None, None]

        v_sq = (properties.XI1 - u_4d) ** 2 + properties.XI2 ** 2 + properties.XI3 ** 2

        M = xp.exp(-v_sq / T_4d)

        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV
        F = n_4d * M / Z

        return F


