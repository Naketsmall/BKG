from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    def __init__(self, n_ghost):
        self.n_ghost = n_ghost # Количество клеток с ГУ с одной стороны

    @abstractmethod
    def apply(self, F):
        pass


class PeriodicBoundaryCondition(BoundaryCondition):
    def __init__(self, n_ghost):
        super().__init__(n_ghost)

    def apply(self, F):
        F[: self.n_ghost] = F[-2*self.n_ghost : -self.n_ghost]
        F[-self.n_ghost :] = F[self.n_ghost : 2*self.n_ghost]

class ZeroGradBoundaryCondition(BoundaryCondition):
    def __init__(self, n_ghost):
        super().__init__(n_ghost)

    def apply(self, F):
        F[: self.n_ghost] = F[self.n_ghost]
        F[-self.n_ghost:] = F[-self.n_ghost-1]
