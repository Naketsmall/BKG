from abc import ABC, abstractmethod
from src.config.libloader import xp


class Mesh(ABC):

    def __init__(self, x, n_ghost):
        dx_l, dx_r = x[1] - x[0], x[-1] - x[-2]

        idx = xp.arange(1, n_ghost + 1)

        l_ghost = x[0] - dx_l*idx[::-1]
        r_ghost = x[-1] + dx_r * idx

        self.x = xp.concatenate([l_ghost, x, r_ghost])
        self.dx = self.x[1:] - self.x[:-1]

    def get_centers(self):
        return self.x[:-1] + self.get_dx()/2

    @abstractmethod
    def get_dx(self):
        pass

    @abstractmethod
    def update_mesh(self):
        pass

    @abstractmethod
    def adapt(self, split_ids, merge_ids):
        pass


class UnadaptableMesh(Mesh):
    def __init__(self, x, n_ghost):
        super().__init__(x, n_ghost)

    def get_dx(self):
        return self.dx

    def update_mesh(self):
        pass

    def adapt(self, split_ids, merge_ids):
        pass



class MeshAdapter():
    pass