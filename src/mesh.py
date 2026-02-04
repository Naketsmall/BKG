from abc import ABC, abstractmethod


class Mesh(ABC):

    def __init__(self, x):
        self.x = x

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
    def __init__(self, x):
        super().__init__(x)
        self.dx = self.x[1:] - self.x[:-1]

    def get_dx(self):
        return self.dx

    def update_mesh(self):
        pass

    def adapt(self, split_ids, merge_ids):
        pass



class MeshAdapter():
    pass