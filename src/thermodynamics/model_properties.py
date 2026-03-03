from src.thermodynamics.boundary_condition import BoundaryCondition
from src.config.libloader import xp
from src.mesh import Mesh


class ModelProperties:
    def __init__(self, config: dict, mesh: Mesh, bc: BoundaryCondition):

        self.mesh = mesh
        self.bc = bc

        self.xi = xp.linspace(config['XI_LEFT'], config['XI_RIGHT'], config['n_xi'])
        self.xi_cell_size = self.xi[1] - self.xi[0]
        self.dV = (self.xi[1] - self.xi[0]) ** 3
        self.XI1 = self.xi[:, None, None]
        self.XI2 = self.xi[None, :, None]
        self.XI3 = self.xi[None, None, :]
        self.XI_SQUARE = (self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2)

        self.Kn = config['Kn']
        self.Pr = config['Pr']
        self.w = config['w']