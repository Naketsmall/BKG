from src.thermodynamics.boundary_condition import BoundaryCondition
from src.config.libloader import xp
from src.mesh import Mesh


class ModelProperties:
    def __init__(self, config: dict, mesh: Mesh, bc: BoundaryCondition):

        self.mesh = mesh
        self.bc = bc

        self.xi = xp.linspace(config['XI_LEFT'], config['XI_RIGHT'], config['n_xi'])
        self.xi_cell_size = self.xi[1] - self.xi[0]
        self.dV = self.xi_cell_size ** 3


        self.xi_sq = self.xi ** 2

        self.xi_x = self.xi[:, None, None]
        self.xi_y = self.xi[None, :, None]
        self.xi_z = self.xi[None, None, :]

        self.xi_sq_x = self.xi_sq[:, None, None]
        self.xi_sq_y = self.xi_sq[None, :, None]
        self.xi_sq_z = self.xi_sq[None, None, :]


        self.Kn = config['Kn']
        self.Pr = config['Pr']
        self.w = config['w']