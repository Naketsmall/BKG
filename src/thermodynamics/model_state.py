from src.thermodynamics.model_properties import ModelProperties

from src.config.libloader import xp


class ModelState:

    def __init__(self, properties: ModelProperties, config: dict):
        self.F = None
        self.init_conditions(properties, config)

    def init_conditions(self, properties: ModelProperties, config: dict):

        N = len(properties.mesh.x)

        n = xp.zeros((len(properties.mesh.x) + 1), dtype=config['dtype'])
        n[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_N'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        u = xp.zeros((len(properties.mesh.x) + 1), dtype=config['dtype'])
        u[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_U'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        T = xp.zeros((len(properties.mesh.x) + 1), dtype=config['dtype'])
        T[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_T'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        self.F = xp.zeros((len(properties.mesh.x) + 1, len(properties.xi), len(properties.xi), len(properties.xi)))
        self.F[properties.bc.n_ghost:-properties.bc.n_ghost, :, :, :] = self.init_F_vectorized(n, u, T, properties, properties.bc.n_ghost)

    @staticmethod
    def init_F_vectorized(n, u, T, properties: ModelProperties, n_ghost):
        n_loc = n[n_ghost:-n_ghost]
        u_loc = u[n_ghost:-n_ghost]
        T_loc = T[n_ghost:-n_ghost]

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

    def get_F(self):
        return self.F

    def set_F(self, F):
        if F.shape != self.F.shape:
            raise ValueError(f"Shape mismatch error: F.shape: {F.shape} != self.F.shape: {self.F.shape}")
        self.F = F
