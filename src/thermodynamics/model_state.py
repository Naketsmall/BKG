from src.thermodynamics.model_properties import ModelProperties

from src.config.libloader import xp


class ModelState:

    def __init__(self, properties: ModelProperties, config: dict):
        self.F = None
        self.init_conditions(properties, config)

    def init_conditions(self, properties: ModelProperties, config: dict):

        N = len(properties.mesh.x)

        n = xp.zeros((len(properties.mesh.x) + 1), dtype=xp.float64)
        n[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_N'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        u = xp.zeros((len(properties.mesh.x) + 1), dtype=xp.float64)
        u[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_U'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        T = xp.zeros((len(properties.mesh.x) + 1), dtype=xp.float64)
        T[properties.bc.n_ghost:-properties.bc.n_ghost] = config['F_BEG_T'](
            (properties.mesh.x[properties.bc.n_ghost-1 : N-properties.bc.n_ghost] +
             properties.mesh.x[properties.bc.n_ghost : N-properties.bc.n_ghost+1])/2
        )

        self.F = xp.zeros((len(properties.mesh.x) + 1, len(properties.xi), len(properties.xi), len(properties.xi)))
        self.F[properties.bc.n_ghost:-properties.bc.n_ghost, :, :, :] = self.init_F_vectorized(n, u, T, properties, properties.bc.n_ghost)

    @staticmethod
    def init_F_vectorized(n, u, T, properties: ModelProperties, n_ghost):
        """
        Инициализирует 4D [x, xi1, xi2, xi3] пространство локальным максвеллианом с параметрами (n, u, T)
        :param n_ghost колмчество граничных ячеек.
        Z вычисляется дискретно как сумма, а не аналитически
        """

        n_4d = n[n_ghost:-n_ghost, None, None, None]
        u_4d = u[n_ghost:-n_ghost, None, None, None]
        T_4d = T[n_ghost:-n_ghost, None, None, None]

        v_sq = (properties.XI1 - u_4d) ** 2 + properties.XI2 ** 2 + properties.XI3 ** 2

        M = xp.exp(-v_sq / T_4d)

        Z = xp.sum(M, axis=(1, 2, 3), keepdims=True) * properties.dV
        F = n_4d * M / Z

        return F

    def get_F(self):
        return self.F

    def set_F(self, F):
        if F.shape != self.F.shape:
            raise ValueError(f"Shape mismatch error: F.shape: {F.shape} != self.F.shape: {self.F.shape}")
        self.F = F
