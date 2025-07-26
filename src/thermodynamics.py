import numpy as np
from config.configuration import F_BEG_N, F_BEG_T, F_BEG_U


class BKG:
    def __init__(self, n_x, n_xi):
        self.x = np.linspace(0, 10, n_x)
        self.xi = np.linspace(-20, 20, n_xi)
        self.xi_cell_size = self.xi[1] - self.xi[0]
        self.h = self.x[1] - self.x[0]
        self.init_conditions()

    def init_conditions(self):

        n = np.zeros((len(self.x) + 1))
        n[1:-1] = F_BEG_N(self.x[:-1] + self.h / 2)

        u = np.zeros((len(self.x) + 1))
        u[1:-1] = F_BEG_U(self.x[:-1] + self.h / 2)

        T = np.zeros((len(self.x) + 1))
        T[1:-1] = F_BEG_T(self.x[:-1] + self.h / 2)

        self.F = np.zeros((len(self.x) + 1, len(self.xi), len(self.xi), len(self.xi)))
        self.F[1:-1, :, :, :] = self.init_F_vectorized(n, u, T)

    def init_F_vectorized(self, n, u, T):
        self.XI1, self.XI2, self.XI3 = np.meshgrid(self.xi, self.xi, self.xi, indexing='ij')
        n_cells = len(self.x) - 1

        n_4d = n[1:-1].reshape(-1, 1, 1, 1)
        u_4d = u[1:-1].reshape(-1, 1, 1, 1)
        T_4d = T[1:-1].reshape(-1, 1, 1, 1)

        A = n_4d / ((np.pi * T_4d) ** (1.5))
        v_sq = (self.XI1 - u_4d) ** 2 + self.XI2 ** 2 + self.XI3 ** 2

        F = A * np.exp(-v_sq / T_4d)
        return F

    def get_n(self):
        return np.sum(self.F[1:-1] * self.xi_cell_size**3, axis=(1, 2, 3))

    def get_u1(self, n):
        return self.xi @ np.sum(self.F[1:-1] * self.xi_cell_size**3, axis=(2, 3)).T / n

    def get_T(self, n, u):
        return 2 * (np.sum((self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2) * self.F[1:-1] * self.xi_cell_size**3, axis=(1, 2, 3)) / n - u ** 2) / 3

    def get_q(self, u):
        q = 0.5 * np.sum(((self.XI1 - u.reshape(-1, 1, 1, 1)) * (self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2) * self.F[1:-1])
                         * self.xi_cell_size, axis=(1, 2, 3))
        return q

    def get_macros(self):
        n = self.get_n()
        u = self.get_u1(n)
        T = self.get_T(n, u)
        q = self.get_q(u)
        return n, u, T, q

    """
    def get_mu(self, T, w):
        return T ** w

    def get_nu(self, n, T, w):
        return n * T / get_mu(T, w) * 0.9 / Kn

    def get_Fs(self):
        pass
    """


















