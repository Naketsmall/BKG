import numpy as np


class BKG:
    def __init__(self, config: dict):
        """
        :param n_x: количество граней между ячейками. Соответственно n_cells = n_x - 1
        """
        self.x = np.linspace(config['X_LEFT'], config['X_RIGHT'], config['n_x'])
        self.xi = np.linspace(config['XI_LEFT'], config['XI_RIGHT'], config['n_xi'])
        self.xi_cell_size = self.xi[1] - self.xi[0]
        self.dV = (self.xi[1] - self.xi[0]) ** 3
        self.h = self.x[1] - self.x[0]

        self.Kn = config['Kn']
        self.Pr = config['Pr']
        self.w = config['w']

        self.init_conditions(config)

    def init_conditions(self, config):

        n = np.zeros((len(self.x) + 1))
        n[1:-1] = config['F_BEG_N']( (self.x[:-1] + self.x[1:])/2 )

        u = np.zeros((len(self.x) + 1))
        u[1:-1] = config['F_BEG_U']( (self.x[:-1] + self.x[1:])/2 )

        T = np.zeros((len(self.x) + 1), dtype=np.float64)
        T[1:-1] = config['F_BEG_T']( (self.x[:-1] + self.x[1:])/2 )

        self.F = np.zeros((len(self.x) + 1, len(self.xi), len(self.xi), len(self.xi)))
        self.F[1:-1, :, :, :] = self.init_F_vectorized(n, u, T)

    def init_F_vectorized(self, n, u, T):
        self.XI1 = self.xi[:, None, None]
        self.XI2 = self.xi[None, :, None]
        self.XI3 = self.xi[None, None, :]
        self.XI_SQUARE = (self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2)

        n_4d = n[1:-1, None, None, None]
        u_4d = u[1:-1, None, None, None]
        T_4d = T[1:-1, None, None, None]

        v_sq = (self.XI1 - u_4d) ** 2 + self.XI2 ** 2 + self.XI3 ** 2

        M = np.exp(-v_sq / T_4d)

        Z = np.sum(M, axis=(1, 2, 3), keepdims=True) * self.dV
        F = n_4d * M / Z

        return F

    def get_n(self):
        return np.einsum('ijkl->i', self.F[1:-1]) * self.dV

    def get_u1(self, n):
        u_num = np.einsum('ijkl,j->i', self.F[1:-1], self.xi) * self.dV
        return u_num / np.maximum(n, 1e-15)

    def get_T(self, n, u):
        E = np.einsum('ijkl, jkl -> i', self.F[1:-1], self.XI_SQUARE) * self.dV
        return (2 / 3) * (E / np.maximum(n, 1e-15) - u ** 2)

    def get_q(self, u):
        xi1_shifted = self.XI1[None, :, :, :] - u[:, None, None, None]
        q = 0.5 * np.einsum('ijkl, ijkl -> i', xi1_shifted * self.XI_SQUARE, self.F[1:-1]) * self.dV
        return q

    def get_macros(self):
        n = self.get_n()
        u = self.get_u1(n)
        T = self.get_T(n, u)
        q = self.get_q(u)
        return n, u, T, q

    def get_mu(self, T):
        return T**self.w

    def get_nu(self, n, T):
        return n * T ** (1-self.w) * 0.9/self.Kn

    def calculate(self, CFL, t_max, step_f, right_part=True):
        t_cur = 0
        while t_cur < t_max:
            print(f"calculation: {t_cur} / {t_max}")
            n, u, T, q = self.get_macros()
            tau = min(CFL * self.h / np.max(np.abs(self.xi)), 1 / np.max(self.get_nu(n, T)), max(t_max - t_cur, 1e-15))
            if right_part:
                J = self.get_J()
            for j in range(len(self.xi)):
                xi_v = self.xi[j]
                if right_part:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, tau, xi_v, J[:, j, :, :])
                else:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, tau, xi_v)
            t_cur += tau


    def get_J(self):
        n, u, T, q = self.get_macros()

        n_4d = n[:, None, None, None]
        u_4d = u[:, None, None, None]
        T_4d = np.maximum(T[:, None, None, None], 1e-12)
        q_4d = q[:, None, None, None]

        tx = (self.XI1[None] - u_4d) / np.sqrt(T_4d)
        ty = self.XI2[None] / np.sqrt(T_4d)
        tz = self.XI3[None] / np.sqrt(T_4d)
        c_sq = tx * tx + ty * ty + tz * tz
        np.exp(-c_sq, out=c_sq)
        M = c_sq

        Z = np.sum(M, axis=(1, 2, 3), keepdims=True) * self.dV
        fM = n_4d * M / Z
        S = 2 * q_4d / (np.maximum(n_4d, 1e-12) * T_4d ** 1.5)

        shakhov_correction = (4 / 5) * (1 - self.Pr) * S * tx * (c_sq - 2.5)
        fS = fM * (1 + shakhov_correction)



        nu =  self.get_nu(np.maximum(n, 1e-12), T) # T ** (1 - self.w) * (0.9 / self.Kn)
        nu_4d = nu.reshape(-1, 1, 1, 1)

        J = nu_4d * (fS - self.F[1:-1])
        return J
