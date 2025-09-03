import numpy as np

class BKG:
    def __init__(self, config: dict):
        """
        :param n_x: количество граней между ячейками. Соответственно n_cells = n_x - 1
        """
        self.x = np.linspace(config['X_LEFT'], config['X_RIGHT'], config['n_x'])
        self.xi = np.linspace(config['XI_LEFT'], config['XI_RIGHT'], config['n_xi'])
        self.xi_cell_size = self.xi[1] - self.xi[0]
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

        T = np.zeros((len(self.x) + 1))
        T[1:-1] = config['F_BEG_T']( (self.x[:-1] + self.x[1:])/2 )

        self.F = np.zeros((len(self.x) + 1, len(self.xi), len(self.xi), len(self.xi)))
        self.F[1:-1, :, :, :] = self.init_F_vectorized(n, u, T)

    def init_F_vectorized(self, n, u, T):
        self.XI1, self.XI2, self.XI3 = np.meshgrid(self.xi, self.xi, self.xi, indexing='ij')
        self.XI_SQUARE = (self.XI1 ** 2 + self.XI2 ** 2 + self.XI3 ** 2)

        n_4d = n[1:-1].reshape(-1, 1, 1, 1)
        u_4d = u[1:-1].reshape(-1, 1, 1, 1)
        T_4d = T[1:-1].reshape(-1, 1, 1, 1)

        v_sq = (self.XI1 - u_4d) ** 2 + self.XI2 ** 2 + self.XI3 ** 2
        M = np.exp(-v_sq / T_4d)

        # дискретная нормировка по n
        Z = np.sum(M, axis=(1, 2, 3), keepdims=True) * (self.xi_cell_size ** 3)
        F = n_4d * M / Z

        return F

    def get_n(self):
        return np.sum(self.F[1:-1] * self.xi_cell_size**3, axis=(1, 2, 3))

    def get_u1(self, n):
        return np.sum(self.XI1 * self.F[1:-1] * (self.xi_cell_size ** 3), axis=(1, 2, 3)) / n

    def get_T(self, n, u):
        return 2 /3 * (np.sum(self.XI_SQUARE * self.F[1:-1] * self.xi_cell_size**3, axis=(1, 2, 3)) / n - u ** 2)

    def get_q(self, u):
        q = 0.5 * np.sum(((self.XI1 - u.reshape(-1, 1, 1, 1)) * self.XI_SQUARE * self.F[1:-1])
                         * self.xi_cell_size**3, axis=(1, 2, 3))
        return q

    def get_macros(self):
        n = self.get_n()
        u = self.get_u1(n)
        T = self.get_T(n, u)
        q = self.get_q(u)
        return n, u, T, q

    def calculate(self, CFL, t_max, step_f):
        tau = CFL * self.h / np.max(np.abs(self.xi))

        for step_i in range(int(t_max / tau)):
            print(f"calculation: {step_i} / {int(t_max / tau)}")

            for j in range(len(self.xi)):
                xi_v = self.xi[j]
                self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, tau, xi_v)

        last_tau = t_max % tau
        for j in range(len(self.xi)):
            xi_v = self.xi[j]
            self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, last_tau, xi_v)

    """
    def get_mu(self, T, w):
        return T ** w

    def get_nu(self, n, T, w):
        return n * T / get_mu(T, w) * 0.9 / Kn

    def get_Fs(self):
        pass
    """


















