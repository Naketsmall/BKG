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
        # Нет проверки на деление на 0. Можно return np.where(n > 0, u/n, 0.0)
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

    def get_mu(self, T):
        return T**self.w

    def get_nu(self, n, T):
        return n * T ** (1-self.w) * 0.9/self.Kn
    """
    def calculate(self, CFL, t_max, step_f, right_part=True):
        tau = CFL * self.h / np.max(np.abs(self.xi))

        for step_i in range(int(t_max / tau)):
            print(f"calculation: {step_i} / {int(t_max / tau)}")

            if right_part:
                J = self.get_J()
            for j in range(len(self.xi)):
                xi_v = self.xi[j]
                if right_part:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, tau, xi_v, J[:, j, :, :])
                else:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, tau, xi_v)

        last_tau = t_max % tau
        if last_tau >= 1e-15:
            if right_part:
                J = self.get_J()
            for j in range(len(self.xi)):
                xi_v = self.xi[j]
                if right_part:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, last_tau, xi_v, J[:, j, :, :])
                else:
                    self.F[:, j, :, :] = step_f(self.F[:, j, :, :], self.h, last_tau, xi_v)
    """
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

        n_4d = n.reshape(-1, 1, 1, 1)
        u_4d = u.reshape(-1, 1, 1, 1)
        T_4d = np.maximum(T.reshape(-1, 1, 1, 1), 1e-12)
        q_4d = q.reshape(-1, 1, 1, 1)

        thermal_velocity_x = self.XI1[None, :, :, :] - u_4d
        thermal_velocity_y = np.broadcast_to(self.XI2[None, :, :, :], thermal_velocity_x.shape)
        thermal_velocity_z = np.broadcast_to(self.XI3[None, :, :, :], thermal_velocity_x.shape)

        thermal_velocity = np.stack([thermal_velocity_x,
                                     thermal_velocity_y,
                                     thermal_velocity_z], axis=-1)

        sqrtT = np.sqrt(T_4d)
        c = thermal_velocity / sqrtT[..., None]
        c_sq = np.sum(c ** 2, axis=-1)


        M = np.exp(-c_sq)
        Z = np.sum(M, axis=(1, 2, 3), keepdims=True) * (self.xi_cell_size ** 3)
        fM = n_4d * M / Z
        S = 2 * q_4d / (np.maximum(n_4d, 1e-12) * T_4d ** 1.5)

        shakhov_correction = (4 / 5) * (1 - self.Pr) * S * c[..., 0] * (c_sq - 2.5)
        fS = fM * (1 + shakhov_correction)

        nu =  self.get_nu(np.maximum(n, 1e-12), T) # T ** (1 - self.w) * (0.9 / self.Kn)
        nu_4d = nu.reshape(-1, 1, 1, 1)

        J = nu_4d * (fS - self.F[1:-1])
        return J
