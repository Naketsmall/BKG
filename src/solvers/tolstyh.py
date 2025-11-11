import numpy as np
from numba import njit

from src.solvers.base import Solver


def thomas_solve(A, D):
    """
    Решает систему A @ x = D
    A : (n, n)
    D : (..., n) или (n, batch_shape...)

    Возвращает массив той же формы, что и D
    """
    n = A.shape[0]

    # Диагонали
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]

    # Приводим D в форму (batch, n)
    D = np.moveaxis(D, 0, -1)  # теперь D.shape = (..., n)
    batch_shape = D.shape[:-1]
    D = D.reshape(-1, n)  # склеиваем батчи

    X = np.zeros_like(D)
    for i in range(D.shape[0]):
        X[i] = _thomas_algorithm_optimized(a.copy(), b.copy(), c.copy(), D[i].copy())

    X = X.reshape(*batch_shape, n)
    return np.moveaxis(X, -1, 0)  # возвращаем обратно


@njit(fastmath=True)
def _thomas_algorithm_optimized(a, b, c, d):
    """Классический метод прогонки для одной RHS"""
    n = len(d)
    x = np.zeros(n)

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def delta0(n):
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix[1:], -1)
    np.fill_diagonal(matrix[:, 1:], 1)
    return matrix


def delta2(n):
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix[1:], 1)
    np.fill_diagonal(matrix, -2)
    np.fill_diagonal(matrix[:, 1:], 1)
    return matrix


def s_interp(s):
    return 0.5 * (s[:-1] + s[1:])


def delta(s, n):
    return 0.5 * (delta0(n) - s * delta2(n))


def delta_cons(s, n):
    return 0.5 * (delta0(n) - (s[1:] * 1 / 2 * (delta0(n) + delta2(n)) - s[:-1] * 1 / 2 * (delta0(n) - delta2(n))))


def Ah(s, n):
    return np.eye(n) + delta2(n) / 6 - s / 4 * delta0(n)


def Ah_cons(s, n):
    return np.eye(n) + 1 / 6 * delta2(n) - 1 / 4 * (s[1:] * 1 / 2 * (delta2(n) + delta0(n) + 4 * np.eye(n)) - \
                                                    s[:1] * 1 / 2 * (delta2(n) - delta0(n) + 4 * np.eye(n)))


def Bh(s, h, n):
    return 1 / h * delta(s, n)


def Bh_cons(s, h, n):
    return 1 / h * delta_cons(s, n)


def get_v(u, A, B, n):
    u_pred = np.tensordot(B, u, axes=(1, 0))
    v = thomas_solve(A, u_pred)
    return v


def step_SSP3(u, A, B, dt, coef_per=1):
    F = lambda f: get_v(f, A, B, len(f)) * (-coef_per)
    u1 = u + dt * F(u)
    u2 = 3 / 4 * u + 1 / 4 * (u1 + dt * F(u1))
    return 1 / 3 * u + 2 / 3 * (u2 + dt * F(u2))


def step_SSP5(u, A, B, dt, coef_per=1):
    F = lambda f: get_v(f, A, B, len(f)) * (-coef_per)
    u1 = u + 0.39175222700392 * dt * F(u)
    u2 = 0.44437049406734 * u + 0.55562950593266 * u1 + 0.36841059262959 * dt * F(u1)
    u3 = 0.62010185138540 * u + 0.37989814861460 * u2 + 0.25189177424738 * dt * F(u2)
    u4 = 0.17807995410773 * u + 0.82192004589227 * u3 + 0.54497475021237 * dt * F(u3)
    return 0.00683325884039 * u + 0.51723167208978 * u2 + 0.12759831133288 * u3 \
        + 0.34833675773694 * u4 + 0.08460416338212 * dt * F(u3) + 0.22600748319395 * dt * F(u4)


def ZBC2a(F):
    F[0] = F[1]
    F[-1] = F[-2]
    return np.vstack(([F[0], F[0]], F, [F[-1], F[-1]]))


class SolverL3(Solver):
    def __init__(self, n_x, h):
        #s = np.array([(4 / 15) ** 0.5 for i in range(n_x + 5)])
        s = np.array([0 for i in range(n_x + 5)])
        self.Ap, self.Bp = Ah_cons(s, n_x + 4), Bh_cons(s, h, n_x + 4) # s зависит от знка направления переноса
        self.An, self.Bn = Ah_cons(-s, n_x + 4), Bh_cons(-s, h, n_x + 4)

    def step(self, F, h, tau, coef_per=1, J=None):
        if coef_per >= 1:
            F = step_SSP3(ZBC2a(F), self.Ap, self.Bp, tau, coef_per)[2:-2]
        else:
            F = step_SSP3(ZBC2a(F), self.An, self.Bn, tau, coef_per)[2:-2]
        if J is not None:
            F[1:-1] += tau * J
        return F

