"""
from src.config.libloader import xp, cuda_is_available
from src.solvers.base import Solver, ZBC
from src.thermodynamics import ModelProperties


def weno5_flux_positive(f, eps=1e-6):

    f_im2 = xp.roll(f, 2)
    f_im1 = xp.roll(f, 1)
    f_i   = f
    f_ip1 = xp.roll(f, -1)
    f_ip2 = xp.roll(f, -2)

    # Кандидатные потоки
    f0 = (1/3)*f_im2 - (7/6)*f_im1 + (11/6)*f_i
    f1 = -(1/6)*f_im1 + (5/6)*f_i + (1/3)*f_ip1
    f2 = (1/3)*f_i + (5/6)*f_ip1 - (1/6)*f_ip2

    # Индикаторы гладкости
    beta0 = (
        (13/12)*(f_im2 - 2*f_im1 + f_i)**2 +
        (1/4)*(f_im2 - 4*f_im1 + 3*f_i)**2
    )
    beta1 = (
        (13/12)*(f_im1 - 2*f_i + f_ip1)**2 +
        (1/4)*(f_im1 - f_ip1)**2
    )
    beta2 = (
        (13/12)*(f_i - 2*f_ip1 + f_ip2)**2 +
        (1/4)*(3*f_i - 4*f_ip1 + f_ip2)**2
    )

    # Оптимальные веса
    d0, d1, d2 = 0.1, 0.6, 0.3

    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha2 = d2 / (eps + beta2)**2

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    return w0*f0 + w1*f1 + w2*f2


def weno5_flux_negative(f, eps=1e-6):
    return xp.roll(
        weno5_flux_positive(xp.flip(f), eps),
        1
    )[::-1]


class SolverWENO5(Solver):

    def _step(self, F, h, tau, coef_per=1):
        ZBC(F)

        alpha = xp.abs(coef_per)

        f = coef_per * F

        f_plus  = 0.5 * (f + alpha * F)
        f_minus = 0.5 * (f - alpha * F)

        flux_p = weno5_flux_positive(f_plus)
        flux_m = weno5_flux_negative(f_minus)

        flux = flux_p + flux_m

        F[2:-2] -= tau / h * (flux[3:-1] - flux[2:-2])


class SolverWENO5RK3(SolverWENO5):

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        F0 = F.copy()

        # ---------- RK stage 1 ----------
        for j in range(len(properties.xi)):
            self._step(
                F[:, j, :, :],
                properties.h,
                tau,
                properties.xi[j]
            )

        F1 = F.copy()

        # ---------- RK stage 2 ----------
        for j in range(len(properties.xi)):
            self._step(
                F[:, j, :, :],
                properties.h,
                tau,
                properties.xi[j]
            )

        F[:] = 0.75 * F0 + 0.25 * F
        F2 = F.copy()

        # ---------- RK stage 3 ----------
        for j in range(len(properties.xi)):
            self._step(
                F[:, j, :, :],
                properties.h,
                tau,
                properties.xi[j]
            )

        F[:] = (1/3) * F0 + (2/3) * F

        # ---------- collisions ----------
        super()._calculate_collisions(F, tau, properties, prop_calc)
"""