from src.solvers.base import Solver
from src.config.libloader import xp
from src.thermodynamics import ModelProperties


class WENO5RK3(Solver):
    __name__ = "WENO5RK3"

    def __init__(self, eps=1e-12):
        self.eps = eps

    def _weno5(self, f):
        eps = self.eps

        im2 = f[:-5]
        im1 = f[1:-4]
        i0 = f[2:-3]
        ip1 = f[3:-2]
        ip2 = f[4:-1]

        beta0 = (13 / 12) * (im2 - 2 * im1 + i0) ** 2 + (1 / 4) * (im2 - 4 * im1 + 3 * i0) ** 2
        beta1 = (13 / 12) * (im1 - 2 * i0 + ip1) ** 2 + (1 / 4) * (im1 - ip1) ** 2
        beta2 = (13 / 12) * (i0 - 2 * ip1 + ip2) ** 2 + (1 / 4) * (3 * i0 - 4 * ip1 + ip2) ** 2

        d0, d1, d2 = 0.1, 0.6, 0.3

        a0 = d0 / (eps + beta0) ** 2
        a1 = d1 / (eps + beta1) ** 2
        a2 = d2 / (eps + beta2) ** 2

        asum = a0 + a1 + a2

        w0 = a0 / asum
        w1 = a1 / asum
        w2 = a2 / asum

        q0 = (1 / 3) * im2 - (7 / 6) * im1 + (11 / 6) * i0
        q1 = -(1 / 6) * im1 + (5 / 6) * i0 + (1 / 3) * ip1
        q2 = (1 / 3) * i0 + (5 / 6) * ip1 - (1 / 6) * ip2

        return w0 * q0 + w1 * q1 + w2 * q2

    def _step(self, F, h, tau, bc, xi):
        ng = bc.n_ghost
        N_x = F.shape[0]

        bc.apply(F)
        xi = xi[None, :, None, None]
        alpha = xp.abs(xi)

        f = xi * F
        f_p = 0.5 * (f + alpha * F)
        f_m = 0.5 * (f - alpha * F)

        flux_p = self._weno5(f_p)
        flux_m = self._weno5(f_m[::-1])[::-1]

        flux = flux_p + flux_m

        rhs = xp.zeros_like(F)

        rhs[ng:-ng] = -(
                flux[ng - 2: N_x - ng - 2] -
                flux[ng - 3: N_x - ng - 3]
        ) / h

        return rhs

    def calculate_layer(self, F, tau, properties: ModelProperties, prop_calc):
        h = properties.h
        xi = properties.xi
        bc = properties.bc

        F0 = F.copy()

        k1 = self._step(F, h, tau, bc, xi)
        F1 = F + tau * k1

        k2 = self._step(F1, h, tau, bc, xi)
        F2 = 0.75 * F0 + 0.25 * (F1 + tau * k2)

        k3 = self._step(F2, h, tau, bc, xi)
        F[:] = (1 / 3) * F0 + (2 / 3) * (F2 + tau * k3)

        super()._calculate_collisions(F, tau, properties, prop_calc)

