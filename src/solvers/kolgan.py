from src.solvers.base import *
from src.config.libloader import xp


class SolverKolgan(Solver):
    __name__ = "Kolgan (fast)"

    def __init__(self):
        self._buffers_allocated = False

    def _minmod(self, F, sigma):
        sigma.fill(0)

        du_minus = F[1:-1] - F[:-2]
        du_plus = F[2:] - F[1:-1]

        mask = du_minus * du_plus > 0

        sigma_inner = sigma[1:-1]

        sigma_inner[mask] = xp.sign(du_minus[mask]) * xp.minimum(
            xp.abs(du_minus[mask]),
            xp.abs(du_plus[mask])
        )

    def _alloc_buffers(self, F):
        self._sigma = xp.zeros_like(F)
        self._buffers_allocated = True

    def _step(self, F, t, tau, properties, prop_calc):
        if not self._buffers_allocated:
            self._alloc_buffers(F)

        ng = properties.bc.n_ghost

        properties.bc.apply(F, t, properties, prop_calc)

        sigma = self._sigma
        self._minmod(F, sigma)

        xi = properties.xi[None, :, None, None]
        dx = properties.mesh.get_dx()[:, None, None, None]

        F_l = F[:-1] + 0.5 * sigma[:-1]
        F_r = F[1:] - 0.5 * sigma[1:]

        Ws = xp.where(xi >= 0, F_l, F_r)

        flux_diff = Ws[:-1] - Ws[1:]

        start = ng
        end = F.shape[0] - ng

        F[start:end] += (tau / dx[start:end]) * xi * flux_diff[start - 1:end - 1]

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):
        self._step(F, t, tau, properties, prop_calc)
        super()._calculate_collisions(F, tau, properties, prop_calc)
