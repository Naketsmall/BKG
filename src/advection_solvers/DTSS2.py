from src.advection_solvers.base import *
from src.config.libloader import xp


class SolverDTSS2(Solver):
    __name__ = "BDF2 + DTSS"

    def __init__(
        self,
        explicit_solver,
        n_iter=15,
        omega=0.4,
        cfl_pseudo=0.5
    ):
        self.explicit_solver = explicit_solver

        self.n_iter = n_iter
        self.omega = omega
        self.cfl_pseudo = cfl_pseudo

        self._buffers_allocated = False
        self._has_prev = False

    def _alloc_buffers(self, F, properties):

        self._F_prev = xp.zeros_like(F)
        self._F_curr = xp.zeros_like(F)
        self._F_iter = xp.zeros_like(F)
        self._F_tmp = xp.zeros_like(F)

        self._RHS = xp.zeros_like(F)
        self._AF = xp.zeros_like(F)
        self._residual = xp.zeros_like(F)

        xi_max = xp.max(xp.abs(properties.xi))
        dx_min = xp.min(properties.mesh.get_dx())

        self._tau_pseudo = self.cfl_pseudo * dx_min / xi_max

        self._buffers_allocated = True

    def _transport_operator(self, F, t, tau_pseudo, properties, prop_calc):

        F_tmp = self._F_tmp
        AF = self._AF

        F_tmp[:] = F

        self.explicit_solver._step(
            F_tmp, t, tau_pseudo, properties, prop_calc
        )

        # AF = -(F_tmp - F) / tau_pseudo
        xp.subtract(F_tmp, F, out=AF)
        AF *= -1.0 / tau_pseudo

    def _step(self, F, t, tau, properties, prop_calc):

        if not self._buffers_allocated:
            self._alloc_buffers(F, properties)

        F_curr = self._F_curr
        F_iter = self._F_iter
        RHS = self._RHS
        AF = self._AF
        residual = self._residual

        F_curr[:] = F
        F_iter[:] = F

        tau_pseudo = self._tau_pseudo

        if not self._has_prev:

            RHS[:] = F_curr
            alpha = tau

        else:

            F_prev = self._F_prev

            RHS[:] = F_curr
            RHS *= 4.0 / 3.0
            RHS -= (1.0 / 3.0) * F_prev

            alpha = (2.0 / 3.0) * tau

        for _ in range(self.n_iter):

            self._transport_operator(
                F_iter, t, tau_pseudo, properties, prop_calc
            )

            # residual = RHS - (F_iter + alpha * AF)

            residual[:] = F_iter
            residual += alpha * AF
            residual *= -1
            residual += RHS

            # F_iter += omega * residual
            F_iter += self.omega * residual

        F[:] = F_iter

        self._F_prev[:] = F_curr
        self._has_prev = True

    def calculate_layer(self, F, t, tau, properties: ModelProperties, prop_calc):

        self._step(F, t, tau, properties, prop_calc)

        super()._calculate_collisions(
            F, tau, properties, prop_calc
        )