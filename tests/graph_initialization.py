from scipy.special import erf

from src.config.configuration import *
from src.thermodynamics import *
import matplotlib.pyplot as plt
from src.solvers import SolverGodunov, SolverRK


def n_exact(x, t):
    return 1/4*(3-erf(x/t))

def u_exact(x, t):
    return 1/(4*np.pi**0.5)*np.exp(-(x/t)**2)/n_exact(x, t)

CFL = 0.9
t_max = 1

n_x = 100
n_xi = 60

model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W}

model = BKG(model_config)
solver = SolverGodunov()
model.calculate(CFL, t_max, solver._step)

fig, axs = plt.subplots(1, 3)
fig.suptitle(f'n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max}, CFL:{CFL}')
n, u, T, q = model.get_macros()

axs[0].set_title('n (density)')
axs[0].scatter(model.x[:-1]+model.h/2, n, linewidth=0.01)
axs[0].plot(model.x[:-1]+model.h/2, n_exact(model.x[:-1]+model.h/2, t_max), color='black')
axs[0].grid()

axs[1].set_title('u (velocity)')
axs[1].scatter(model.x[:-1]+model.h/2, u, linewidth=0.01)
axs[1].plot(model.x[:-1]+model.h/2, u_exact(model.x[:-1]+model.h/2, t_max), color='black')
axs[1].grid()

axs[2].set_title('T (temperature)')
axs[2].scatter(model.x[:-1]+model.h/2, T, linewidth=0.01)
axs[2].grid()
plt.savefig(f'infographics/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}.png', dpi=300)
plt.show()