from scipy.special import erf

from config.configuration import *
from src.thermodynamics import *
import matplotlib.pyplot as plt
from src.solvers import SolverGodunov, SolverRK

from src.datio import write_to_csv

CFL = 0.99
t_max = 0.2 * 1.415

n_x = 100
n_xi = 60

model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W}
#write_to_csv([0], [0], [0], [0], [0], f'infographics/Riemann/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.dat')

model = BKG(model_config)
solver = SolverGodunov()
model.calculate(CFL, t_max, solver.step, right_part=True)


fig, axs = plt.subplots(1, 3)
fig.suptitle(f'n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max}, CFL:{CFL}_Kn:{TD_KN}')

x = model.x[:-1]+model.h/2
n, u, T, q = model.get_macros()

axs[0].set_title('n (density)')
axs[0].scatter(x, n, linewidth=0.01)
axs[0].plot(x, n, color='blue')
axs[0].grid()

axs[1].set_title('u (velocity)')
axs[1].scatter(x, u, linewidth=0.01)
axs[1].plot(x, u, color='blue')
axs[1].grid()

axs[2].set_title('T (temperature)')
axs[2].scatter(x, T, linewidth=0.01)
axs[2].plot(x, T, color='blue')
axs[2].grid()
plt.savefig(f'infographics/Riemann/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.png', dpi=300)
write_to_csv(x, n, u, T, q, f'calculated_data/Riemann/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.dat')
plt.show()
