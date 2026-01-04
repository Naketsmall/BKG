from src.config.configuration import *
from src.solvers.godunov import SolverGodunov
from src.solvers.rk2 import SolverRK

from src.solvers.tolstyh import SolverL3
import matplotlib.pyplot as plt

#from src.datio import write_to_csv

from src.config.libloader import xp, cuda_is_available
from src.thermodynamics import ModelProperties, ModelState, ShakhovSolver, PropertyCalculator

CFL = 0.99
t_max = 0.2 * 1.415
TD_KN = 9e-5

n_x = 100
n_xi = 40

model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W}


#model = BKG(model_config, xp)
#solver = SolverGodunov()
#solver = SolverRK()
#solver = SolverL3(n_x+1, model.h)
#model.calculate(CFL, t_max, solver._step, right_part=True)

properties = ModelProperties(model_config)
state = (ModelState(properties, model_config))
solver = ShakhovSolver(state, properties, SolverGodunov())

solver.calculate(CFL, t_max)



x = properties.x[:-1]+properties.h/2
if cuda_is_available:
    x = xp.asnumpy(x)
n, u, T, q = PropertyCalculator.get_solution_macros(state.F, properties)


fig, axs = plt.subplots(1, 3)
fig.suptitle(f'n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max}, CFL:{CFL}_Kn:{TD_KN}')

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

path = 'Tolstyh2'
#plt.savefig(f'infographics/{path}/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.png', dpi=300)
#write_to_csv(x, n, u, T, q, f'calculated_data/{path}/n_x:{n_x}_xi:({XI_LEFT},{XI_RIGHT},{n_xi})_t:{t_max}_CFL:{CFL}_Kn:{TD_KN}.dat')
plt.show()
