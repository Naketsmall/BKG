from src.boundary_condition import ZeroGradBoundaryCondition, EvapCondBoundaryCondition
from src.config.configuration import *
from src.mesh import UnadaptableMesh
from src.solvers.WENO5RK3 import WENO5RK3
from src.solvers.godunov import SolverGodunov
from src.solvers.kolgan import SolverKolgan
from src.solvers.rk2 import SolverRK

from src.solvers.tolstyh import SolverL3
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

#from src.datio import write_to_csv

from src.config.libloader import xp, cuda_is_available
from src.thermodynamics import ModelProperties, ModelState, ShakhovSolver, PropertyCalculator

CFL = 0.8
t_max = 0.2
TD_KN = 0.1

n_x = 20
n_xi = 30


model_config = {'X_LEFT': X_LEFT, 'X_RIGHT': X_RIGHT, 'n_x': n_x,
                'XI_LEFT': XI_LEFT, 'XI_RIGHT': XI_RIGHT, 'n_xi': n_xi,
                'F_BEG_N': F_BEG_N, 'F_BEG_U': F_BEG_U, 'F_BEG_T': F_BEG_T,
                'Kn': TD_KN, 'Pr': TD_PR, 'w': TD_W}


def graded_linspace(xp, n_points, a=0.01, length=1.0):
    n_cells = n_points - 1

    # находим d из суммы арифметической прогрессии
    d = (2 * length / n_cells - 2 * a) / (n_cells - 1)

    # размеры ячеек
    k = xp.arange(n_cells)
    dx = a + k * d

    # сами точки
    x = xp.zeros(n_points)
    x[1:] = xp.cumsum(dx)

    return x


mesh1 = UnadaptableMesh(graded_linspace(xp, n_points=n_x, a=0.01, length=X_RIGHT))
mesh2 = UnadaptableMesh(xp.linspace(X_LEFT, X_RIGHT, n_x, endpoint=True))


bc = EvapCondBoundaryCondition(2, lambda t: 5., lambda t: 5.)


adv_solver = SolverRK()
properties = ModelProperties(model_config, mesh1, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)

solver.calculate(CFL, t_max)



#x = properties.mesh.x[bc.n_ghost:len(properties.mesh.x)-bc.n_ghost+1]+properties.mesh.h/2
x = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]


if cuda_is_available:
    x = xp.asnumpy(x)
n, u, T, q = PropertyCalculator.get_solution_macros(state.F, properties)

fig, axs = plt.subplots(1, 3)
fig.suptitle(f'{adv_solver.get_name()}, n_x:{n_x}, x:({X_LEFT},{X_RIGHT},{n_x}), xi:({XI_LEFT},{XI_RIGHT},{n_xi}), t:{t_max.__round__(3)}, CFL:{CFL}, Kn:{TD_KN}')

print(x.shape, n.shape)

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

properties = ModelProperties(model_config, mesh2, bc)
state = ModelState(properties, model_config)
solver = ShakhovSolver(state, properties, adv_solver)

solver.calculate(CFL, t_max)



#x = properties.mesh.x[bc.n_ghost:len(properties.mesh.x)-bc.n_ghost+1]+properties.mesh.h/2
x2 = properties.mesh.get_centers()[bc.n_ghost:len(properties.mesh.x) - bc.n_ghost + 1]
if cuda_is_available:
    x = xp.asnumpy(x)
n2, u2, T2, q2 = PropertyCalculator.get_solution_macros(state.F, properties)




axs[0].scatter(x2, n2, linewidth=0.01)
axs[0].plot(x2, n2, color='red')
#axs[0].grid()

axs[1].scatter(x2, u2, linewidth=0.01)
axs[1].plot(x2, u2, color='red')
#axs[1].grid()

axs[2].scatter(x2, T2, linewidth=0.01)
axs[2].plot(x2, T2, color='red')
#axs[2].grid()

plt.show()
