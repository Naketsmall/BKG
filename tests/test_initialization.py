from src.thermodynamics import *
import matplotlib.pyplot as plt

model = BKG(10, 100)


fig, axs = plt.subplots(1, 3)
fig.suptitle('Initial conditions after creating class')
n, u, T, q = model.get_macros()
print('n:', n)
print('u:', u)
print('T:', T)

print(np.sum(model.F * model.xi_cell_size**3 * model.h, axis=(0,1,2,3)))

axs[0].set_title('n (density)')
axs[0].plot(n)
axs[0].grid()
axs[1].set_title('u (velocity)')
axs[1].plot(u)
axs[1].grid()
axs[2].set_title('T (temperature)')
axs[2].plot(T)
axs[2].grid()
plt.show()