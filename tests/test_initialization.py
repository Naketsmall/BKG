from src.thermodynamics import *

import unittest

class TestInitialConditions(unittest.TestCase):
    def test_create_0_10_100_n(self):
        model = BKG(100, 100)
        x = model.x
        n, u, T, q = model.get_macros()
        x_n = F_BEG_N( (x[:-1] + x[1:])/2 )

        assert np.allclose(n, x_n, atol=1e-10)

    def test_create_0_10_100_u(self):
        model = BKG(100, 100)
        x = model.x
        n, u, T, q = model.get_macros()
        x_u = F_BEG_U((x[:-1] + x[1:]) / 2)

        assert np.allclose(u, x_u, atol=1e-10)

    def test_create_0_10_100_T(self):
        model = BKG(100, 100)
        x = model.x
        n, u, T, q = model.get_macros()
        x_T = F_BEG_T((x[:-1] + x[1:]) / 2)

        assert np.allclose(T, x_T, atol=1e-10)


if __name__ == "__main__":
    unittest.main()



"""
model = BKG(80, 100)


fig, axs = plt.subplots(1, 3)
fig.suptitle('Initial conditions after creating class')
n, u, T, q = model.get_macros()
print('x', model.x+model.h/2)
print('n:', n)
print('u:', u)
print('T:', T)

#print(np.sum(model.F * model.xi_cell_size**3 * model.h, axis=(0,1,2,3)))

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
"""