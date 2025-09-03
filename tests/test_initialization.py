from src.thermodynamics import *

import unittest

"""
MARK: NOT UPDATED FOR MODEL INITIALIZATION WITH CONFIG. SO NOW IT'S NOT PROPERLY WORKING
"""


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
