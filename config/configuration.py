import numpy as np

TD_KN=0.1
TD_PR=0.72
TD_W=0.81

X_LEFT, X_RIGHT = 0, 1
XI_LEFT, XI_RIGHT = -10, 10

F_BEG_N = lambda x: 0.75 - 0.25 * np.sign(x-0.5)
F_BEG_U = lambda x: 0
F_BEG_T = lambda x: 1


CFL = 0.25
TIME_MAX = 0.1