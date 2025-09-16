import numpy as np

#TD_KN=0.1
TD_KN = 0.09
TD_PR=0.72
#TD_W=0.81
TD_W = 1

X_LEFT, X_RIGHT = -1, 3
XI_LEFT, XI_RIGHT = -10, 10

#F_BEG_N = lambda x: 0.75 - 0.25 * np.sign(x)
F_BEG_N = np.vectorize(lambda x: 1. if x <= 0.5 else 0.125)
#F_BEG_N = lambda x: 1
F_BEG_U = lambda x: 0
F_BEG_T = np.vectorize(lambda x: 1. if x <= 0.5 else 0.8)