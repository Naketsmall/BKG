cuda_is_available = 0
try:
    import cupy as cp

    cp.cuda.Device(0).compute_capability
    xp = cp
    cuda_is_available = 1
    print("Используем GPU через CuPy")
except Exception:
    import numpy as np
    xp = np
    print("libloader.py: GPU недоступно, используем CPU через NumPy")