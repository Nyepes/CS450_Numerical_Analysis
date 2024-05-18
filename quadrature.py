import numpy as np

def f(x):
    return 4 / (1.0 + x * x)

x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
print(np.trapz(f(x), x))