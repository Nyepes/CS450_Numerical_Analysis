import numpy as np

def newton_method(f, f_prime, x0, error = 1e-10):
    while (np.abs(f(x0)) >= error):
        x0 = x0 - (f(x0) / f_prime(x0))
    return x0
def secant_method(f, x0, x1, error = 1e-10):
    prev = f(x0)
    cur = f(x1)
    while (np.abs(cur) >= error):
        temp = x1
        x1 = x1 - (cur * (x1 - x0)) / (cur - prev)
        prev = cur
        cur = f(x1)
        x0 = temp
    return x1


# Examples:
# def f(x):
#     return x ** 3 - 10 * x ** 2
# def gradf(x):
#     return 3 * x ** 2 - 20 * x

# newton_method(f, gradf, 25)
# secant_method(f, 25, 15)