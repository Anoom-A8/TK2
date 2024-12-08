# simpson_method.py
import numpy as np

def composite_simpson(func, a, b, N):
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    S = y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-2:2])
    return (h / 3) * S