import numpy as np

def romberg_integration(func, a, b, max_levels=10):
    """
    Custom Romberg integration implementation.
    Combines trapezoidal approximations with Richardson extrapolation.
    """

    def trapezoidal(func, a, b, n):
        """Compute the trapezoidal rule with n intervals."""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        return h * (y[0] / 2 + sum(y[1:-1]) + y[-1] / 2)

    # Initialize Romberg table
    R = np.zeros((max_levels, max_levels))

    # First level: Trapezoidal rule with one interval
    R[0, 0] = trapezoidal(func, a, b, 1)

    # Fill the Romberg table
    for k in range(1, max_levels):
        # Trapezoidal rule with 2^k intervals
        n_intervals = 2**k
        R[k, 0] = trapezoidal(func, a, b, n_intervals)

        # Richardson extrapolation
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)

        # Check for convergence
        if k > 1 and abs(R[k, k] - R[k - 1, k - 1]) < 1e-6:
            return R[k, k]

    # If no convergence, return the highest accuracy result
    return R[max_levels - 1, max_levels - 1]