import numpy as np

def romberg_integral(function, lower_bound, upper_bound, max_depth=10):
    def trapezoidal_approximation(function, lower, upper, divisions):
        step = (upper - lower) / divisions
        x_points = np.linspace(lower, upper, divisions + 1)
        y_points = function(x_points)
        return step * (y_points[0] / 2 + sum(y_points[1:-1]) + y_points[-1] / 2)

    romberg_table = np.zeros((max_depth, max_depth))
    romberg_table[0, 0] = trapezoidal_approximation(function, lower_bound, upper_bound, 1)

    for depth in range(1, max_depth):
        interval_count = 2**depth
        romberg_table[depth, 0] = trapezoidal_approximation(function, lower_bound, upper_bound, interval_count)

        for level in range(1, depth + 1):
            romberg_table[depth, level] = (
                (4**level * romberg_table[depth, level - 1] - romberg_table[depth - 1, level - 1]) /
                (4**level - 1)
            )

        if depth > 1 and abs(romberg_table[depth, depth] - romberg_table[depth - 1, depth - 1]) < 1e-6:
            return romberg_table[depth, depth]

    return romberg_table[max_depth - 1, max_depth - 1]
