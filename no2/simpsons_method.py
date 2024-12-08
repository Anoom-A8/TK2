import numpy as np

def simpson_integration(function, lower_limit, upper_limit, sub_intervals):
    if sub_intervals % 2 != 0:
        raise ValueError("Sub-interval count must be even for Simpson's rule.")
    step_size = (upper_limit - lower_limit) / sub_intervals
    x_values = np.linspace(lower_limit, upper_limit, sub_intervals + 1)
    y_values = function(x_values)
    integral = y_values[0] + y_values[-1] + 4 * sum(y_values[1::2]) + 2 * sum(y_values[2:-1:2])
    return (step_size / 3) * integral
