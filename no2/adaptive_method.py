def adaptive_integration(function, start, end, tolerance=1e-4):
    def trapezoid_estimate(function, start, end):
        return (end - start) * (function(start) + function(end)) / 2

    def adaptive_refine(function, start, end, tolerance, whole):
        midpoint = (start + end) / 2
        left_area = trapezoid_estimate(function, start, midpoint)
        right_area = trapezoid_estimate(function, midpoint, end)
        total_area = abs(left_area + right_area - whole)

        if total_area <= 15 * tolerance:
            return left_area + right_area + total_area / 15
        else:
            return (
                adaptive_refine(function, start, midpoint, tolerance / 2, left_area) +
                adaptive_refine(function, midpoint, end, tolerance / 2, right_area)
            )

    initial_estimate = trapezoid_estimate(function, start, end)
    return adaptive_refine(function, start, end, tolerance, initial_estimate)
