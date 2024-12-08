def adaptive_quadrature(func, a, b, tol=1e-4):
    """
    Custom adaptive quadrature implementation.
    Recursively divides the interval until the error is below the tolerance.
    """

    def integrate(func, a, b):
        """Estimate integral using the trapezoidal rule."""
        return (b - a) * (func(a) + func(b)) / 2

    def recursive_quad(func, a, b, tol, whole):
        """Recursively apply the quadrature method."""
        mid = (a + b) / 2
        left = integrate(func, a, mid)
        right = integrate(func, mid, b)
        error = abs(left + right - whole)

        if error <= 15 * tol:  # Scale error based on Simpson's rule estimate
            return left + right + error / 15
        else:
            # Recursively refine the intervals
            left_result = recursive_quad(func, a, mid, tol / 2, left)
            right_result = recursive_quad(func, mid, b, tol / 2, right)
            return left_result + right_result

    # Compute the whole integral initially
    whole = integrate(func, a, b)
    return recursive_quad(func, a, b, tol, whole)