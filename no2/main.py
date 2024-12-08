import time
import matplotlib.pyplot as plt
from log_loss import log_loss_y1
from simpsons_method import composite_simpson
from adaptive_method import adaptive_quadrature
from romberg_method import romberg_integration

# Define the integration range
a, b = 0, 10
tolerance = 1e-4  # For adaptive quadrature
N_values = [10, 50, 100, 500, 1000]  # Subdivisions for Composite Simpson
m_values = [4, 6, 7, 9, 10]  # Levels for Romberg Integration

def run_composite_simpson():
    print("\nComposite Simpson's Rule Results:")
    print(f"{'N (Subdivisions)':<20}{'Result':<15}{'Time (s)':<10}")
    print("-" * 45)
    results = []
    times = []
    for N in N_values:
        start = time.perf_counter()
        result = composite_simpson(log_loss_y1, a, b, N)
        end = time.perf_counter()
        results.append(result)
        times.append(end - start)
        print(f"{N:<20}{result:<15.6f}{(end - start):.8f}")
    return results, times

def run_adaptive_quadrature():
    print("\nAdaptive Quadrature Results:")
    print(f"{'Tolerance':<15}{'Result':<15}{'Time (s)':<10}")
    print("-" * 40)
    start = time.perf_counter()
    result = adaptive_quadrature(log_loss_y1, a, b, tol=tolerance)
    end = time.perf_counter()
    computation_time = end - start
    print(f"{tolerance:<15}{result:<15.6f}{computation_time:.8f}")
    return result, computation_time

def run_romberg_experiments():
    print("\nRomberg Integration Results:")
    print(f"{'m (Levels)':<12}{'N (Subdivisions)':<20}{'Result':<15}{'Time (s)':<10}")
    print("-" * 55)
    results = []
    times = []
    for m in m_values:
        start = time.perf_counter()
        result = romberg_integration(log_loss_y1, a, b, max_levels=m)
        end = time.perf_counter()
        
        # Number of subdivisions is 2^(m-1)
        N = 2**(m - 1)
        computation_time = end - start

        results.append(result)
        times.append(computation_time)
        print(f"{m:<12}{N:<20}{result:<15.6f}{computation_time:.8f}")
    return results, times

def visualize_results(simpson_results, simpson_times, adaptive_result, adaptive_time, romberg_results, romberg_times):
    # Visualization
    plt.figure(figsize=(15, 8))

    # Accuracy Plot
    plt.subplot(2, 1, 1)
    plt.plot(N_values, simpson_results, marker='o', label="Composite Simpson")
    plt.axhline(adaptive_result, color='g', linestyle='--', label="Adaptive Quadrature")
    plt.plot(m_values, romberg_results, marker='s', label="Romberg Integration")
    plt.xlabel("Subdivisions (N) / Levels (m)")
    plt.ylabel("Integral Value")
    plt.title("Integral Results Comparison")
    plt.legend()
    plt.grid()

    # Computation Time Plot
    plt.subplot(2, 1, 2)
    plt.plot(N_values, simpson_times, marker='o', label="Composite Simpson")
    plt.bar(["Adaptive Quadrature"], [adaptive_time], color='g', label="Adaptive Quadrature")
    plt.plot(m_values, romberg_times, marker='s', label="Romberg Integration")
    plt.xlabel("Subdivisions (N) / Levels (m)")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Computation Time Comparison")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run Composite Simpson
    simpson_results, simpson_times = run_composite_simpson()

    # Run Adaptive Quadrature
    adaptive_result, adaptive_time = run_adaptive_quadrature()

    # Run Romberg Integration
    romberg_results, romberg_times = run_romberg_experiments()

    # Visualize All Results
    visualize_results(simpson_results, simpson_times, adaptive_result, adaptive_time, romberg_results, romberg_times)