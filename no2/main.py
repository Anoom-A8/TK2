import time
import matplotlib.pyplot as plt
from simpsons_method import simpson_integration
from adaptive_method import adaptive_integration
from romberg_method import romberg_integral
from log_loss import log_loss_y1  # Import the log-loss function for y=1

# Integration range and parameters
start_range, end_range = 0, 10
tolerance_level = 1e-4  # Tolerance for adaptive quadrature
simpson_subdivisions = [10, 50, 100, 500, 1000]  # Subdivisions for Simpson's Rule
romberg_levels = [4, 6, 7, 9, 10]  # Levels for Romberg Integration

def execute_simpson():
    print("\nSimpson's Rule Results:")
    print(f"{'Subdivisions (N)':<20}{'Result':<15}{'Time (s)':<10}")
    print("-" * 45)
    results = []
    timings = []
    for subdivisions in simpson_subdivisions:
        start_time = time.time()
        result = simpson_integration(log_loss_y1, start_range, end_range, subdivisions)
        end_time = time.time()
        results.append(result)
        timings.append(end_time - start_time)
        print(f"{subdivisions:<20}{result:<15.6f}{(end_time - start_time):.8f}")
    return results, timings

def execute_adaptive():
    print("\nAdaptive Quadrature Results:")
    print(f"{'Tolerance':<15}{'Result':<15}{'Time (s)':<10}")
    print("-" * 40)
    start_time = time.time()
    result = adaptive_integration(log_loss_y1, start_range, end_range, tolerance=tolerance_level)
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"{tolerance_level:<15}{result:<15.6f}{computation_time:.8f}")
    return result, computation_time

def execute_romberg():
    print("\nRomberg Integration Results:")
    print(f"{'Levels (m)':<12}{'Subdivisions (N)':<20}{'Result':<15}{'Time (s)':<10}")
    print("-" * 55)
    results = []
    timings = []
    for levels in romberg_levels:
        start_time = time.time()
        result = romberg_integral(log_loss_y1, start_range, end_range, max_depth=levels)
        end_time = time.time()

        # Number of subdivisions is 2^(m-1)
        subdivisions = 2**(levels - 1)
        computation_time = end_time - start_time

        results.append(result)
        timings.append(computation_time)
        print(f"{levels:<12}{subdivisions:<20}{result:<15.6f}{computation_time:.8f}")
    return results, timings

def visualize_results(simpson_results, simpson_times, adaptive_result, adaptive_time, romberg_results, romberg_times):
    plt.figure(figsize=(15, 8))

    # Accuracy Plot
    plt.subplot(2, 1, 1)
    plt.plot(simpson_subdivisions, simpson_results, label="Simpson's Rule", marker='o')
    plt.axhline(adaptive_result, color='green', linestyle='--', label="Adaptive Quadrature")
    plt.plot(romberg_levels, romberg_results, label="Romberg Integration", marker='s')
    plt.title("Accuracy of Integration Methods")
    plt.xlabel("Subdivisions (N) / Levels (m)")
    plt.ylabel("Integral Value")
    plt.legend()
    plt.grid()

    # Computation Time Plot
    plt.subplot(2, 1, 2)
    plt.plot(simpson_subdivisions, simpson_times, label="Simpson's Timing", marker='o')
    plt.bar(["Adaptive Quadrature"], [adaptive_time], color='green', label="Adaptive Timing")
    plt.plot(romberg_levels, romberg_times, label="Romberg Timing", marker='s')
    plt.title("Computation Time of Methods")
    plt.xlabel("Subdivisions (N) / Levels (m)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Execute Simpson's Rule
    simpson_results, simpson_times = execute_simpson()

    # Execute Adaptive Quadrature
    adaptive_result, adaptive_time = execute_adaptive()

    # Execute Romberg Integration
    romberg_results, romberg_times = execute_romberg()

    # Visualize the results
    visualize_results(simpson_results, simpson_times, adaptive_result, adaptive_time, romberg_results, romberg_times)
