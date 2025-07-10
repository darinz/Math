# Python code extracted from 10-numerical-methods.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def rectangle_rule():
    """
    Implement rectangle rule for numerical integration.
    
    Mathematical foundation:
    - Approximates ∫f(x)dx ≈ h∑f(x_i) where h = (b-a)/n
    - Uses left endpoints of subintervals
    - Error bound: |E| ≤ (b-a)²M/(2n) where M = max|f''(x)|
    - Order of accuracy: O(h)
    """
    
    # Example: ∫x² dx from 0 to 2
    def f(x):
        return x**2
    
    a, b = 0, 2
    exact_value = 8/3  # Exact value: x³/3 from 0 to 2
    
    # Rectangle rule with different numbers of subintervals
    n_values = [5, 10, 20, 50, 100]
    approximations = []
    
    print("Rectangle Rule for ∫x² dx from 0 to 2:")
    print(f"Exact value: {exact_value:.6f}")
    print()
    
    for n in n_values:
        h = (b - a) / n  # Step size
        x_points = np.linspace(a, b, n+1)
        y_points = f(x_points[:-1])  # Left endpoints
        
        # Rectangle rule: h * sum of function values at left endpoints
        approximation = h * np.sum(y_points)
        approximations.append(approximation)
        
        error = abs(approximation - exact_value)
        print(f"n = {n:3d}: Approximation = {approximation:.6f}, Error = {error:.6f}")
    
    return n_values, approximations, exact_value

n_values, approximations, exact_value = rectangle_rule()

# Visualize rectangle rule
def visualize_rectangle_rule():
    """
    Visualize the rectangle rule approximation.
    Shows how rectangles approximate the area under the curve.
    """
    def f(x):
        return x**2
    
    a, b = 0, 2
    n = 10
    h = (b - a) / n
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points[:-1])
    
    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Plot function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x²')
    
    # Plot rectangles
    for i in range(n):
        x_left = x_points[i]
        x_right = x_points[i+1]
        y_height = y_points[i]
        
        # Draw rectangle
        plt.bar(x_left, y_height, width=h, alpha=0.3, color='red', align='edge')
        
        # Add rectangle borders
        plt.plot([x_left, x_left, x_right, x_right, x_left], 
                [0, y_height, y_height, 0, 0], 'r-', linewidth=1)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Rectangle Rule for Numerical Integration\n(Red rectangles approximate area under curve)')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_rectangle_rule()

# Code Block 2
def trapezoidal_rule():
    """
    Implement trapezoidal rule for numerical integration.
    
    Mathematical foundation:
    - Approximates ∫f(x)dx ≈ (h/2)[f(x₀) + 2∑f(x_i) + f(xₙ)]
    - Uses linear interpolation between points
    - Error bound: |E| ≤ (b-a)³M/(12n²) where M = max|f''(x)|
    - Order of accuracy: O(h²) - better than rectangle rule
    """
    
    def f(x):
        return x**2
    
    a, b = 0, 2
    exact_value = 8/3
    
    # Trapezoidal rule with different numbers of subintervals
    n_values = [5, 10, 20, 50, 100]
    approximations = []
    
    print("Trapezoidal Rule for ∫x² dx from 0 to 2:")
    print(f"Exact value: {exact_value:.6f}")
    print()
    
    for n in n_values:
        h = (b - a) / n
        x_points = np.linspace(a, b, n+1)
        y_points = f(x_points)
        
        # Trapezoidal rule: h/2 * (f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ))
        approximation = h/2 * (y_points[0] + 2*np.sum(y_points[1:-1]) + y_points[-1])
        approximations.append(approximation)
        
        error = abs(approximation - exact_value)
        print(f"n = {n:3d}: Approximation = {approximation:.6f}, Error = {error:.6f}")
    
    return n_values, approximations, exact_value

n_values, approximations, exact_value = trapezoidal_rule()

# Visualize trapezoidal rule
def visualize_trapezoidal_rule():
    """
    Visualize the trapezoidal rule approximation.
    Shows how trapezoids approximate the area under the curve.
    """
    def f(x):
        return x**2
    
    a, b = 0, 2
    n = 10
    h = (b - a) / n
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points)
    
    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Plot function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x²')
    
    # Plot trapezoids
    for i in range(n):
        x_left = x_points[i]
        x_right = x_points[i+1]
        y_left = y_points[i]
        y_right = y_points[i+1]
        
        # Draw trapezoid
        plt.fill([x_left, x_right, x_right, x_left], [0, 0, y_right, y_left], 
                alpha=0.3, color='red')
        
        # Add trapezoid borders
        plt.plot([x_left, x_right], [y_left, y_right], 'r-', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Trapezoidal Rule for Numerical Integration\n(Red trapezoids approximate area under curve)')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_trapezoidal_rule()

# Code Block 3
def simpsons_rule():
    """
    Implement Simpson's rule for numerical integration.
    
    Mathematical foundation:
    - Approximates ∫f(x)dx ≈ (h/3)[f(x₀) + 4∑f(x_odd) + 2∑f(x_even) + f(xₙ)]
    - Uses quadratic interpolation between points
    - Error bound: |E| ≤ (b-a)⁵M/(180n⁴) where M = max|f⁽⁴⁾(x)|
    - Order of accuracy: O(h⁴) - much better than trapezoidal rule
    - Requires even number of subintervals
    """
    
    def f(x):
        return x**2
    
    a, b = 0, 2
    exact_value = 8/3
    
    # Simpson's rule with different numbers of subintervals (must be even)
    n_values = [6, 10, 20, 50, 100]
    approximations = []
    
    print("Simpson's Rule for ∫x² dx from 0 to 2:")
    print(f"Exact value: {exact_value:.6f}")
    print()
    
    for n in n_values:
        h = (b - a) / n
        x_points = np.linspace(a, b, n+1)
        y_points = f(x_points)
        
        # Simpson's rule: h/3 * (f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ))
        approximation = h/3 * (y_points[0] + 4*np.sum(y_points[1:-1:2]) + 
                              2*np.sum(y_points[2:-1:2]) + y_points[-1])
        approximations.append(approximation)
        
        error = abs(approximation - exact_value)
        print(f"n = {n:3d}: Approximation = {approximation:.6f}, Error = {error:.6f}")
    
    return n_values, approximations, exact_value

n_values, approximations, exact_value = simpsons_rule()

# Compare different integration methods
def compare_integration_methods():
    """
    Compare the accuracy of different numerical integration methods.
    Shows how error decreases with increasing n for each method.
    """
    def f(x):
        return x**2
    
    a, b = 0, 2
    exact_value = 8/3
    n = 20
    
    # Rectangle rule
    h = (b - a) / n
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points[:-1])
    rectangle_approx = h * np.sum(y_points)
    
    # Trapezoidal rule
    y_points_full = f(x_points)
    trapezoidal_approx = h/2 * (y_points_full[0] + 2*np.sum(y_points_full[1:-1]) + y_points_full[-1])
    
    # Simpson's rule
    simpson_approx = h/3 * (y_points_full[0] + 4*np.sum(y_points_full[1:-1:2]) + 
                           2*np.sum(y_points_full[2:-1:2]) + y_points_full[-1])
    
    print("Comparison of Integration Methods (n = 20):")
    print(f"Exact value: {exact_value:.6f}")
    print(f"Rectangle rule: {rectangle_approx:.6f}, Error: {abs(rectangle_approx - exact_value):.6f}")
    print(f"Trapezoidal rule: {trapezoidal_approx:.6f}, Error: {abs(trapezoidal_approx - exact_value):.6f}")
    print(f"Simpson's rule: {simpson_approx:.6f}, Error: {abs(simpson_approx - exact_value):.6f}")
    
    return rectangle_approx, trapezoidal_approx, simpson_approx

rectangle_approx, trapezoidal_approx, simpson_approx = compare_integration_methods()

# Code Block 4
def finite_differences():
    """
    Implement finite difference methods for numerical differentiation.
    
    Mathematical foundation:
    - Forward difference: f'(x) ≈ (f(x+h) - f(x))/h
    - Backward difference: f'(x) ≈ (f(x) - f(x-h))/h  
    - Central difference: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
    - Error analysis: Forward/Backward O(h), Central O(h²)
    """
    
    def f(x):
        return x**3
    
    x0 = 1.0
    exact_derivative = 3 * x0**2  # f'(x) = 3x²
    
    # Different step sizes
    h_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    print("Numerical Differentiation Methods:")
    print(f"Function: f(x) = x³")
    print(f"Point: x = {x0}")
    print(f"Exact derivative: f'({x0}) = {exact_derivative}")
    print()
    
    # Forward difference: f'(x) ≈ (f(x+h) - f(x))/h
    print("Forward Difference Method:")
    for h in h_values:
        forward_diff = (f(x0 + h) - f(x0)) / h
        error = abs(forward_diff - exact_derivative)
        print(f"h = {h:.3f}: Approximation = {forward_diff:.6f}, Error = {error:.6f}")
    
    print()
    
    # Backward difference: f'(x) ≈ (f(x) - f(x-h))/h
    print("Backward Difference Method:")
    for h in h_values:
        backward_diff = (f(x0) - f(x0 - h)) / h
        error = abs(backward_diff - exact_derivative)
        print(f"h = {h:.3f}: Approximation = {backward_diff:.6f}, Error = {error:.6f}")
    
    print()
    
    # Central difference: f'(x) ≈ (f(x+h) - f(x-h))/(2h)
    print("Central Difference Method:")
    for h in h_values:
        central_diff = (f(x0 + h) - f(x0 - h)) / (2*h)
        error = abs(central_diff - exact_derivative)
        print(f"h = {h:.3f}: Approximation = {central_diff:.6f}, Error = {error:.6f}")
    
    return h_values, exact_derivative

h_values, exact_derivative = finite_differences()

# Code Block 5
def bisection_method():
    """Implement bisection method for finding roots"""
    
    def f(x):
        return x**2 - 4  # Roots at x = ±2
    
    a, b = 0, 3  # Initial interval [0, 3]
    tolerance = 1e-6
    max_iterations = 100
    
    print("Bisection Method:")
    print(f"Function: f(x) = x² - 4")
    print(f"Initial interval: [{a}, {b}]")
    print(f"Tolerance: {tolerance}")
    print()
    
    iterations = []
    errors = []
    
    for i in range(max_iterations):
        c = (a + b) / 2  # Midpoint
        f_c = f(c)
        
        iterations.append(i + 1)
        errors.append(abs(f_c))
        
        print(f"Iteration {i+1:2d}: c = {c:.6f}, f(c) = {f_c:.6f}")
        
        if abs(f_c) < tolerance:
            print(f"\nRoot found: x = {c:.6f}")
            print(f"Function value: f({c:.6f}) = {f_c:.6f}")
            print(f"Number of iterations: {i+1}")
            break
        
        if f(a) * f_c < 0:
            b = c  # Root is in left half
        else:
            a = c  # Root is in right half
    else:
        print("Maximum iterations reached")
    
    return iterations, errors, c

iterations, errors, root = bisection_method()

# Visualize bisection method
def visualize_bisection_method():
    def f(x):
        return x**2 - 4
    
    a, b = 0, 3
    x_plot = np.linspace(-1, 4, 1000)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Plot function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² - 4')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot initial interval
    plt.plot([a, a], [0, f(a)], 'r--', alpha=0.7, label='Initial interval')
    plt.plot([b, b], [0, f(b)], 'r--', alpha=0.7)
    plt.scatter([a, b], [f(a), f(b)], c='red', s=100)
    
    # Plot convergence
    plt.scatter(root, 0, c='green', s=200, label=f'Root: x = {root:.6f}')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Bisection Method')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_bisection_method()

# Code Block 6
def newton_method():
    """Implement Newton's method for finding roots"""
    
    def f(x):
        return x**2 - 4
    
    def f_prime(x):
        return 2 * x
    
    x0 = 3.0  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    
    print("Newton's Method:")
    print(f"Function: f(x) = x² - 4")
    print(f"Derivative: f'(x) = 2x")
    print(f"Initial guess: x₀ = {x0}")
    print(f"Tolerance: {tolerance}")
    print()
    
    iterations = []
    errors = []
    x_values = [x0]
    
    for i in range(max_iterations):
        x_old = x_values[-1]
        f_x = f(x_old)
        f_prime_x = f_prime(x_old)
        
        if abs(f_prime_x) < 1e-10:
            print("Derivative too close to zero")
            break
        
        x_new = x_old - f_x / f_prime_x
        x_values.append(x_new)
        
        iterations.append(i + 1)
        errors.append(abs(f_x))
        
        print(f"Iteration {i+1:2d}: x = {x_new:.6f}, f(x) = {f_x:.6f}")
        
        if abs(f_x) < tolerance:
            print(f"\nRoot found: x = {x_new:.6f}")
            print(f"Function value: f({x_new:.6f}) = {f_x:.6f}")
            print(f"Number of iterations: {i+1}")
            break
    else:
        print("Maximum iterations reached")
    
    return iterations, errors, x_values

iterations, errors, x_values = newton_method()

# Visualize Newton's method
def visualize_newton_method():
    def f(x):
        return x**2 - 4
    
    def f_prime(x):
        return 2 * x
    
    x0 = 3.0
    x_plot = np.linspace(-1, 4, 1000)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Plot function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x² - 4')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot Newton iterations
    for i in range(len(x_values) - 1):
        x_old = x_values[i]
        x_new = x_values[i + 1]
        f_old = f(x_old)
        f_prime_old = f_prime(x_old)
        
        # Plot tangent line
        x_tangent = np.linspace(x_old - 0.5, x_old + 0.5, 100)
        y_tangent = f_old + f_prime_old * (x_tangent - x_old)
        
        plt.plot(x_tangent, y_tangent, 'r--', alpha=0.5)
        plt.scatter(x_old, f_old, c='red', s=100)
        plt.plot([x_old, x_new], [f_old, 0], 'g-', alpha=0.7)
    
    plt.scatter(x_values[-1], 0, c='green', s=200, label=f'Root: x = {x_values[-1]:.6f}')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title("Newton's Method")
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_newton_method()

# Code Block 7
def numerical_gradient_descent():
    """Implement gradient descent using numerical gradients"""
    
    def objective_function(x, y):
        return x**2 + y**2
    
    def numerical_gradient(f, x, y, h=1e-6):
        """Compute numerical gradient using finite differences"""
        grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
        grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return np.array([grad_x, grad_y])
    
    # Gradient descent with numerical gradients
    def gradient_descent_numerical(f, start_point, learning_rate=0.1, max_iter=100):
        x, y = start_point
        history = [(x, y)]
        
        for i in range(max_iter):
            gradient = numerical_gradient(f, x, y)
            x = x - learning_rate * gradient[0]
            y = y - learning_rate * gradient[1]
            history.append((x, y))
            
            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                break
        
        return np.array(history)
    
    # Run gradient descent
    start_point = (2.0, 2.0)
    path = gradient_descent_numerical(objective_function, start_point)
    
    print("Numerical Gradient Descent:")
    print(f"Starting point: {start_point}")
    print(f"Final point: {path[-1]}")
    print(f"Final value: {objective_function(path[-1, 0], path[-1, 1]):.6f}")
    print(f"Number of iterations: {len(path)}")
    
    return path, objective_function

path, objective_function = numerical_gradient_descent()

# Visualize numerical gradient descent
def visualize_numerical_gradient_descent():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Optimization path
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Numerical gradient descent')
    plt.scatter(path[0, 0], path[0, 1], c='red', s=100, label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], c='green', s=100, label='End')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numerical Gradient Descent on f(x,y) = x² + y²')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_numerical_gradient_descent()

# --- Advanced Numerical Methods: Monte Carlo, Adaptive, Multidimensional, Error Analysis ---

# Monte Carlo Integration

def monte_carlo_integration(f, a, b, n_samples=10000):
    """Monte Carlo integration: ∫f(x)dx ≈ (b-a) * (1/N) * Σf(x_i) where x_i ~ U(a,b)."""
    x_samples = np.random.uniform(a, b, n_samples)
    f_samples = f(x_samples)
    integral = (b - a) * np.mean(f_samples)
    error = (b - a) * np.std(f_samples) / np.sqrt(n_samples)
    return integral, error

def test_monte_carlo():
    """Test Monte Carlo integration on ∫x² dx from 0 to 2."""
    def f(x):
        return x**2
    exact = 8/3
    integral, error = monte_carlo_integration(f, 0, 2, 10000)
    print(f"Monte Carlo: {integral:.6f} ± {error:.6f}")
    print(f"Exact value: {exact:.6f}")
    print(f"Relative error: {abs(integral - exact)/exact:.6f}")

test_monte_carlo()

# Adaptive Integration

def adaptive_integration(f, a, b, tol=1e-6):
    """Adaptive integration using Simpson's rule with error estimation."""
    def simpson(f, a, b):
        c = (a + b) / 2
        h = (b - a) / 6
        return h * (f(a) + 4*f(c) + f(b))
    
    def adaptive_recursive(f, a, b, tol):
        c = (a + b) / 2
        S1 = simpson(f, a, b)
        S2 = simpson(f, a, c) + simpson(f, c, b)
        
        if abs(S1 - S2) < 15 * tol:
            return S2
        else:
            return adaptive_recursive(f, a, c, tol/2) + adaptive_recursive(f, c, b, tol/2)
    
    return adaptive_recursive(f, a, b, tol)

def test_adaptive_integration():
    """Test adaptive integration on a function with varying smoothness."""
    def f(x):
        return np.sin(10*x) * np.exp(-x)
    
    exact, _ = integrate.quad(f, 0, 2)
    adaptive_result = adaptive_integration(f, 0, 2)
    print(f"Adaptive integration: {adaptive_result:.6f}")
    print(f"Exact value: {exact:.6f}")
    print(f"Relative error: {abs(adaptive_result - exact)/abs(exact):.6f}")

test_adaptive_integration()

# Secant Method

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """Secant method for root finding: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))."""
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if abs(f1) < tol:
            return x1
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x_new
    return x1

def test_secant_method():
    """Test secant method on f(x) = x² - 4."""
    def f(x):
        return x**2 - 4
    
    root = secant_method(f, 1.0, 3.0)
    print(f"Secant method root: {root:.6f}")
    print(f"Function value at root: {f(root):.6f}")

test_secant_method()

# Fixed Point Iteration

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """Fixed point iteration: x_{n+1} = g(x_n)."""
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def test_fixed_point():
    """Test fixed point iteration on x = cos(x)."""
    def g(x):
        return np.cos(x)
    
    fixed_point = fixed_point_iteration(g, 1.0)
    print(f"Fixed point: {fixed_point:.6f}")
    print(f"cos(fixed_point): {np.cos(fixed_point):.6f}")

test_fixed_point()

# Multidimensional Numerical Methods

def multidimensional_integration(f, bounds, n_samples=10000):
    """Monte Carlo integration for multidimensional functions."""
    # Generate random points in the hypercube
    dim = len(bounds)
    points = np.random.uniform(0, 1, (n_samples, dim))
    
    # Scale points to the actual domain
    scaled_points = np.zeros_like(points)
    volume = 1.0
    for i, (a, b) in enumerate(bounds):
        scaled_points[:, i] = a + (b - a) * points[:, i]
        volume *= (b - a)
    
    f_values = f(scaled_points)
    integral = volume * np.mean(f_values)
    error = volume * np.std(f_values) / np.sqrt(n_samples)
    return integral, error

def test_multidimensional_integration():
    """Test multidimensional integration on ∫∫(x² + y²) dx dy over [0,1]²."""
    def f(x):
        return x[:, 0]**2 + x[:, 1]**2
    
    bounds = [(0, 1), (0, 1)]
    integral, error = multidimensional_integration(f, bounds)
    exact = 2/3  # ∫∫(x² + y²) dx dy = ∫x² dx + ∫y² dy = 1/3 + 1/3 = 2/3
    print(f"2D Monte Carlo: {integral:.6f} ± {error:.6f}")
    print(f"Exact value: {exact:.6f}")

test_multidimensional_integration()

# Advanced Error Analysis

def condition_number_analysis():
    """Analyze condition number for numerical differentiation."""
    def f(x):
        return np.exp(x)
    
    def f_prime(x):
        return np.exp(x)
    
    x = 1.0
    h_values = np.logspace(-16, 0, 100)
    errors = []
    
    for h in h_values:
        # Forward difference
        approx = (f(x + h) - f(x)) / h
        exact = f_prime(x)
        error = abs(approx - exact) / abs(exact)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors, 'b-', label='Forward difference error')
    plt.xlabel('Step size h')
    plt.ylabel('Relative error')
    plt.title('Error Analysis for Numerical Differentiation')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Find optimal step size
    min_idx = np.argmin(errors)
    optimal_h = h_values[min_idx]
    print(f"Optimal step size: {optimal_h:.2e}")
    print(f"Minimum relative error: {errors[min_idx]:.2e}")

condition_number_analysis()

# Error Propagation Analysis

def error_propagation_example():
    """Demonstrate error propagation in composite functions."""
    def f(x):
        return x**2
    
    def g(x):
        return np.sin(x)
    
    def composite(x):
        return f(g(x))
    
    x = 1.0
    h = 1e-6
    
    # Numerical derivatives
    df_dx = (f(x + h) - f(x - h)) / (2 * h)
    dg_dx = (g(x + h) - g(x - h)) / (2 * h)
    d_composite_dx = (composite(x + h) - composite(x - h)) / (2 * h)
    
    # Chain rule: d(f∘g)/dx = f'(g(x)) * g'(x)
    chain_rule = df_dx * dg_dx
    actual_derivative = d_composite_dx
    
    print(f"Chain rule result: {chain_rule:.6f}")
    print(f"Actual derivative: {actual_derivative:.6f}")
    print(f"Relative error: {abs(chain_rule - actual_derivative)/abs(actual_derivative):.6f}")

error_propagation_example()

# Test all advanced methods
def test_advanced_methods():
    """Test all advanced numerical methods."""
    print("=== Testing Advanced Numerical Methods ===")
    
    # Test Monte Carlo
    def f(x):
        return np.exp(-x**2)
    mc_result, mc_error = monte_carlo_integration(f, -2, 2, 10000)
    print(f"Monte Carlo integration: {mc_result:.6f} ± {mc_error:.6f}")
    
    # Test adaptive integration
    adaptive_result = adaptive_integration(f, -2, 2)
    print(f"Adaptive integration: {adaptive_result:.6f}")
    
    # Test secant method
    def g(x):
        return x**3 - 2
    secant_root = secant_method(g, 0.0, 2.0)
    print(f"Secant method root: {secant_root:.6f}")
    
    # Test fixed point
    def h(x):
        return np.sqrt(2 - x**2)
    fixed_pt = fixed_point_iteration(h, 1.0)
    print(f"Fixed point: {fixed_pt:.6f}")

test_advanced_methods()

