# Numerical Methods in Calculus

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Numerical methods provide computational techniques for solving calculus problems when analytical solutions are difficult or impossible to obtain. These methods are essential for practical applications in engineering, physics, machine learning, and data science where exact solutions may not be available or computationally feasible.

**Key Concepts:**
- **Approximation:** Numerical methods provide approximate solutions with controlled error
- **Discretization:** Continuous problems are converted to discrete computational problems
- **Convergence:** Methods improve accuracy as computational effort increases
- **Stability:** Small errors in input don't lead to large errors in output

**Relevance to AI/ML:**
- Numerical integration appears in probability calculations and model evaluation
- Numerical differentiation is used in gradient computation and sensitivity analysis
- Understanding numerical methods helps choose appropriate algorithms and assess accuracy

---

## 10.1 Numerical Integration

### Mathematical Foundations

Numerical integration approximates definite integrals when analytical solutions are unavailable. The goal is to compute:
\[
\int_a^b f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
\]
where \( w_i \) are weights and \( x_i \) are evaluation points.

**Key Properties:**
- **Rectangle Rule:** Uses function values at left endpoints of subintervals
- **Trapezoidal Rule:** Uses linear interpolation between points
- **Simpson's Rule:** Uses quadratic interpolation for higher accuracy
- **Error Analysis:** Error typically decreases as \( O(h^p) \) where \( h \) is step size and \( p \) is order

**Relevance to AI/ML:**
- Computing expectations and probabilities in probabilistic models
- Evaluating model performance metrics (AUC, etc.)
- Numerical optimization and sampling methods

### Python Implementation: Rectangle Rule

The rectangle rule approximates the integral by summing rectangles under the curve. For a partition \( a = x_0 < x_1 < \cdots < x_n = b \):
\[
\int_a^b f(x) \, dx \approx h \sum_{i=0}^{n-1} f(x_i)
\]
where \( h = (b-a)/n \) is the step size.

```python
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
```

**Explanation:**
- The rectangle rule approximates the integral by summing the areas of rectangles
- Each rectangle has height equal to the function value at the left endpoint
- The approximation improves as the number of subintervals increases
- The visualization shows how rectangles approximate the area under the curve

---

### Python Implementation: Trapezoidal Rule

The trapezoidal rule uses linear interpolation between points. For a partition \( a = x_0 < x_1 < \cdots < x_n = b \):
\[
\int_a^b f(x) \, dx \approx \frac{h}{2} \left(f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right)
\]
where \( h = (b-a)/n \) is the step size.

```python
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
```

**Explanation:**
- The trapezoidal rule uses linear interpolation between adjacent points
- Each trapezoid has area \( \frac{h}{2}(f(x_i) + f(x_{i+1})) \)
- The method is more accurate than the rectangle rule (O(h²) vs O(h))
- The visualization shows how trapezoids approximate the area under the curve

---

### Python Implementation: Simpson's Rule

Simpson's rule uses quadratic interpolation for even higher accuracy. For an even number of subintervals:
\[
\int_a^b f(x) \, dx \approx \frac{h}{3} \left(f(x_0) + 4\sum_{i=1,3,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,\ldots}^{n-2} f(x_i) + f(x_n)\right)
\]

```python
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
```

**Explanation:**
- Simpson's rule uses quadratic interpolation for higher accuracy
- The method requires an even number of subintervals
- Error decreases as O(h⁴), making it much more accurate than other methods
- The comparison shows the relative accuracy of different methods

---

## 10.2 Numerical Differentiation

### Mathematical Foundations

Numerical differentiation approximates derivatives when analytical differentiation is difficult. The goal is to compute:
\[
f'(x) \approx \frac{f(x+h) - f(x)}{h}
\]
for some small step size \( h \).

**Key Methods:**
- **Forward Difference:** \( f'(x) \approx \frac{f(x+h) - f(x)}{h} \)
- **Backward Difference:** \( f'(x) \approx \frac{f(x) - f(x-h)}{h} \)
- **Central Difference:** \( f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \)

**Error Analysis:**
- Forward/Backward: Error \( O(h) \)
- Central: Error \( O(h²) \) - more accurate
- Trade-off between accuracy and step size

**Relevance to AI/ML:**
- Computing gradients for optimization algorithms
- Sensitivity analysis of model parameters
- Finite difference methods for complex functions

### Python Implementation: Finite Difference Methods

```python
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
```

**Explanation:**
- Forward difference uses the function value at the current point and one step ahead
- Backward difference uses the function value at the current point and one step back
- Central difference uses points on both sides, providing better accuracy
- The error analysis shows how accuracy depends on step size and method choice

---

## 10.3 Root Finding Methods

### Bisection Method

```python
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
```

### Newton's Method

```python
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
```

## 10.4 Applications in Machine Learning

### Gradient Descent with Numerical Gradients

```python
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
```

## Summary

- **Numerical Integration**: Rectangle, trapezoidal, and Simpson's rules for approximating definite integrals
- **Numerical Differentiation**: Forward, backward, and central difference methods for approximating derivatives
- **Root Finding**: Bisection and Newton's methods for finding zeros of functions
- **Applications**: Gradient descent with numerical gradients when analytical derivatives are unavailable
- **Error Analysis**: Understanding convergence and accuracy of numerical methods

## Next Steps

Numerical methods provide essential tools for solving calculus problems computationally. These techniques are fundamental to scientific computing, machine learning algorithms, and engineering applications where analytical solutions are not feasible. 