# Integration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Integration is the reverse process of differentiation and is essential for calculating areas, volumes, and cumulative effects. In machine learning, integration is used for probability calculations, expected values, and continuous optimization.

### Why Integration Matters in AI/ML

Integration plays a crucial role in machine learning and data science:

1. **Probability Theory**: Computing probabilities, expected values, and cumulative distribution functions
2. **Bayesian Inference**: Marginalization and evidence computation
3. **Continuous Optimization**: Area under curves, cumulative effects
4. **Signal Processing**: Fourier transforms and spectral analysis
5. **Neural Networks**: Activation function integrals and normalization

### Mathematical Foundation

Integration can be understood in two complementary ways:

1. **Antiderivative**: If F'(x) = f(x), then F(x) is an antiderivative of f(x)
2. **Area Under Curve**: The definite integral ∫ₐᵇ f(x) dx represents the signed area between the curve y = f(x) and the x-axis from x = a to x = b

### Fundamental Theorem of Calculus

The fundamental theorem connects differentiation and integration:

**Part 1**: If F(x) = ∫ₐˣ f(t) dt, then F'(x) = f(x)

**Part 2**: If F(x) is any antiderivative of f(x), then ∫ₐᵇ f(x) dx = F(b) - F(a)

This theorem is the foundation that makes integration computationally tractable.

## 4.1 Antiderivatives and Indefinite Integrals

The indefinite integral finds the antiderivative of a function. Unlike definite integrals, indefinite integrals include an arbitrary constant of integration.

### Mathematical Definition

The indefinite integral of f(x) is:

$$\int f(x) dx = F(x) + C$$

where F'(x) = f(x) and C is the constant of integration.

### Key Properties

1. **Linearity**: ∫(af(x) + bg(x)) dx = a∫f(x) dx + b∫g(x) dx
2. **Power Rule**: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C (for n ≠ -1)
3. **Exponential**: ∫eˣ dx = eˣ + C
4. **Trigonometric**: ∫sin(x) dx = -cos(x) + C, ∫cos(x) dx = sin(x) + C

### Why the Constant of Integration Matters

The constant C represents the fact that any function differing by a constant has the same derivative. This is crucial for:
- Initial value problems
- Boundary conditions
- Physical interpretations

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate

# Comprehensive antiderivative demonstration
def demonstrate_antiderivatives_comprehensive():
    x = sp.Symbol('x')
    
    print("=== ANTIDERIVATIVE EXAMPLES ===\n")
    
    # Basic power functions
    print("1. POWER FUNCTIONS")
    power_examples = [x**0, x**1, x**2, x**3, x**(-1), x**(-2)]
    for expr in power_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")
    
    # Exponential and logarithmic functions
    print("\n2. EXPONENTIAL AND LOGARITHMIC FUNCTIONS")
    exp_expr = sp.exp(x)
    log_expr = sp.log(x)
    print(f"   ∫ {exp_expr} dx = {sp.integrate(exp_expr, x)}")
    print(f"   ∫ {log_expr} dx = {sp.integrate(log_expr, x)}")
    
    # Trigonometric functions
    print("\n3. TRIGONOMETRIC FUNCTIONS")
    trig_examples = [sp.sin(x), sp.cos(x), sp.tan(x), 1/sp.cos(x)**2]
    for expr in trig_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")
    
    # Composite functions
    print("\n4. COMPOSITE FUNCTIONS")
    composite_examples = [
        sp.sin(x**2) * x,  # Requires substitution
        sp.exp(x**2) * x,  # Requires substitution
        sp.log(x) / x,     # Requires substitution
        sp.sin(x) * sp.cos(x)  # Requires trigonometric identity
    ]
    
    for expr in composite_examples:
        try:
            integral = sp.integrate(expr, x)
            print(f"   ∫ {expr} dx = {integral}")
        except:
            print(f"   ∫ {expr} dx = (requires advanced techniques)")

demonstrate_antiderivatives_comprehensive()

# Visualize antiderivatives with multiple constants
def visualize_antiderivatives():
    def f(x):
        return x**2
    
    def F(x, C=0):
        return x**3 / 3 + C
    
    x_vals = np.linspace(-2, 2, 100)
    y_vals = f(x_vals)
    
    # Multiple antiderivatives with different constants
    constants = [-2, -1, 0, 1, 2]
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original function
    ax1.plot(x_vals, y_vals, 'b-', linewidth=3, label='f(x) = x²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Original Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Antiderivatives
    for i, (C, color) in enumerate(zip(constants, colors)):
        Y_vals = F(x_vals, C)
        ax2.plot(x_vals, Y_vals, color=color, linewidth=2, 
                label=f'F(x) = x³/3 + {C}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title('Antiderivatives (Different Constants)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate that all antiderivatives have the same derivative
    print("\nVerification: All antiderivatives have the same derivative")
    x_test = 1.0
    for C in constants:
        # Numerical derivative of antiderivative
        h = 1e-7
        derivative = (F(x_test + h, C) - F(x_test - h, C)) / (2 * h)
        print(f"  F(x) = x³/3 + {C}: F'({x_test}) = {derivative:.6f}")

visualize_antiderivatives()

# Advanced antiderivative examples
def advanced_antiderivative_examples():
    x = sp.Symbol('x')
    
    print("\n=== ADVANCED ANTIDERIVATIVE EXAMPLES ===\n")
    
    # Rational functions
    print("1. RATIONAL FUNCTIONS")
    rational_examples = [
        1 / (x + 1),
        1 / (x**2 + 1),
        x / (x**2 + 1),
        1 / (x**2 - 1)
    ]
    
    for expr in rational_examples:
        try:
            integral = sp.integrate(expr, x)
            print(f"   ∫ {expr} dx = {integral}")
        except:
            print(f"   ∫ {expr} dx = (complex result)")
    
    # Exponential combinations
    print("\n2. EXPONENTIAL COMBINATIONS")
    exp_examples = [
        sp.exp(x) * sp.sin(x),
        sp.exp(x) * sp.cos(x),
        x * sp.exp(x),
        x**2 * sp.exp(x)
    ]
    
    for expr in exp_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")
    
    # Trigonometric combinations
    print("\n3. TRIGONOMETRIC COMBINATIONS")
    trig_examples = [
        sp.sin(x)**2,
        sp.cos(x)**2,
        sp.sin(x) * sp.cos(x),
        sp.sin(x)**3
    ]
    
    for expr in trig_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")

advanced_antiderivative_examples()

# Numerical verification of antiderivatives
def numerical_verification():
    print("\n=== NUMERICAL VERIFICATION ===\n")
    
    def f(x):
        return x**2
    
    def F(x, C=0):
        return x**3 / 3 + C
    
    def numerical_derivative(F, x, h=1e-7):
        return (F(x + h) - F(x - h)) / (2 * h)
    
    # Test points
    test_points = [0.5, 1.0, 1.5, 2.0]
    
    print("Verifying that F'(x) = f(x):")
    for x_test in test_points:
        # Analytical derivative of antiderivative
        analytical_deriv = f(x_test)
        
        # Numerical derivative of antiderivative
        numerical_deriv = numerical_derivative(lambda x: F(x, 0), x_test)
        
        error = abs(analytical_deriv - numerical_deriv)
        print(f"  x = {x_test}: f(x) = {analytical_deriv:.6f}, F'(x) = {numerical_deriv:.6f}, error = {error:.2e}")

numerical_verification()

# Physical interpretation of antiderivatives
def physical_interpretation():
    print("\n=== PHYSICAL INTERPRETATION ===\n")
    
    # Example: Position, velocity, and acceleration
    def acceleration(t):
        return 2 * t  # a(t) = 2t
    
    def velocity(t, v0=0):
        return t**2 + v0  # v(t) = ∫a(t)dt = t² + v₀
    
    def position(t, x0=0, v0=0):
        return t**3 / 3 + v0 * t + x0  # x(t) = ∫v(t)dt = t³/3 + v₀t + x₀
    
    # Test time points
    t_vals = np.linspace(0, 5, 100)
    
    # Calculate values
    a_vals = [acceleration(t) for t in t_vals]
    v_vals = [velocity(t) for t in t_vals]
    x_vals = [position(t) for t in t_vals]
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(t_vals, a_vals, 'r-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Acceleration: a(t) = 2t')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_vals, v_vals, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity: v(t) = t²')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(t_vals, x_vals, 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position: x(t) = t³/3')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Verify relationships
    print("Physical relationships:")
    t_test = 2.0
    print(f"  At t = {t_test}s:")
    print(f"    Acceleration: a({t_test}) = {acceleration(t_test)} m/s²")
    print(f"    Velocity: v({t_test}) = {velocity(t_test)} m/s")
    print(f"    Position: x({t_test}) = {position(t_test)} m")
    print(f"    Verification: v'({t_test}) = {acceleration(t_test)} ✓")
    print(f"    Verification: x'({t_test}) = {velocity(t_test)} ✓")

physical_interpretation()
```

### Applications in Machine Learning

Antiderivatives are fundamental to:

1. **Activation Functions**: Computing integrals of activation functions for normalization
2. **Loss Functions**: Understanding cumulative loss over time
3. **Probability Distributions**: Computing cumulative distribution functions
4. **Signal Processing**: Fourier transforms and spectral analysis
5. **Optimization**: Understanding cumulative effects in gradient-based methods

## 4.2 Definite Integrals

Definite integrals calculate the area under a curve between two points. They represent the net accumulation of a quantity over an interval and are fundamental to many applications in science and engineering.

### Mathematical Definition

The definite integral of f(x) from a to b is:

$$\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i) \Delta x$$

where Δx = (b-a)/n and xᵢ are sample points in the interval [a, b].

### Geometric Interpretation

- **Positive Area**: When f(x) ≥ 0, the integral represents the area between the curve and the x-axis
- **Negative Area**: When f(x) ≤ 0, the integral represents the negative of the area
- **Net Area**: The definite integral gives the net signed area (positive minus negative)

### Fundamental Theorem of Calculus (Part 2)

If F(x) is any antiderivative of f(x), then:

$$\int_a^b f(x) dx = F(b) - F(a) = [F(x)]_a^b$$

This provides a powerful computational method for evaluating definite integrals.

```python
# Comprehensive definite integration demonstration
def definite_integration_comprehensive():
    x = sp.Symbol('x')
    
    print("=== DEFINITE INTEGRATION EXAMPLES ===\n")
    
    # Basic examples
    print("1. BASIC EXAMPLES")
    
    # ∫₀¹ x² dx = [x³/3]₀¹ = 1/3
    integral1 = sp.integrate(x**2, (x, 0, 1))
    print(f"   ∫₀¹ x² dx = {integral1}")
    
    # ∫₀^π sin(x) dx = [-cos(x)]₀^π = 2
    integral2 = sp.integrate(sp.sin(x), (x, 0, sp.pi))
    print(f"   ∫₀^π sin(x) dx = {integral2}")
    
    # ∫₀^∞ e^(-x) dx = [-e^(-x)]₀^∞ = 1
    integral3 = sp.integrate(sp.exp(-x), (x, 0, sp.oo))
    print(f"   ∫₀^∞ e^(-x) dx = {integral3}")
    
    # ∫₋₁¹ x³ dx = [x⁴/4]₋₁¹ = 0 (odd function over symmetric interval)
    integral4 = sp.integrate(x**3, (x, -1, 1))
    print(f"   ∫₋₁¹ x³ dx = {integral4}")
    
    # Area calculations
    print("\n2. AREA CALCULATIONS")
    
    # Area under y = x² from 0 to 2
    area1 = sp.integrate(x**2, (x, 0, 2))
    print(f"   Area under y = x² from 0 to 2: {area1}")
    
    # Area between y = x and y = x² from 0 to 1
    area2 = sp.integrate(x - x**2, (x, 0, 1))
    print(f"   Area between y = x and y = x² from 0 to 1: {area2}")
    
    # Improper integrals
    print("\n3. IMPROPER INTEGRALS")
    
    # ∫₀^∞ e^(-x) dx = 1
    improper1 = sp.integrate(sp.exp(-x), (x, 0, sp.oo))
    print(f"   ∫₀^∞ e^(-x) dx = {improper1}")
    
    # ∫₋∞^∞ e^(-x²/2) dx = √(2π)
    improper2 = sp.integrate(sp.exp(-x**2/2), (x, -sp.oo, sp.oo))
    print(f"   ∫₋∞^∞ e^(-x²/2) dx = {improper2}")
    
    # ∫₁^∞ 1/x² dx = 1
    improper3 = sp.integrate(1/x**2, (x, 1, sp.oo))
    print(f"   ∫₁^∞ 1/x² dx = {improper3}")

definite_integration_comprehensive()

# Advanced numerical integration with error analysis
def numerical_integration_advanced():
    print("\n=== NUMERICAL INTEGRATION ANALYSIS ===\n")
    
    # Test functions
    def f1(x): return x**2
    def f2(x): return np.sin(x)
    def f3(x): return np.exp(-x)
    def f4(x): return 1 / (1 + x**2)
    
    # Integration intervals
    intervals = [
        (f1, 0, 1, "∫₀¹ x² dx", 1/3),
        (f2, 0, np.pi, "∫₀^π sin(x) dx", 2),
        (f3, 0, np.inf, "∫₀^∞ e^(-x) dx", 1),
        (f4, 0, 1, "∫₀¹ 1/(1+x²) dx", np.pi/4)
    ]
    
    print("Function\t\tExact\t\tNumerical\tError")
    print("-" * 60)
    
    for func, a, b, name, exact in intervals:
        if b == np.inf:
            result, error = integrate.quad(func, a, b)
        else:
            result, error = integrate.quad(func, a, b)
        
        abs_error = abs(result - exact)
        rel_error = abs_error / abs(exact) if exact != 0 else abs_error
        
        print(f"{name}\t{exact:.6f}\t{result:.6f}\t{abs_error:.2e}")
        print(f"  Estimated error: {error:.2e}")
        print(f"  Relative error: {rel_error:.2e}")
        print()

numerical_integration_advanced()

# Comprehensive visualization of definite integrals
def visualize_definite_integrals():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Example 1: Positive area
    x1 = np.linspace(0, 1, 1000)
    y1 = x1**2
    
    ax1.fill_between(x1, y1, alpha=0.3, color='blue')
    ax1.plot(x1, y1, 'b-', linewidth=2, label='f(x) = x²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('∫₀¹ x² dx = 1/3 (Positive Area)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Example 2: Negative area
    x2 = np.linspace(0, np.pi, 1000)
    y2 = -np.sin(x2)
    
    ax2.fill_between(x2, y2, alpha=0.3, color='red')
    ax2.plot(x2, y2, 'r-', linewidth=2, label='f(x) = -sin(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('∫₀^π -sin(x) dx = -2 (Negative Area)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Example 3: Net area (positive and negative)
    x3 = np.linspace(0, 2*np.pi, 1000)
    y3 = np.sin(x3)
    
    # Separate positive and negative regions
    positive_mask = y3 >= 0
    negative_mask = y3 < 0
    
    ax3.fill_between(x3[positive_mask], y3[positive_mask], alpha=0.3, color='green')
    ax3.fill_between(x3[negative_mask], y3[negative_mask], alpha=0.3, color='red')
    ax3.plot(x3, y3, 'b-', linewidth=2, label='f(x) = sin(x)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.set_title('∫₀^{2π} sin(x) dx = 0 (Net Area)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Example 4: Area between curves
    x4 = np.linspace(0, 1, 1000)
    y4_upper = x4
    y4_lower = x4**2
    
    ax4.fill_between(x4, y4_upper, y4_lower, alpha=0.3, color='purple')
    ax4.plot(x4, y4_upper, 'b-', linewidth=2, label='y = x')
    ax4.plot(x4, y4_lower, 'r-', linewidth=2, label='y = x²')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Area between y = x and y = x²')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_definite_integrals()

# Riemann sum approximation
def riemann_sum_analysis():
    print("\n=== RIEMANN SUM ANALYSIS ===\n")
    
    def f(x):
        return x**2
    
    def riemann_sum(f, a, b, n, method='left'):
        """Compute Riemann sum using specified method"""
        x = np.linspace(a, b, n+1)
        h = (b - a) / n
        
        if method == 'left':
            x_points = x[:-1]
        elif method == 'right':
            x_points = x[1:]
        elif method == 'midpoint':
            x_points = (x[:-1] + x[1:]) / 2
        elif method == 'trapezoidal':
            return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))
        
        return h * np.sum(f(x_points))
    
    # Test different methods and numbers of subintervals
    a, b = 0, 1
    exact = 1/3
    
    methods = ['left', 'right', 'midpoint', 'trapezoidal']
    n_values = [10, 50, 100, 500]
    
    print("Method\t\tN=10\t\tN=50\t\tN=100\t\tN=500")
    print("-" * 80)
    
    for method in methods:
        results = []
        for n in n_values:
            result = riemann_sum(f, a, b, n, method)
            results.append(result)
        
        errors = [abs(r - exact) for r in results]
        print(f"{method:12s}", end="")
        for result in results:
            print(f"{result:.6f}\t", end="")
        print()
        print(f"{'Error':12s}", end="")
        for error in errors:
            print(f"{error:.2e}\t", end="")
        print()
        print()

riemann_sum_analysis()

# Properties of definite integrals
def integral_properties():
    print("\n=== INTEGRAL PROPERTIES ===\n")
    
    x = sp.Symbol('x')
    
    # Linearity: ∫(af(x) + bg(x)) dx = a∫f(x) dx + b∫g(x) dx
    print("1. LINEARITY")
    a, b = 2, 3
    f_expr = x**2
    g_expr = sp.sin(x)
    
    # Left side: ∫(2x² + 3sin(x)) dx from 0 to 1
    left_side = sp.integrate(a * f_expr + b * g_expr, (x, 0, 1))
    
    # Right side: 2∫x² dx + 3∫sin(x) dx from 0 to 1
    right_side = a * sp.integrate(f_expr, (x, 0, 1)) + b * sp.integrate(g_expr, (x, 0, 1))
    
    print(f"   ∫(2x² + 3sin(x)) dx = {left_side}")
    print(f"   2∫x² dx + 3∫sin(x) dx = {right_side}")
    print(f"   Linearity holds: {left_side == right_side}")
    
    # Additivity: ∫ₐᵇ f(x) dx + ∫ᵇᶜ f(x) dx = ∫ₐᶜ f(x) dx
    print("\n2. ADDITIVITY")
    a, b, c = 0, 1, 2
    
    left_sum = sp.integrate(x**2, (x, a, b)) + sp.integrate(x**2, (x, b, c))
    right_sum = sp.integrate(x**2, (x, a, c))
    
    print(f"   ∫₀¹ x² dx + ∫₁² x² dx = {left_sum}")
    print(f"   ∫₀² x² dx = {right_sum}")
    print(f"   Additivity holds: {left_sum == right_sum}")
    
    # Symmetry properties
    print("\n3. SYMMETRY PROPERTIES")
    
    # Even function: ∫₋ₐᵃ f(x) dx = 2∫₀ᵃ f(x) dx
    even_func = x**2
    a_val = 2
    
    even_left = sp.integrate(even_func, (x, -a_val, a_val))
    even_right = 2 * sp.integrate(even_func, (x, 0, a_val))
    
    print(f"   ∫₋₂² x² dx = {even_left}")
    print(f"   2∫₀² x² dx = {even_right}")
    print(f"   Even function property holds: {even_left == even_right}")
    
    # Odd function: ∫₋ₐᵃ f(x) dx = 0
    odd_func = x**3
    odd_result = sp.integrate(odd_func, (x, -a_val, a_val))
    print(f"   ∫₋₂² x³ dx = {odd_result}")
    print(f"   Odd function property holds: {odd_result == 0}")

integral_properties()

### Applications in Machine Learning

Definite integrals are essential for:

1. **Probability Calculations**: Computing probabilities from probability density functions
2. **Expected Values**: Calculating means and moments of continuous distributions
3. **Loss Function Analysis**: Understanding cumulative loss over intervals
4. **Feature Importance**: Computing area-based feature contributions
5. **Model Evaluation**: Area under ROC curves and precision-recall curves

## 4.3 Integration Techniques

### Substitution Method

```python
# Integration by substitution
def substitution_examples():
    # ∫ x*e^(x²) dx using u = x²
    # du = 2x dx, so dx = du/(2x)
    # ∫ x*e^(x²) dx = ∫ x*e^u * du/(2x) = (1/2)∫ e^u du = (1/2)e^u + C = (1/2)e^(x²) + C
    
    # Using SymPy
    integral = sp.integrate(x * sp.exp(x**2), x)
    print(f"∫ x*e^(x²) dx = {integral}")
    
    # ∫ sin(2x) dx using u = 2x
    integral2 = sp.integrate(sp.sin(2*x), x)
    print(f"∫ sin(2x) dx = {integral2}")

substitution_examples()
```

### Integration by Parts

```python
# Integration by parts: ∫ u dv = uv - ∫ v du
def integration_by_parts_examples():
    # ∫ x*e^x dx
    # Let u = x, dv = e^x dx
    # Then du = dx, v = e^x
    # ∫ x*e^x dx = x*e^x - ∫ e^x dx = x*e^x - e^x + C
    
    integral = sp.integrate(x * sp.exp(x), x)
    print(f"∫ x*e^x dx = {integral}")
    
    # ∫ x*ln(x) dx
    integral2 = sp.integrate(x * sp.log(x), x)
    print(f"∫ x*ln(x) dx = {integral2}")

integration_by_parts_examples()
```

## 4.4 Applications in Probability and Statistics

### Probability Density Functions

```python
# Normal distribution integration
def normal_distribution_integration():
    # Standard normal distribution: f(x) = (1/√(2π)) * e^(-x²/2)
    def normal_pdf(x):
        return (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    
    # ∫₋∞^∞ f(x) dx = 1 (total probability)
    total_prob, error = integrate.quad(normal_pdf, -np.inf, np.inf)
    print(f"Total probability: {total_prob:.6f}")
    
    # P(-1 ≤ X ≤ 1) = ∫₋₁¹ f(x) dx
    prob_1sigma, error = integrate.quad(normal_pdf, -1, 1)
    print(f"P(-1 ≤ X ≤ 1): {prob_1sigma:.6f}")
    
    # P(-2 ≤ X ≤ 2) = ∫₋₂² f(x) dx
    prob_2sigma, error = integrate.quad(normal_pdf, -2, 2)
    print(f"P(-2 ≤ X ≤ 2): {prob_2sigma:.6f}")
    
    # Visualize
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = normal_pdf(x_vals)
    
    plt.figure(figsize=(12, 5))
    
    # Full distribution
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Standard Normal Distribution')
    plt.grid(True)
    
    # ±1σ region
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    mask = (x_vals >= -1) & (x_vals <= 1)
    plt.fill_between(x_vals[mask], y_vals[mask], alpha=0.5, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('P(-1 ≤ X ≤ 1)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

normal_distribution_integration()
```

### Expected Values

```python
# Expected value calculations
def expected_value_examples():
    # E[X] = ∫ x*f(x) dx for continuous random variables
    
    # Exponential distribution: f(x) = λ*e^(-λx) for x ≥ 0
    def exponential_pdf(x, lambda_param=1):
        return lambda_param * np.exp(-lambda_param * x)
    
    # E[X] = ∫₀^∞ x*λ*e^(-λx) dx = 1/λ
    def expected_value_exponential(lambda_param=1):
        integrand = lambda x: x * exponential_pdf(x, lambda_param)
        result, error = integrate.quad(integrand, 0, np.inf)
        return result
    
    expected_val = expected_value_exponential(1)
    print(f"E[X] for exponential(λ=1): {expected_val:.6f}")
    print(f"Theoretical value: {1/1:.6f}")
    
    # Variance: Var(X) = E[X²] - (E[X])²
    def variance_exponential(lambda_param=1):
        integrand = lambda x: x**2 * exponential_pdf(x, lambda_param)
        e_x_squared, _ = integrate.quad(integrand, 0, np.inf)
        e_x = expected_value_exponential(lambda_param)
        return e_x_squared - e_x**2
    
    variance = variance_exponential(1)
    print(f"Var(X) for exponential(λ=1): {variance:.6f}")
    print(f"Theoretical value: {1/1**2:.6f}")

expected_value_examples()
```

## 4.5 Numerical Integration Methods

### Trapezoidal Rule

```python
# Trapezoidal rule implementation
def trapezoidal_rule(f, a, b, n=1000):
    """Approximate ∫ₐᵇ f(x) dx using trapezoidal rule"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# Compare different methods
def compare_integration_methods():
    f = lambda x: x**2
    
    # Exact value
    exact = 1/3
    
    # Trapezoidal rule
    trapezoidal = trapezoidal_rule(f, 0, 1, 100)
    
    # SciPy quad
    scipy_result, _ = integrate.quad(f, 0, 1)
    
    print(f"Exact value: {exact:.6f}")
    print(f"Trapezoidal rule: {trapezoidal:.6f}")
    print(f"SciPy quad: {scipy_result:.6f}")
    
    # Error analysis
    trapezoidal_error = abs(trapezoidal - exact)
    scipy_error = abs(scipy_result - exact)
    
    print(f"Trapezoidal error: {trapezoidal_error:.2e}")
    print(f"SciPy error: {scipy_error:.2e}")

compare_integration_methods()
```

### Monte Carlo Integration

```python
# Monte Carlo integration
def monte_carlo_integration(f, a, b, n=10000):
    """Approximate ∫ₐᵇ f(x) dx using Monte Carlo method"""
    x_random = np.random.uniform(a, b, n)
    y_random = f(x_random)
    return (b - a) * np.mean(y_random)

# Example: ∫₀¹ x² dx
def monte_carlo_example():
    f = lambda x: x**2
    exact = 1/3
    
    # Run multiple times to see variation
    results = []
    for i in range(10):
        result = monte_carlo_integration(f, 0, 1, 10000)
        results.append(result)
    
    print(f"Exact value: {exact:.6f}")
    print(f"Monte Carlo results: {[f'{r:.6f}' for r in results]}")
    print(f"Mean: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")

monte_carlo_example()
```

## 4.6 Applications in Machine Learning

### Loss Function Integration

```python
# Integration in loss functions
def loss_function_integration():
    # Expected loss over a distribution
    def expected_loss(loss_func, data_distribution, n_samples=10000):
        """Calculate expected loss over a data distribution"""
        samples = data_distribution.rvs(n_samples)
        losses = [loss_func(x) for x in samples]
        return np.mean(losses)
    
    # Example: Expected MSE loss
    from scipy.stats import norm
    
    def mse_loss(x, target=0):
        return (x - target)**2
    
    # Expected MSE when data follows N(0, 1)
    expected_mse = expected_loss(mse_loss, norm(0, 1))
    print(f"Expected MSE for N(0,1): {expected_mse:.6f}")
    
    # Theoretical: E[(X-0)²] = Var(X) = 1
    print(f"Theoretical value: {1:.6f}")

loss_function_integration()
```

### Area Under ROC Curve (AUC)

```python
# AUC calculation using integration
def calculate_auc():
    # Simulate ROC curve data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate scores for two classes
    scores_class_0 = np.random.normal(0, 1, n_samples//2)
    scores_class_1 = np.random.normal(1, 1, n_samples//2)
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    y_scores = np.concatenate([scores_class_0, scores_class_1])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    print(f"AUC: {auc_score:.6f}")
    
    # Visualize ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

calculate_auc()
```

## Summary

- **Antiderivatives** reverse the process of differentiation
- **Definite integrals** calculate areas and cumulative effects
- **Integration techniques** include substitution and integration by parts
- **Numerical methods** provide approximations when symbolic integration is difficult
- **Applications** include probability calculations, expected values, and performance metrics
- **Machine learning** uses integration for AUC, expected losses, and probability distributions

## Next Steps

Integration provides the foundation for understanding areas, volumes, and cumulative effects. In the next section, we'll explore applications of integration in optimization and advanced calculus concepts. 