# Derivatives and Differentiation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)

## Introduction

Derivatives measure how a function changes as its input changes. This concept is fundamental to optimization, which is central to machine learning algorithms.

### Why Derivatives Matter in AI/ML

Derivatives are the cornerstone of optimization in machine learning. They enable us to:

1. **Find Optimal Solutions**: Locate minima and maxima of loss functions
2. **Gradient-Based Optimization**: Implement algorithms like gradient descent, Adam, and RMSprop
3. **Neural Network Training**: Compute gradients for backpropagation
4. **Model Sensitivity Analysis**: Understand how changes in inputs affect outputs
5. **Feature Importance**: Determine which features contribute most to predictions

### Mathematical Foundation

The derivative of a function f(x) at a point x₀ is defined as the limit of the difference quotient:

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$

This represents the instantaneous rate of change of the function at that point, or geometrically, the slope of the tangent line to the curve at that point.

### Physical and Geometric Interpretation

- **Rate of Change**: How quickly the function value changes with respect to the input
- **Slope**: The steepness of the function at a particular point
- **Velocity**: In physics, the derivative of position with respect to time
- **Sensitivity**: How sensitive the output is to small changes in the input

## 2.1 Definition of Derivatives

The derivative captures the instantaneous rate of change of a function. It's the foundation for understanding how functions behave locally and globally.

### Key Concepts:

- **Instantaneous Rate**: The rate of change at a specific point, not over an interval
- **Tangent Line**: The line that best approximates the function near a point
- **Local Linearity**: Functions behave approximately linearly near any point
- **Differentiability**: A function is differentiable if its derivative exists at a point

### Example: Understanding the Difference Quotient

The difference quotient $\frac{f(x + h) - f(x)}{h}$ represents the average rate of change over the interval [x, x+h]. As h approaches 0, this becomes the instantaneous rate of change.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative

# Define a function for detailed analysis
def f(x):
    """Function to analyze: f(x) = x²"""
    return x**2

# Numerical derivative using finite differences with detailed analysis
def numerical_derivative_detailed(f, x, h=1e-7):
    """
    Compute numerical derivative using central difference method
    This is more accurate than forward difference for most functions
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def forward_difference(f, x, h):
    """Forward difference approximation"""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    """Backward difference approximation"""
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    """Central difference approximation (most accurate)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Symbolic derivative using SymPy
x = sp.Symbol('x')
f_sym = x**2
f_prime = sp.diff(f_sym, x)
print(f"Symbolic derivative of x²: {f_prime}")

# Compare different numerical methods
x_test = 2.0
h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

print("\nNumerical Derivative Comparison at x = 2:")
print("h\t\tForward\t\tBackward\tCentral\t\tExact")
print("-" * 70)

for h in h_values:
    forward = forward_difference(f, x_test, h)
    backward = backward_difference(f, x_test, h)
    central = central_difference(f, x_test, h)
    exact = 2 * x_test  # f'(x) = 2x
    
    print(f"{h:.1e}\t{forward:.6f}\t{backward:.6f}\t{central:.6f}\t{exact:.6f}")

# Visualize the derivative concept
x_vals = np.linspace(-2, 2, 100)
y_vals = [f(x) for x in x_vals]

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Original function
ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Original Function')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Tangent line at x = 1
x_tangent = 1.0
y_tangent = f(x_tangent)
slope = 2 * x_tangent  # f'(x) = 2x
tangent_x = np.linspace(x_tangent - 0.5, x_tangent + 0.5, 100)
tangent_y = y_tangent + slope * (tangent_x - x_tangent)

ax2.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
ax2.plot(tangent_x, tangent_y, 'r--', linewidth=2, label=f'Tangent at x=1 (slope={slope})')
ax2.scatter(x_tangent, y_tangent, color='red', s=100, zorder=5)
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('Tangent Line')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Secant lines approaching tangent
x_point = 1.0
h_values_vis = [0.5, 0.2, 0.1]
colors = ['green', 'orange', 'purple']

ax3.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
ax3.scatter(x_point, f(x_point), color='red', s=100, zorder=5, label='Point of interest')

for i, h in enumerate(h_values_vis):
    x1, x2 = x_point, x_point + h
    y1, y2 = f(x1), f(x2)
    slope_secant = (y2 - y1) / (x2 - x1)
    
    # Plot secant line
    secant_x = np.linspace(x1 - 0.3, x2 + 0.3, 100)
    secant_y = y1 + slope_secant * (secant_x - x1)
    ax3.plot(secant_x, secant_y, '--', color=colors[i], linewidth=2, 
             label=f'Secant h={h} (slope={slope_secant:.2f})')

ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.set_title('Secant Lines Approaching Tangent')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Derivatives comparison
numerical_derivatives = [numerical_derivative_detailed(f, x) for x in x_vals]
symbolic_derivatives = [2*x for x in x_vals]  # f'(x) = 2x

ax4.plot(x_vals, numerical_derivatives, 'r-', linewidth=2, label='Numerical f\'(x)')
ax4.plot(x_vals, symbolic_derivatives, 'g--', linewidth=2, label='Symbolic f\'(x) = 2x')
ax4.set_xlabel('x')
ax4.set_ylabel('f\'(x)')
ax4.set_title('Derivatives Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Error analysis
print("\nError Analysis:")
print("Comparing numerical vs symbolic derivatives:")
x_test_points = [0.5, 1.0, 1.5, 2.0]
h_optimal = 1e-7

for x_test in x_test_points:
    numerical_val = numerical_derivative_detailed(f, x_test, h_optimal)
    symbolic_val = 2 * x_test
    error = abs(numerical_val - symbolic_val)
    print(f"x = {x_test}: numerical = {numerical_val:.10f}, symbolic = {symbolic_val:.10f}, error = {error:.2e}")
```

### Mathematical Insight: Why the Central Difference is Better

The central difference formula $\frac{f(x+h) - f(x-h)}{2h}$ is generally more accurate than the forward difference $\frac{f(x+h) - f(x)}{h}$ because:

1. **Taylor Series Analysis**: The central difference eliminates the first-order error term
2. **Symmetry**: It uses points equally spaced on both sides of x
3. **Error Reduction**: The truncation error is O(h²) instead of O(h)

### Applications in Machine Learning

Understanding derivatives is crucial for:
- **Gradient Descent**: Finding the direction of steepest descent
- **Backpropagation**: Computing gradients through neural networks
- **Optimization**: Locating minima of loss functions
- **Sensitivity Analysis**: Understanding model behavior

## 2.2 Basic Differentiation Rules

Understanding differentiation rules is essential for computing derivatives efficiently. These rules form the foundation for automatic differentiation systems used in modern machine learning frameworks.

### Fundamental Rules

The basic differentiation rules provide systematic methods for computing derivatives of common function combinations:

1. **Power Rule**: $\frac{d}{dx}(x^n) = nx^{n-1}$
2. **Constant Rule**: $\frac{d}{dx}(c) = 0$
3. **Constant Multiple Rule**: $\frac{d}{dx}(cf(x)) = c\frac{d}{dx}f(x)$
4. **Sum Rule**: $\frac{d}{dx}(f(x) + g(x)) = \frac{d}{dx}f(x) + \frac{d}{dx}g(x)$
5. **Product Rule**: $\frac{d}{dx}(f(x)g(x)) = f(x)\frac{d}{dx}g(x) + g(x)\frac{d}{dx}f(x)$
6. **Quotient Rule**: $\frac{d}{dx}(\frac{f(x)}{g(x)}) = \frac{g(x)\frac{d}{dx}f(x) - f(x)\frac{d}{dx}g(x)}{g(x)^2}$

### Mathematical Justification

These rules can be derived from the limit definition of the derivative. For example, the product rule follows from:

$$\frac{d}{dx}(f(x)g(x)) = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

By adding and subtracting $f(x+h)g(x)$ in the numerator and using the limit properties, we obtain the product rule.

```python
# Comprehensive demonstration of differentiation rules
def demonstrate_rules_comprehensive():
    x = sp.Symbol('x')
    
    print("=== BASIC DIFFERENTIATION RULES ===\n")
    
    # Power rule: d/dx(x^n) = n*x^(n-1)
    print("1. POWER RULE")
    power_examples = [x**2, x**3, x**0.5, x**(-1), x**(-2)]
    for expr in power_examples:
        deriv = sp.diff(expr, x)
        print(f"   d/dx({expr}) = {deriv}")
    
    # Constant rule: d/dx(c) = 0
    print("\n2. CONSTANT RULE")
    const_expr = 5
    const_deriv = sp.diff(const_expr, x)
    print(f"   d/dx({const_expr}) = {const_deriv}")
    
    # Constant multiple rule: d/dx(cf(x)) = c*d/dx(f(x))
    print("\n3. CONSTANT MULTIPLE RULE")
    const_mult_expr = 3 * x**2
    const_mult_deriv = sp.diff(const_mult_expr, x)
    print(f"   d/dx({const_mult_expr}) = {const_mult_deriv}")
    
    # Sum rule: d/dx(f + g) = d/dx(f) + d/dx(g)
    print("\n4. SUM RULE")
    sum_expr = x**2 + 3*x + 1
    sum_deriv = sp.diff(sum_expr, x)
    print(f"   d/dx({sum_expr}) = {sum_deriv}")
    
    # Product rule: d/dx(f*g) = f*dg + g*df
    print("\n5. PRODUCT RULE")
    product_expr = x**2 * sp.sin(x)
    product_deriv = sp.diff(product_expr, x)
    print(f"   d/dx({product_expr}) = {product_deriv}")
    
    # Quotient rule: d/dx(f/g) = (g*df - f*dg)/g²
    print("\n6. QUOTIENT RULE")
    quotient_expr = x**2 / (x + 1)
    quotient_deriv = sp.diff(quotient_expr, x)
    print(f"   d/dx({quotient_expr}) = {quotient_deriv}")
    
    # Chain rule preview
    print("\n7. CHAIN RULE PREVIEW")
    chain_expr = sp.sin(x**2)
    chain_deriv = sp.diff(chain_expr, x)
    print(f"   d/dx({chain_expr}) = {chain_deriv}")

demonstrate_rules_comprehensive()

# Visualize the rules with practical examples
def visualize_differentiation_rules():
    x_vals = np.linspace(-2, 2, 1000)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Power rule visualization
    y1 = x_vals**2
    dy1 = 2 * x_vals
    
    ax1.plot(x_vals, y1, 'b-', linewidth=2, label='f(x) = x²')
    ax1.plot(x_vals, dy1, 'r--', linewidth=2, label='f\'(x) = 2x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Power Rule: d/dx(x²) = 2x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Product rule visualization
    y2 = x_vals**2 * np.sin(x_vals)
    dy2 = 2 * x_vals * np.sin(x_vals) + x_vals**2 * np.cos(x_vals)
    
    ax2.plot(x_vals, y2, 'b-', linewidth=2, label='f(x) = x²sin(x)')
    ax2.plot(x_vals, dy2, 'r--', linewidth=2, label='f\'(x) = 2xsin(x) + x²cos(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Product Rule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Quotient rule visualization
    y3 = x_vals**2 / (x_vals + 1)
    dy3 = (2 * x_vals * (x_vals + 1) - x_vals**2) / (x_vals + 1)**2
    
    # Handle division by zero
    mask = np.abs(x_vals + 1) > 1e-10
    ax3.plot(x_vals[mask], y3[mask], 'b-', linewidth=2, label='f(x) = x²/(x+1)')
    ax3.plot(x_vals[mask], dy3[mask], 'r--', linewidth=2, label='f\'(x)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Quotient Rule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sum rule visualization
    y4 = x_vals**2 + 3*x_vals + 1
    dy4 = 2*x_vals + 3
    
    ax4.plot(x_vals, y4, 'b-', linewidth=2, label='f(x) = x² + 3x + 1')
    ax4.plot(x_vals, dy4, 'r--', linewidth=2, label='f\'(x) = 2x + 3')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Sum Rule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_differentiation_rules()

# Advanced examples with error analysis
def advanced_differentiation_examples():
    x = sp.Symbol('x')
    
    print("\n=== ADVANCED DIFFERENTIATION EXAMPLES ===\n")
    
    # Exponential and logarithmic functions
    print("1. EXPONENTIAL AND LOGARITHMIC FUNCTIONS")
    exp_expr = sp.exp(x)
    log_expr = sp.log(x)
    print(f"   d/dx(e^x) = {sp.diff(exp_expr, x)}")
    print(f"   d/dx(ln(x)) = {sp.diff(log_expr, x)}")
    
    # Trigonometric functions
    print("\n2. TRIGONOMETRIC FUNCTIONS")
    trig_expr = sp.sin(x)
    trig_expr2 = sp.cos(x)
    trig_expr3 = sp.tan(x)
    print(f"   d/dx(sin(x)) = {sp.diff(trig_expr, x)}")
    print(f"   d/dx(cos(x)) = {sp.diff(trig_expr2, x)}")
    print(f"   d/dx(tan(x)) = {sp.diff(trig_expr3, x)}")
    
    # Complex combinations
    print("\n3. COMPLEX COMBINATIONS")
    complex_expr = sp.exp(x**2) * sp.sin(x)
    complex_deriv = sp.diff(complex_expr, x)
    print(f"   d/dx(e^(x²) * sin(x)) = {complex_deriv}")
    
    # Implicit differentiation example
    print("\n4. IMPLICIT DIFFERENTIATION")
    y = sp.Symbol('y')
    implicit_expr = x**2 + y**2 - 1  # Circle: x² + y² = 1
    # Solve for dy/dx: 2x + 2y*dy/dx = 0
    dy_dx = -x/y
    print(f"   For x² + y² = 1: dy/dx = {dy_dx}")

advanced_differentiation_examples()

# Numerical verification of rules
def numerical_verification():
    print("\n=== NUMERICAL VERIFICATION ===\n")
    
    def f1(x): return x**2  # Power rule
    def f2(x): return x**2 * np.sin(x)  # Product rule
    def f3(x): return x**2 / (x + 1)  # Quotient rule
    
    x_test = 1.5
    h = 1e-7
    
    # Test power rule
    numerical_power = numerical_derivative_detailed(f1, x_test, h)
    symbolic_power = 2 * x_test
    print(f"Power rule at x = {x_test}:")
    print(f"  Numerical: {numerical_power:.6f}")
    print(f"  Symbolic:  {symbolic_power:.6f}")
    print(f"  Error:     {abs(numerical_power - symbolic_power):.2e}")
    
    # Test product rule
    numerical_product = numerical_derivative_detailed(f2, x_test, h)
    symbolic_product = 2 * x_test * np.sin(x_test) + x_test**2 * np.cos(x_test)
    print(f"\nProduct rule at x = {x_test}:")
    print(f"  Numerical: {numerical_product:.6f}")
    print(f"  Symbolic:  {symbolic_product:.6f}")
    print(f"  Error:     {abs(numerical_product - symbolic_product):.2e}")

numerical_verification()

### Applications in Machine Learning

These differentiation rules are fundamental to:

1. **Automatic Differentiation**: Modern frameworks like TensorFlow and PyTorch use these rules to compute gradients automatically
2. **Loss Function Derivatives**: Computing gradients of complex loss functions
3. **Activation Function Derivatives**: Derivatives of sigmoid, ReLU, tanh, etc.
4. **Optimization Algorithms**: All gradient-based optimization methods rely on these rules

## 2.3 Chain Rule

The chain rule is one of the most important differentiation rules, especially in machine learning. It allows us to compute derivatives of composite functions, which are ubiquitous in neural networks and other complex models.

### Mathematical Foundation

The chain rule states that if y = f(u) and u = g(x), then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)$$

This can be extended to longer chains: if y = f(g(h(x))), then:

$$\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

### Why the Chain Rule is Crucial in ML

1. **Neural Networks**: Each layer applies a function to the output of the previous layer
2. **Activation Functions**: Functions like sigmoid, tanh, ReLU are applied to linear combinations
3. **Loss Functions**: Often involve multiple nested functions
4. **Backpropagation**: The entire algorithm is based on the chain rule

### Intuitive Understanding

Think of the chain rule as "how much does the final output change when I change the input?" This involves:
- How much the intermediate variable changes with respect to the input
- How much the final output changes with respect to the intermediate variable

```python
# Comprehensive chain rule demonstration
def chain_rule_example_comprehensive():
    x = sp.Symbol('x')
    
    print("=== CHAIN RULE EXAMPLES ===\n")
    
    # Basic chain rule examples
    print("1. BASIC CHAIN RULE EXAMPLES")
    
    # f(x) = sin(x²)
    f_expr1 = sp.sin(x**2)
    f_deriv1 = sp.diff(f_expr1, x)
    print(f"   f(x) = sin(x²)")
    print(f"   f'(x) = {f_deriv1}")
    print(f"   Explanation: d/dx(sin(x²)) = cos(x²) * d/dx(x²) = cos(x²) * 2x")
    
    # f(x) = e^(x²)
    f_expr2 = sp.exp(x**2)
    f_deriv2 = sp.diff(f_expr2, x)
    print(f"\n   f(x) = e^(x²)")
    print(f"   f'(x) = {f_deriv2}")
    print(f"   Explanation: d/dx(e^(x²)) = e^(x²) * d/dx(x²) = e^(x²) * 2x")
    
    # f(x) = ln(x² + 1)
    f_expr3 = sp.log(x**2 + 1)
    f_deriv3 = sp.diff(f_expr3, x)
    print(f"\n   f(x) = ln(x² + 1)")
    print(f"   f'(x) = {f_deriv3}")
    print(f"   Explanation: d/dx(ln(x² + 1)) = 1/(x² + 1) * d/dx(x² + 1) = 2x/(x² + 1)")
    
    # Multiple chain rule applications
    print("\n2. MULTIPLE CHAIN RULE APPLICATIONS")
    
    # f(x) = sin(e^(x²))
    f_expr4 = sp.sin(sp.exp(x**2))
    f_deriv4 = sp.diff(f_expr4, x)
    print(f"   f(x) = sin(e^(x²))")
    print(f"   f'(x) = {f_deriv4}")
    
    # f(x) = e^(sin(x²))
    f_expr5 = sp.exp(sp.sin(x**2))
    f_deriv5 = sp.diff(f_expr5, x)
    print(f"\n   f(x) = e^(sin(x²))")
    print(f"   f'(x) = {f_deriv5}")
    
    # Practical ML examples
    print("\n3. MACHINE LEARNING EXAMPLES")
    
    # Sigmoid function: σ(x) = 1/(1 + e^(-x))
    sigmoid_expr = 1 / (1 + sp.exp(-x))
    sigmoid_deriv = sp.diff(sigmoid_expr, x)
    print(f"   Sigmoid: σ(x) = 1/(1 + e^(-x))")
    print(f"   σ'(x) = {sigmoid_deriv}")
    print(f"   Simplified: σ'(x) = σ(x)(1 - σ(x))")
    
    # Tanh function: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
    tanh_expr = sp.tanh(x)
    tanh_deriv = sp.diff(tanh_expr, x)
    print(f"\n   Tanh: tanh(x)")
    print(f"   tanh'(x) = {tanh_deriv}")
    print(f"   Simplified: tanh'(x) = 1 - tanh²(x)")

chain_rule_example_comprehensive()

# Visualize chain rule with detailed analysis
def visualize_chain_rule():
    x_vals = np.linspace(-2, 2, 1000)
    
    # Example: f(x) = sin(x²)
    def f(x): return np.sin(x**2)
    def f_prime(x): return 2 * x * np.cos(x**2)
    
    # Break down the chain: u = x², f = sin(u)
    def u(x): return x**2
    def u_prime(x): return 2 * x
    def f_of_u(u_val): return np.sin(u_val)
    def f_of_u_prime(u_val): return np.cos(u_val)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original function and its derivative
    ax1.plot(x_vals, f(x_vals), 'b-', linewidth=2, label='f(x) = sin(x²)')
    ax1.plot(x_vals, f_prime(x_vals), 'r--', linewidth=2, label='f\'(x) = 2x*cos(x²)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Composite Function and Its Derivative')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Inner function u(x) = x²
    ax2.plot(x_vals, u(x_vals), 'g-', linewidth=2, label='u(x) = x²')
    ax2.plot(x_vals, u_prime(x_vals), 'g--', linewidth=2, label='u\'(x) = 2x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.set_title('Inner Function u(x) = x²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Outer function f(u) = sin(u)
    u_vals = np.linspace(0, 4, 1000)  # u = x² ranges from 0 to 4 for x in [-2, 2]
    ax3.plot(u_vals, f_of_u(u_vals), 'm-', linewidth=2, label='f(u) = sin(u)')
    ax3.plot(u_vals, f_of_u_prime(u_vals), 'm--', linewidth=2, label='f\'(u) = cos(u)')
    ax3.set_xlabel('u')
    ax3.set_ylabel('f(u)')
    ax3.set_title('Outer Function f(u) = sin(u)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Chain rule verification
    x_test = np.linspace(-2, 2, 100)
    chain_rule_result = f_of_u_prime(u(x_test)) * u_prime(x_test)
    direct_derivative = f_prime(x_test)
    
    ax4.plot(x_test, chain_rule_result, 'b-', linewidth=2, label='Chain rule: f\'(u)*u\'(x)')
    ax4.plot(x_test, direct_derivative, 'r--', linewidth=2, label='Direct derivative')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Derivative')
    ax4.set_title('Chain Rule Verification')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\nChain Rule Numerical Verification:")
    x_test_point = 1.0
    h = 1e-7
    
    # Direct derivative
    direct_deriv = numerical_derivative_detailed(f, x_test_point, h)
    
    # Chain rule calculation
    u_val = u(x_test_point)
    du_dx = u_prime(x_test_point)
    df_du = f_of_u_prime(u_val)
    chain_rule_deriv = df_du * du_dx
    
    print(f"At x = {x_test_point}:")
    print(f"  Direct derivative: {direct_deriv:.6f}")
    print(f"  Chain rule: f'(u)*u'(x) = {df_du:.6f} * {du_dx:.6f} = {chain_rule_deriv:.6f}")
    print(f"  Error: {abs(direct_deriv - chain_rule_deriv):.2e}")

visualize_chain_rule()

# Backpropagation example using chain rule
def backpropagation_example():
    print("\n=== BACKPROPAGATION EXAMPLE ===\n")
    
    # Simple neural network: y = σ(w*x + b)
    # where σ is the sigmoid function
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # Forward pass
    def forward_pass(x, w, b):
        z = w * x + b  # Linear combination
        y = sigmoid(z)  # Activation function
        return y, z
    
    # Backward pass using chain rule
    def backward_pass(x, w, b, target):
        y, z = forward_pass(x, w, b)
        
        # Loss: L = (y - target)²
        loss = (y - target)**2
        
        # Chain rule for gradients:
        # dL/dw = dL/dy * dy/dz * dz/dw
        # dL/db = dL/dy * dy/dz * dz/db
        
        dL_dy = 2 * (y - target)  # dL/dy
        dy_dz = sigmoid_derivative(z)  # dy/dz
        dz_dw = x  # dz/dw
        dz_db = 1  # dz/db
        
        dL_dw = dL_dy * dy_dz * dz_dw
        dL_db = dL_dy * dy_dz * dz_db
        
        return dL_dw, dL_db, loss
    
    # Test the backpropagation
    x = 2.0
    w = 1.0
    b = 0.5
    target = 0.8
    
    print(f"Input: x = {x}")
    print(f"Weight: w = {w}")
    print(f"Bias: b = {b}")
    print(f"Target: {target}")
    
    y, z = forward_pass(x, w, b)
    print(f"Forward pass: z = {z:.4f}, y = {y:.4f}")
    
    dL_dw, dL_db, loss = backward_pass(x, w, b, target)
    print(f"Loss: {loss:.4f}")
    print(f"Gradients: dL/dw = {dL_dw:.4f}, dL/db = {dL_db:.4f}")
    
    # Verify with numerical gradients
    def loss_function(w, b):
        y, _ = forward_pass(x, w, b)
        return (y - target)**2
    
    h = 1e-7
    numerical_dw = (loss_function(w + h, b) - loss_function(w - h, b)) / (2 * h)
    numerical_db = (loss_function(w, b + h) - loss_function(w, b - h)) / (2 * h)
    
    print(f"\nNumerical verification:")
    print(f"  Numerical dL/dw: {numerical_dw:.4f}")
    print(f"  Analytical dL/dw: {dL_dw:.4f}")
    print(f"  Error: {abs(numerical_dw - dL_dw):.2e}")
    
    print(f"  Numerical dL/db: {numerical_db:.4f}")
    print(f"  Analytical dL/db: {dL_db:.4f}")
    print(f"  Error: {abs(numerical_db - dL_db):.2e}")

backpropagation_example()

# Advanced chain rule examples
def advanced_chain_rule_examples():
    x = sp.Symbol('x')
    
    print("\n=== ADVANCED CHAIN RULE EXAMPLES ===\n")
    
    # Multiple nested functions
    print("1. MULTIPLE NESTED FUNCTIONS")
    complex_expr = sp.sin(sp.exp(sp.log(x**2 + 1)))
    complex_deriv = sp.diff(complex_expr, x)
    print(f"   f(x) = sin(e^(ln(x² + 1)))")
    print(f"   f'(x) = {complex_deriv}")
    
    # Parametric functions
    print("\n2. PARAMETRIC FUNCTIONS")
    t = sp.Symbol('t')
    x_param = sp.cos(t)
    y_param = sp.sin(t)
    
    # dy/dx = (dy/dt)/(dx/dt)
    dy_dt = sp.diff(y_param, t)
    dx_dt = sp.diff(x_param, t)
    dy_dx = dy_dt / dx_dt
    print(f"   x = cos(t), y = sin(t)")
    print(f"   dy/dx = {dy_dx}")
    
    # Implicit differentiation with chain rule
    print("\n3. IMPLICIT DIFFERENTIATION")
    y = sp.Symbol('y')
    implicit_expr = x**2 + y**2 - 1
    # Differentiate both sides with respect to x
    # 2x + 2y*dy/dx = 0
    # dy/dx = -x/y
    print(f"   For x² + y² = 1:")
    print(f"   dy/dx = -x/y (using chain rule on y²)")

advanced_chain_rule_examples()

### Applications in Machine Learning

The chain rule is fundamental to:

1. **Backpropagation**: The core algorithm for training neural networks
2. **Automatic Differentiation**: Modern frameworks compute gradients using the chain rule
3. **Activation Functions**: Derivatives of sigmoid, tanh, ReLU, etc.
4. **Loss Functions**: Gradients of complex loss functions
5. **Optimization**: All gradient-based optimization methods

## 2.4 Partial Derivatives

Partial derivatives are essential for functions of multiple variables, which are ubiquitous in machine learning. They measure how a function changes with respect to one variable while holding all others constant.

### Mathematical Foundation

For a function f(x₁, x₂, ..., xₙ), the partial derivative with respect to xᵢ is:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_i, ..., x_n)}{h}$$

This represents the rate of change of f in the direction of the i-th variable.

### Why Partial Derivatives Matter in ML

1. **Multivariable Functions**: Most ML models have multiple parameters
2. **Gradient Computation**: Gradients are vectors of partial derivatives
3. **Optimization**: Gradient descent requires partial derivatives
4. **Feature Sensitivity**: Understanding how each feature affects the output

### Geometric Interpretation

- **Directional Sensitivity**: How much the function changes in a specific direction
- **Tangent Planes**: Partial derivatives define the tangent plane to a surface
- **Gradient Vector**: The vector of all partial derivatives points in the direction of steepest ascent

```python
# Comprehensive partial derivatives demonstration
def partial_derivatives_comprehensive():
    x, y = sp.symbols('x y')
    
    print("=== PARTIAL DERIVATIVES EXAMPLES ===\n")
    
    # Basic examples
    print("1. BASIC EXAMPLES")
    
    # f(x,y) = x² + y²
    f_expr1 = x**2 + y**2
    df_dx1 = sp.diff(f_expr1, x)
    df_dy1 = sp.diff(f_expr1, y)
    
    print(f"   f(x,y) = x² + y²")
    print(f"   ∂f/∂x = {df_dx1}")
    print(f"   ∂f/∂y = {df_dy1}")
    print(f"   Gradient ∇f = [{df_dx1}, {df_dy1}]")
    
    # f(x,y) = x*y + sin(x)
    f_expr2 = x*y + sp.sin(x)
    df_dx2 = sp.diff(f_expr2, x)
    df_dy2 = sp.diff(f_expr2, y)
    
    print(f"\n   f(x,y) = x*y + sin(x)")
    print(f"   ∂f/∂x = {df_dx2}")
    print(f"   ∂f/∂y = {df_dy2}")
    print(f"   Gradient ∇f = [{df_dx2}, {df_dy2}]")
    
    # More complex examples
    print("\n2. COMPLEX EXAMPLES")
    
    # f(x,y) = e^(x² + y²)
    f_expr3 = sp.exp(x**2 + y**2)
    df_dx3 = sp.diff(f_expr3, x)
    df_dy3 = sp.diff(f_expr3, y)
    
    print(f"   f(x,y) = e^(x² + y²)")
    print(f"   ∂f/∂x = {df_dx3}")
    print(f"   ∂f/∂y = {df_dy3}")
    
    # f(x,y) = sin(x*y) / (x² + y²)
    f_expr4 = sp.sin(x*y) / (x**2 + y**2)
    df_dx4 = sp.diff(f_expr4, x)
    df_dy4 = sp.diff(f_expr4, y)
    
    print(f"\n   f(x,y) = sin(x*y) / (x² + y²)")
    print(f"   ∂f/∂x = {df_dx4}")
    print(f"   ∂f/∂y = {df_dy4}")
    
    # ML-specific examples
    print("\n3. MACHINE LEARNING EXAMPLES")
    
    # Linear regression: f(w,b) = (wx + b - y)²
    w, b = sp.symbols('w b')
    x_data, y_data = sp.symbols('x_data y_data')
    loss_expr = (w * x_data + b - y_data)**2
    
    df_dw = sp.diff(loss_expr, w)
    df_db = sp.diff(loss_expr, b)
    
    print(f"   Loss function: L(w,b) = (wx + b - y)²")
    print(f"   ∂L/∂w = {df_dw}")
    print(f"   ∂L/∂b = {df_db}")
    
    # Logistic regression: f(w,b) = -y*log(σ(wx + b)) - (1-y)*log(1-σ(wx + b))
    sigma_expr = 1 / (1 + sp.exp(-(w * x_data + b)))
    log_loss_expr = -y_data * sp.log(sigma_expr) - (1 - y_data) * sp.log(1 - sigma_expr)
    
    df_dw_log = sp.diff(log_loss_expr, w)
    df_db_log = sp.diff(log_loss_expr, b)
    
    print(f"\n   Logistic loss: L(w,b) = -y*log(σ(wx + b)) - (1-y)*log(1-σ(wx + b))")
    print(f"   ∂L/∂w = {df_dw_log}")
    print(f"   ∂L/∂b = {df_db_log}")

partial_derivatives_comprehensive()

# Visualize partial derivatives
def visualize_partial_derivatives():
    from mpl_toolkits.mplot3d import Axes3D
    
    def f_3d(x, y):
        return x**2 + y**2
    
    def df_dx(x, y):
        return 2 * x
    
    def df_dy(x, y):
        return 2 * y
    
    # Create 3D surface
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = f_3d(X, Y)
    
    fig = plt.figure(figsize=(20, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Surface: f(x,y) = x² + y²')
    
    # Contour plot with gradient vectors
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contour(X, Y, Z, levels=10)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Add gradient vectors
    skip = 5
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               df_dx(X[::skip, ::skip], Y[::skip, ::skip]),
               df_dy(X[::skip, ::skip], Y[::skip, ::skip]),
               color='red', alpha=0.6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Gradient Vectors')
    ax2.grid(True, alpha=0.3)
    
    # ∂f/∂x as a function of x (y fixed)
    ax3 = fig.add_subplot(2, 3, 3)
    y_fixed = 1.0
    x_vals = np.linspace(-2, 2, 100)
    df_dx_vals = df_dx(x_vals, y_fixed)
    ax3.plot(x_vals, df_dx_vals, 'b-', linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('∂f/∂x')
    ax3.set_title(f'∂f/∂x (y = {y_fixed})')
    ax3.grid(True, alpha=0.3)
    
    # ∂f/∂y as a function of y (x fixed)
    ax4 = fig.add_subplot(2, 3, 4)
    x_fixed = 1.0
    y_vals = np.linspace(-2, 2, 100)
    df_dy_vals = df_dy(x_fixed, y_vals)
    ax4.plot(y_vals, df_dy_vals, 'r-', linewidth=2)
    ax4.set_xlabel('y')
    ax4.set_ylabel('∂f/∂y')
    ax4.set_title(f'∂f/∂y (x = {x_fixed})')
    ax4.grid(True, alpha=0.3)
    
    # Gradient magnitude
    ax5 = fig.add_subplot(2, 3, 5)
    grad_magnitude = np.sqrt(df_dx(X, Y)**2 + df_dy(X, Y)**2)
    grad_contour = ax5.contour(X, Y, grad_magnitude, levels=10)
    ax5.clabel(grad_contour, inline=True, fontsize=8)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Gradient Magnitude |∇f|')
    ax5.grid(True, alpha=0.3)
    
    # Direction of steepest ascent
    ax6 = fig.add_subplot(2, 3, 6)
    # Normalize gradient vectors
    grad_norm = np.sqrt(df_dx(X, Y)**2 + df_dy(X, Y)**2)
    df_dx_norm = df_dx(X, Y) / (grad_norm + 1e-10)
    df_dy_norm = df_dy(X, Y) / (grad_norm + 1e-10)
    
    ax6.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               df_dx_norm[::skip, ::skip], df_dy_norm[::skip, ::skip],
               color='purple', alpha=0.6)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Direction of Steepest Ascent')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_partial_derivatives()

# Numerical computation of partial derivatives
def numerical_partial_derivatives():
    def f(x, y):
        return x**2 + y**2
    
    def numerical_partial_x(f, x, y, h=1e-7):
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    
    def numerical_partial_y(f, x, y, h=1e-7):
        return (f(x, y + h) - f(x, y - h)) / (2 * h)
    
    print("\n=== NUMERICAL PARTIAL DERIVATIVES ===\n")
    
    test_points = [(0, 0), (1, 1), (-1, 2), (0.5, -0.5)]
    
    for x_test, y_test in test_points:
        # Numerical derivatives
        num_df_dx = numerical_partial_x(f, x_test, y_test)
        num_df_dy = numerical_partial_y(f, x_test, y_test)
        
        # Analytical derivatives
        ana_df_dx = 2 * x_test
        ana_df_dy = 2 * y_test
        
        print(f"At point ({x_test}, {y_test}):")
        print(f"  ∂f/∂x: numerical = {num_df_dx:.6f}, analytical = {ana_df_dx:.6f}")
        print(f"  ∂f/∂y: numerical = {num_df_dy:.6f}, analytical = {ana_df_dy:.6f}")
        print(f"  Errors: dx = {abs(num_df_dx - ana_df_dx):.2e}, dy = {abs(num_df_dy - ana_df_dy):.2e}")
        print()

numerical_partial_derivatives()

# Gradient descent with partial derivatives
def gradient_descent_2d():
    print("\n=== GRADIENT DESCENT IN 2D ===\n")
    
    def f(x, y):
        return x**2 + y**2
    
    def df_dx(x, y):
        return 2 * x
    
    def df_dy(x, y):
        return 2 * y
    
    def gradient_descent_2d_algorithm(f, df_dx, df_dy, x0, y0, learning_rate=0.1, iterations=50):
        x, y = x0, y0
        history = [(x, y)]
        
        for i in range(iterations):
            # Compute gradients
            grad_x = df_dx(x, y)
            grad_y = df_dy(x, y)
            
            # Update parameters
            x = x - learning_rate * grad_x
            y = y - learning_rate * grad_y
            
            history.append((x, y))
        
        return history
    
    # Run gradient descent
    x0, y0 = 2.0, 1.5
    history = gradient_descent_2d_algorithm(f, df_dx, df_dy, x0, y0, learning_rate=0.1, iterations=20)
    
    # Extract coordinates for plotting
    x_coords = [point[0] for point in history]
    y_coords = [point[1] for point in history]
    z_coords = [point[2] for point in history]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contour plot with optimization path
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    ax1.contour(X, Y, Z, levels=20)
    ax1.plot(x_coords, y_coords, 'r-o', linewidth=2, markersize=4, label='Optimization path')
    ax1.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
    ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Descent Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Function value over iterations
    ax2.plot(range(len(z_coords)), z_coords, 'b-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x,y)')
    ax2.set_title('Function Value vs Iteration')
    ax2.grid(True, alpha=0.3)
    
    # x and y coordinates over iterations
    ax3.plot(range(len(x_coords)), x_coords, 'g-o', linewidth=2, markersize=4, label='x')
    ax3.plot(range(len(y_coords)), y_coords, 'm-o', linewidth=2, markersize=4, label='y')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Coordinate Value')
    ax3.set_title('Parameter Values vs Iteration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gradient magnitude over iterations
    grad_magnitudes = []
    for i, (x_val, y_val) in enumerate(zip(x_coords, y_coords)):
        grad_mag = np.sqrt(df_dx(x_val, y_val)**2 + df_dy(x_val, y_val)**2)
        grad_magnitudes.append(grad_mag)
    
    ax4.plot(range(len(grad_magnitudes)), grad_magnitudes, 'c-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('|∇f|')
    ax4.set_title('Gradient Magnitude vs Iteration')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Starting point: ({x0}, {y0})")
    print(f"Final point: ({x_coords[-1]:.6f}, {y_coords[-1]:.6f})")
    print(f"Final function value: {z_coords[-1]:.6f}")
    print(f"Final gradient magnitude: {grad_magnitudes[-1]:.6f}")

gradient_descent_2d()

# Higher-order partial derivatives
def higher_order_partials():
    x, y = sp.symbols('x y')
    
    print("\n=== HIGHER-ORDER PARTIAL DERIVATIVES ===\n")
    
    # f(x,y) = x³ + y³ + x*y
    f_expr = x**3 + y**3 + x*y
    
    # First-order partials
    df_dx = sp.diff(f_expr, x)
    df_dy = sp.diff(f_expr, y)
    
    # Second-order partials
    d2f_dx2 = sp.diff(df_dx, x)
    d2f_dy2 = sp.diff(df_dy, y)
    d2f_dxdy = sp.diff(df_dx, y)
    d2f_dydx = sp.diff(df_dy, x)
    
    print(f"Function: f(x,y) = {f_expr}")
    print(f"\nFirst-order partials:")
    print(f"  ∂f/∂x = {df_dx}")
    print(f"  ∂f/∂y = {df_dy}")
    
    print(f"\nSecond-order partials:")
    print(f"  ∂²f/∂x² = {d2f_dx2}")
    print(f"  ∂²f/∂y² = {d2f_dy2}")
    print(f"  ∂²f/∂x∂y = {d2f_dxdy}")
    print(f"  ∂²f/∂y∂x = {d2f_dydx}")
    
    # Verify Clairaut's theorem (equality of mixed partials)
    print(f"\nClairaut's theorem verification:")
    print(f"  ∂²f/∂x∂y = ∂²f/∂y∂x: {d2f_dxdy == d2f_dydx}")
    
    # Hessian matrix
    print(f"\nHessian matrix:")
    print(f"  H = [∂²f/∂x²    ∂²f/∂x∂y]")
    print(f"      [∂²f/∂y∂x   ∂²f/∂y²]")
    print(f"  H = [{d2f_dx2}    {d2f_dxdy}]")
    print(f"      [{d2f_dydx}   {d2f_dy2}]")

higher_order_partials()

### Applications in Machine Learning

Partial derivatives are fundamental to:

1. **Gradient Computation**: Computing gradients for optimization
2. **Neural Networks**: Backpropagation through multiple layers
3. **Feature Importance**: Understanding how each feature affects predictions
4. **Optimization**: Gradient descent, Newton's method, etc.
5. **Sensitivity Analysis**: Understanding model behavior

## 2.5 Gradient and Directional Derivatives

The gradient is a vector that contains all the partial derivatives of a function. It points in the direction of steepest ascent and is fundamental to optimization algorithms.

### Mathematical Foundation

For a function f(x₁, x₂, ..., xₙ), the gradient is:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]$$

The directional derivative in the direction of unit vector **u** is:

$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$$

### Key Properties of the Gradient

1. **Direction of Steepest Ascent**: ∇f points in the direction of maximum increase
2. **Magnitude**: |∇f| gives the rate of change in the direction of steepest ascent
3. **Orthogonality**: ∇f is perpendicular to level curves/surfaces
4. **Linearity**: ∇(af + bg) = a∇f + b∇g

### Why Gradients Matter in ML

1. **Optimization**: Gradient descent follows the negative gradient
2. **Feature Importance**: Gradient magnitude indicates sensitivity
3. **Convergence**: Gradient magnitude helps determine convergence
4. **Regularization**: Gradient-based regularization methods

```python
# Comprehensive gradient analysis
def gradient_analysis_comprehensive():
    print("=== GRADIENT ANALYSIS ===\n")
    
    def f(x, y):
        return x**2 + y**2
    
    def gradient_f(x, y):
        return np.array([2*x, 2*y])
    
    def gradient_magnitude(x, y):
        return np.sqrt((2*x)**2 + (2*y)**2)
    
    # Test points
    test_points = [(0, 0), (1, 1), (-1, 0), (0.5, -0.5)]
    
    print("1. GRADIENT CALCULATION")
    for point in test_points:
        x, y = point
        grad = gradient_f(x, y)
        mag = gradient_magnitude(x, y)
        print(f"   Point ({x}, {y}):")
        print(f"     Gradient: [{grad[0]:.4f}, {grad[1]:.4f}]")
        print(f"     Magnitude: {mag:.4f}")
        print(f"     Unit direction: [{grad[0]/mag:.4f}, {grad[1]/mag:.4f}]")
        print()
    
    # Directional derivatives
    print("2. DIRECTIONAL DERIVATIVES")
    x, y = 1.0, 1.0
    grad = gradient_f(x, y)
    
    # Test different directions
    directions = [
        np.array([1, 0]),      # x-direction
        np.array([0, 1]),      # y-direction
        np.array([1, 1]),      # diagonal
        np.array([1, -1])      # opposite diagonal
    ]
    
    for i, direction in enumerate(directions):
        # Normalize direction
        unit_direction = direction / np.linalg.norm(direction)
        directional_deriv = np.dot(grad, unit_direction)
        print(f"   Direction {i+1}: {unit_direction}")
        print(f"   Directional derivative: {directional_deriv:.4f}")
        print()

gradient_analysis_comprehensive()

# Advanced gradient calculation with error analysis
def gradient_2d_advanced(f, x, y, h=1e-7):
    """Calculate gradient of 2D function using central differences"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def f_example(x, y):
    return x**2 + y**2

def f_complex(x, y):
    return np.sin(x*y) + np.exp(x**2 + y**2)

# Compare analytical vs numerical gradients
print("=== GRADIENT COMPARISON ===\n")

def analytical_gradient_f(x, y):
    return np.array([2*x, 2*y])

test_points = [(0.5, 0.5), (1.0, 1.0), (-0.5, 1.0)]

for point in test_points:
    x, y = point
    analytical = analytical_gradient_f(x, y)
    numerical = gradient_2d_advanced(f_example, x, y)
    error = np.linalg.norm(analytical - numerical)
    
    print(f"Point ({x}, {y}):")
    print(f"  Analytical: [{analytical[0]:.6f}, {analytical[1]:.6f}]")
    print(f"  Numerical:  [{numerical[0]:.6f}, {numerical[1]:.6f}]")
    print(f"  Error:      {error:.2e}")
    print()

# Comprehensive gradient field visualization
def visualize_gradient_field():
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Function values
    Z = f_example(X, Y)
    
    # Gradients
    gradients = np.zeros((len(x), len(y), 2))
    for i in range(len(x)):
        for j in range(len(y)):
            gradients[i, j] = gradient_2d_advanced(f_example, X[i, j], Y[i, j])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contour plot with gradient vectors
    contour = ax1.contour(X, Y, Z, levels=15)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.quiver(X, Y, gradients[:, :, 0], gradients[:, :, 1], 
               angles='xy', scale_units='xy', scale=1, alpha=0.6)
    ax1.set_title('Gradient Field of f(x,y) = x² + y²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Gradient magnitude
    grad_magnitude = np.sqrt(gradients[:, :, 0]**2 + gradients[:, :, 1]**2)
    mag_contour = ax2.contour(X, Y, grad_magnitude, levels=15)
    ax2.clabel(mag_contour, inline=True, fontsize=8)
    ax2.set_title('Gradient Magnitude |∇f|')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    
    # 3D surface with gradient
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('f(x,y)')
    ax3.set_title('3D Surface with Gradient')
    
    # Gradient direction heatmap
    grad_direction = np.arctan2(gradients[:, :, 1], gradients[:, :, 0])
    im = ax4.imshow(grad_direction, extent=[-2, 2, -2, 2], origin='lower', cmap='hsv')
    ax4.set_title('Gradient Direction (angle)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()

visualize_gradient_field()

# Directional derivative analysis
def directional_derivative_analysis():
    print("\n=== DIRECTIONAL DERIVATIVE ANALYSIS ===\n")
    
    def f(x, y):
        return x**2 + y**2
    
    def gradient_f(x, y):
        return np.array([2*x, 2*y])
    
    def directional_derivative(f, gradient_f, x, y, direction, h=1e-7):
        """Compute directional derivative numerically"""
        unit_direction = direction / np.linalg.norm(direction)
        
        # Numerical approach
        f_current = f(x, y)
        f_forward = f(x + h * unit_direction[0], y + h * unit_direction[1])
        numerical_deriv = (f_forward - f_current) / h
        
        # Analytical approach
        grad = gradient_f(x, y)
        analytical_deriv = np.dot(grad, unit_direction)
        
        return numerical_deriv, analytical_deriv
    
    # Test point
    x, y = 1.0, 1.0
    print(f"Test point: ({x}, {y})")
    print(f"Gradient: {gradient_f(x, y)}")
    print()
    
    # Test different directions
    directions = [
        np.array([1, 0]),      # x-direction
        np.array([0, 1]),      # y-direction
        np.array([1, 1]),      # diagonal
        np.array([1, -1]),     # opposite diagonal
        np.array([np.cos(np.pi/4), np.sin(np.pi/4)])  # 45 degrees
    ]
    
    for i, direction in enumerate(directions):
        unit_direction = direction / np.linalg.norm(direction)
        num_deriv, ana_deriv = directional_derivative(f, gradient_f, x, y, direction)
        
        print(f"Direction {i+1}: {unit_direction}")
        print(f"  Numerical derivative:  {num_deriv:.6f}")
        print(f"  Analytical derivative: {ana_deriv:.6f}")
        print(f"  Error:                 {abs(num_deriv - ana_deriv):.2e}")
        print()

directional_derivative_analysis()

## 2.6 Applications in Machine Learning

### Gradient Descent Optimization

```python
# Advanced gradient descent analysis
def gradient_descent_analysis():
    print("=== GRADIENT DESCENT ANALYSIS ===\n")
    
    def rosenbrock(x, y):
        """Rosenbrock function: challenging optimization landscape"""
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_gradient(x, y):
        """Gradient of Rosenbrock function"""
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    def gradient_descent_advanced(f, grad_f, start_point, learning_rate=0.001, 
                                 iterations=1000, momentum=0.0):
        """Gradient descent with momentum"""
        x, y = start_point
        history = [(x, y, f(x, y))]
        velocity = np.array([0.0, 0.0])
        
        for i in range(iterations):
            gradient = grad_f(x, y)
            
            # Update with momentum
            velocity = momentum * velocity - learning_rate * gradient
            x, y = x + velocity[0], y + velocity[1]
            
            history.append((x, y, f(x, y)))
            
            # Early stopping if gradient is very small
            if np.linalg.norm(gradient) < 1e-6:
                break
        
        return np.array(history)
    
    # Test different starting points and learning rates
    start_points = [(-1, -1), (0, 0), (1, 1)]
    learning_rates = [0.0001, 0.001, 0.01]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, start_point in enumerate(start_points):
        for j, lr in enumerate(learning_rates):
            if i * 2 + j < 4:  # Only plot first 4 combinations
                path = gradient_descent_advanced(rosenbrock, rosenbrock_gradient, 
                                               start_point, learning_rate=lr, iterations=1000)
                
                ax = axes[i, j] if i < 2 else axes[1, j-2]
                
                # Create contour plot
                x = np.linspace(-2, 2, 100)
                y = np.linspace(-2, 2, 100)
                X, Y = np.meshgrid(x, y)
                Z = rosenbrock(X, Y)
                
                contour = ax.contour(X, Y, Z, levels=20)
                ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.7)
                ax.scatter(path[0, 0], path[0, 1], c='red', s=100, label='Start')
                ax.scatter(path[-1, 0], path[-1, 1], c='green', s=100, label='End')
                ax.set_title(f'Start: {start_point}, LR: {lr}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze convergence
    print("Convergence Analysis:")
    start_point = (-1, -1)
    lr = 0.001
    
    path = gradient_descent_advanced(rosenbrock, rosenbrock_gradient, 
                                   start_point, learning_rate=lr, iterations=1000)
    
    print(f"Starting point: {start_point}")
    print(f"Learning rate: {lr}")
    print(f"Final point: ({path[-1, 0]:.6f}, {path[-1, 1]:.6f})")
    print(f"Final function value: {path[-1, 2]:.6f}")
    print(f"Number of iterations: {len(path)}")
    
    # Plot convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(path[:, 2])
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(path[:, 0], label='x')
    plt.plot(path[:, 1], label='y')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    gradients = [np.linalg.norm(rosenbrock_gradient(x, y)) for x, y in path[:, :2]]
    plt.plot(gradients)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

gradient_descent_analysis()

### Loss Function Derivatives

Understanding loss function derivatives is crucial for training neural networks and other machine learning models.

```python
# Comprehensive loss function analysis
def loss_function_analysis():
    print("=== LOSS FUNCTION ANALYSIS ===\n")
    
    def mse_loss(y_pred, y_true):
        """Mean Squared Error loss"""
        return np.mean((y_pred - y_true)**2)
    
    def mse_derivative(y_pred, y_true):
        """Derivative of MSE with respect to predictions"""
        return 2 * (y_pred - y_true) / len(y_pred)
    
    def cross_entropy_loss(y_pred, y_true):
        """Cross-entropy loss for binary classification"""
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def cross_entropy_derivative(y_pred, y_true):
        """Derivative of cross-entropy with respect to predictions"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    def huber_loss(y_pred, y_true, delta=1.0):
        """Huber loss: combines MSE and MAE"""
        error = y_pred - y_true
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)
    
    def huber_derivative(y_pred, y_true, delta=1.0):
        """Derivative of Huber loss"""
        error = y_pred - y_true
        abs_error = np.abs(error)
        derivative = np.where(abs_error <= delta, error, delta * np.sign(error))
        return derivative / len(y_pred)
    
    # Test data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.9])
    
    # Calculate losses and derivatives
    losses = {
        'MSE': mse_loss(y_pred, y_true),
        'Cross-Entropy': cross_entropy_loss(y_pred, y_true),
        'Huber': huber_loss(y_pred, y_true)
    }
    
    derivatives = {
        'MSE': mse_derivative(y_pred, y_true),
        'Cross-Entropy': cross_entropy_derivative(y_pred, y_true),
        'Huber': huber_derivative(y_pred, y_true)
    }
    
    print("Loss Function Comparison:")
    for name, loss in losses.items():
        print(f"  {name}: {loss:.4f}")
    
    print("\nDerivatives:")
    for name, deriv in derivatives.items():
        print(f"  {name}: {deriv}")
    
    # Visualize loss functions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE loss landscape
    y_true_single = 1.0
    y_pred_range = np.linspace(-2, 4, 100)
    mse_values = [(y_pred - y_true_single)**2 for y_pred in y_pred_range]
    mse_derivatives = [2 * (y_pred - y_true_single) for y_pred in y_pred_range]
    
    ax1.plot(y_pred_range, mse_values, 'b-', linewidth=2, label='MSE Loss')
    ax1.plot(y_pred_range, mse_derivatives, 'r--', linewidth=2, label='MSE Derivative')
    ax1.axvline(y_true_single, color='g', linestyle=':', label='True Value')
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Value')
    ax1.set_title('MSE Loss and Derivative')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cross-entropy loss landscape
    y_pred_range_ce = np.linspace(0.01, 0.99, 100)
    ce_values = [-y_true_single * np.log(y_pred) - (1 - y_true_single) * np.log(1 - y_pred) 
                 for y_pred in y_pred_range_ce]
    ce_derivatives = [(y_pred - y_true_single) / (y_pred * (1 - y_pred)) 
                      for y_pred in y_pred_range_ce]
    
    ax2.plot(y_pred_range_ce, ce_values, 'b-', linewidth=2, label='Cross-Entropy Loss')
    ax2.plot(y_pred_range_ce, ce_derivatives, 'r--', linewidth=2, label='CE Derivative')
    ax2.axvline(y_true_single, color='g', linestyle=':', label='True Value')
    ax2.set_xlabel('Prediction')
    ax2.set_ylabel('Value')
    ax2.set_title('Cross-Entropy Loss and Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Huber loss landscape
    huber_values = [huber_loss(np.array([y_pred]), np.array([y_true_single])) 
                    for y_pred in y_pred_range]
    huber_derivatives = [huber_derivative(np.array([y_pred]), np.array([y_true_single]))[0] 
                         for y_pred in y_pred_range]
    
    ax3.plot(y_pred_range, huber_values, 'b-', linewidth=2, label='Huber Loss')
    ax3.plot(y_pred_range, huber_derivatives, 'r--', linewidth=2, label='Huber Derivative')
    ax3.axvline(y_true_single, color='g', linestyle=':', label='True Value')
    ax3.set_xlabel('Prediction')
    ax3.set_ylabel('Value')
    ax3.set_title('Huber Loss and Derivative')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Comparison of derivatives
    ax4.plot(y_pred_range, mse_derivatives, 'b-', linewidth=2, label='MSE')
    ax4.plot(y_pred_range, huber_derivatives, 'r-', linewidth=2, label='Huber')
    ax4.axvline(y_true_single, color='g', linestyle=':', label='True Value')
    ax4.set_xlabel('Prediction')
    ax4.set_ylabel('Derivative')
    ax4.set_title('Derivative Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

loss_function_analysis()

# Neural network gradient analysis
def neural_network_gradient_analysis():
    print("\n=== NEURAL NETWORK GRADIENT ANALYSIS ===\n")
    
    # Simple neural network: y = σ(w₂ * σ(w₁ * x + b₁) + b₂)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    def forward_pass(x, w1, b1, w2, b2):
        """Forward pass through the network"""
        z1 = w1 * x + b1
        a1 = sigmoid(z1)
        z2 = w2 * a1 + b2
        y = sigmoid(z2)
        return y, z1, a1, z2
    
    def backward_pass(x, y_true, w1, b1, w2, b2):
        """Backward pass using chain rule"""
        y, z1, a1, z2 = forward_pass(x, w1, b1, w2, b2)
        
        # Loss: L = (y - y_true)²
        loss = (y - y_true)**2
        
        # Chain rule for gradients
        # dL/dy = 2(y - y_true)
        # dy/dz2 = σ'(z2)
        # dz2/da1 = w2
        # da1/dz1 = σ'(z1)
        # dz1/dw1 = x, dz1/db1 = 1
        # dz2/dw2 = a1, dz2/db2 = 1
        
        dL_dy = 2 * (y - y_true)
        dy_dz2 = sigmoid_derivative(z2)
        dz2_da1 = w2
        da1_dz1 = sigmoid_derivative(z1)
        
        # Gradients
        dL_dw2 = dL_dy * dy_dz2 * a1
        dL_db2 = dL_dy * dy_dz2
        dL_dw1 = dL_dy * dy_dz2 * dz2_da1 * da1_dz1 * x
        dL_db1 = dL_dy * dy_dz2 * dz2_da1 * da1_dz1
        
        return dL_dw1, dL_db1, dL_dw2, dL_db2, loss
    
    # Test the network
    x = 2.0
    y_true = 0.8
    w1, b1, w2, b2 = 1.0, 0.5, 0.8, 0.2
    
    print(f"Input: x = {x}")
    print(f"Target: y = {y_true}")
    print(f"Parameters: w1={w1}, b1={b1}, w2={w2}, b2={b2}")
    
    y, z1, a1, z2 = forward_pass(x, w1, b1, w2, b2)
    print(f"Forward pass: y = {y:.4f}")
    
    dL_dw1, dL_db1, dL_dw2, dL_db2, loss = backward_pass(x, y_true, w1, b1, w2, b2)
    print(f"Loss: {loss:.4f}")
    print(f"Gradients: dL/dw1={dL_dw1:.4f}, dL/db1={dL_db1:.4f}, dL/dw2={dL_dw2:.4f}, dL/db2={dL_db2:.4f}")
    
    # Verify with numerical gradients
    def loss_function(w1, b1, w2, b2):
        y, _, _, _ = forward_pass(x, w1, b1, w2, b2)
        return (y - y_true)**2
    
    h = 1e-7
    numerical_gradients = []
    
    for param, param_name in [(w1, 'w1'), (b1, 'b1'), (w2, 'w2'), (b2, 'b2')]:
        # Create parameter list for numerical gradient
        params = [w1, b1, w2, b2]
        param_idx = ['w1', 'b1', 'w2', 'b2'].index(param_name)
        
        # Numerical gradient
        params_plus = params.copy()
        params_plus[param_idx] += h
        params_minus = params.copy()
        params_minus[param_idx] -= h
        
        num_grad = (loss_function(*params_plus) - loss_function(*params_minus)) / (2 * h)
        analytical_grad = [dL_dw1, dL_db1, dL_dw2, dL_db2][param_idx]
        
        print(f"\n{param_name}:")
        print(f"  Numerical:  {num_grad:.6f}")
        print(f"  Analytical: {analytical_grad:.6f}")
        print(f"  Error:      {abs(num_grad - analytical_grad):.2e}")

neural_network_gradient_analysis()
```

## Summary

Derivatives are the mathematical foundation of optimization and machine learning. This comprehensive exploration covered:

### Key Concepts:
1. **Definition and Interpretation**: Derivatives measure instantaneous rate of change
2. **Basic Rules**: Power, sum, product, quotient rules for efficient computation
3. **Chain Rule**: Essential for composite functions and backpropagation
4. **Partial Derivatives**: Handle multivariable functions
5. **Gradient**: Vector of partial derivatives pointing in direction of steepest ascent
6. **Directional Derivatives**: Rate of change in specific directions

### Applications in Machine Learning:
1. **Gradient Descent**: Foundation of most optimization algorithms
2. **Backpropagation**: Chain rule applied to neural network training
3. **Loss Functions**: Derivatives guide parameter updates
4. **Activation Functions**: Derivatives of sigmoid, ReLU, tanh, etc.
5. **Convergence Analysis**: Understanding optimization behavior

### Mathematical Insights:
- Derivatives provide local linear approximations
- Gradients point in direction of maximum increase
- Chain rule enables automatic differentiation
- Partial derivatives handle high-dimensional spaces
- Numerical methods provide verification of analytical results

These concepts form the mathematical backbone of modern machine learning, enabling the training of complex models through gradient-based optimization.

## Next Steps

Understanding derivatives enables us to explore their applications in optimization, curve sketching, and machine learning algorithms in the next section. 