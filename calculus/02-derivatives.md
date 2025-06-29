# Derivatives and Differentiation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)

## Introduction

Derivatives measure how a function changes as its input changes. This concept is fundamental to optimization, which is central to machine learning algorithms.

## 2.1 Definition of Derivatives

The derivative of a function f(x) at a point x is the limit of the difference quotient:

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.misc import derivative

# Define a function
def f(x):
    return x**2

# Numerical derivative using finite differences
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x)) / h

# Symbolic derivative using SymPy
x = sp.Symbol('x')
f_sym = x**2
f_prime = sp.diff(f_sym, x)
print(f"Symbolic derivative of x²: {f_prime}")

# Compare numerical and symbolic derivatives
x_vals = np.linspace(-2, 2, 100)
numerical_derivatives = [numerical_derivative(f, x) for x in x_vals]
symbolic_derivatives = [2*x for x in x_vals]  # f'(x) = 2x

plt.figure(figsize=(12, 5))

# Original function
plt.subplot(1, 2, 1)
plt.plot(x_vals, [f(x) for x in x_vals], 'b-', linewidth=2, label='f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True)

# Derivatives
plt.subplot(1, 2, 2)
plt.plot(x_vals, numerical_derivatives, 'r-', linewidth=2, label='Numerical f\'(x)')
plt.plot(x_vals, symbolic_derivatives, 'g--', linewidth=2, label='Symbolic f\'(x) = 2x')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.title('Derivatives')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 2.2 Basic Differentiation Rules

```python
# Power rule, constant rule, sum rule
def demonstrate_rules():
    x = sp.Symbol('x')
    
    # Power rule: d/dx(x^n) = n*x^(n-1)
    power_expr = x**3
    power_deriv = sp.diff(power_expr, x)
    print(f"d/dx(x³) = {power_deriv}")
    
    # Constant rule: d/dx(c) = 0
    const_expr = 5
    const_deriv = sp.diff(const_expr, x)
    print(f"d/dx(5) = {const_deriv}")
    
    # Sum rule: d/dx(f + g) = d/dx(f) + d/dx(g)
    sum_expr = x**2 + 3*x + 1
    sum_deriv = sp.diff(sum_expr, x)
    print(f"d/dx(x² + 3x + 1) = {sum_deriv}")
    
    # Product rule: d/dx(f*g) = f*dg + g*df
    product_expr = x**2 * sp.sin(x)
    product_deriv = sp.diff(product_expr, x)
    print(f"d/dx(x²*sin(x)) = {product_deriv}")
    
    # Quotient rule: d/dx(f/g) = (g*df - f*dg)/g²
    quotient_expr = x**2 / (x + 1)
    quotient_deriv = sp.diff(quotient_expr, x)
    print(f"d/dx(x²/(x+1)) = {quotient_deriv}")

demonstrate_rules()
```

## 2.3 Chain Rule

Essential for backpropagation in neural networks.

```python
# Chain rule demonstration
def chain_rule_example():
    x = sp.Symbol('x')
    
    # f(x) = sin(x²)
    f_expr = sp.sin(x**2)
    f_deriv = sp.diff(f_expr, x)
    print(f"d/dx(sin(x²)) = {f_deriv}")
    
    # f(x) = e^(x²)
    f_expr2 = sp.exp(x**2)
    f_deriv2 = sp.diff(f_expr2, x)
    print(f"d/dx(e^(x²)) = {f_deriv2}")
    
    # f(x) = ln(x² + 1)
    f_expr3 = sp.log(x**2 + 1)
    f_deriv3 = sp.diff(f_expr3, x)
    print(f"d/dx(ln(x² + 1)) = {f_deriv3}")

chain_rule_example()

# Visualize chain rule
def composite_function(x):
    return np.sin(x**2)

def chain_rule_derivative(x):
    return 2 * x * np.cos(x**2)

x_vals = np.linspace(-2, 2, 1000)
y_vals = composite_function(x_vals)
dy_vals = chain_rule_derivative(x_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = sin(x²)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Composite Function')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_vals, dy_vals, 'r-', linewidth=2, label='f\'(x) = 2x*cos(x²)')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.title('Chain Rule Derivative')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 2.4 Partial Derivatives

Critical for multivariable functions in machine learning.

```python
# Partial derivatives
def partial_derivatives():
    x, y = sp.symbols('x y')
    
    # f(x,y) = x² + y²
    f_expr = x**2 + y**2
    df_dx = sp.diff(f_expr, x)
    df_dy = sp.diff(f_expr, y)
    
    print(f"f(x,y) = x² + y²")
    print(f"∂f/∂x = {df_dx}")
    print(f"∂f/∂y = {df_dy}")
    
    # f(x,y) = x*y + sin(x)
    f_expr2 = x*y + sp.sin(x)
    df_dx2 = sp.diff(f_expr2, x)
    df_dy2 = sp.diff(f_expr2, y)
    
    print(f"\nf(x,y) = x*y + sin(x)")
    print(f"∂f/∂x = {df_dx2}")
    print(f"∂f/∂y = {df_dy2}")

partial_derivatives()

# Visualize partial derivatives
from mpl_toolkits.mplot3d import Axes3D

def f_3d(x, y):
    return x**2 + y**2

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = f_3d(X, Y)

fig = plt.figure(figsize=(15, 5))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('f(x,y) = x² + y²')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# ∂f/∂x = 2x
ax2 = fig.add_subplot(132)
df_dx = 2 * X
contour = ax2.contour(X, Y, df_dx, levels=10)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_title('∂f/∂x = 2x')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# ∂f/∂y = 2y
ax3 = fig.add_subplot(133)
df_dy = 2 * Y
contour = ax3.contour(X, Y, df_dy, levels=10)
ax3.clabel(contour, inline=True, fontsize=8)
ax3.set_title('∂f/∂y = 2y')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.tight_layout()
plt.show()
```

## 2.5 Gradient and Directional Derivatives

```python
# Gradient calculation
def gradient_2d(f, x, y, h=1e-7):
    """Calculate gradient of 2D function using finite differences"""
    df_dx = (f(x + h, y) - f(x, y)) / h
    df_dy = (f(x, y + h) - f(x, y)) / h
    return np.array([df_dx, df_dy])

def f_example(x, y):
    return x**2 + y**2

# Calculate gradient at different points
points = [(0, 0), (1, 1), (-1, 0)]
for point in points:
    grad = gradient_2d(f_example, point[0], point[1])
    print(f"Gradient at {point}: {grad}")

# Visualize gradient field
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)

gradients = np.zeros((len(x), len(y), 2))
for i in range(len(x)):
    for j in range(len(y)):
        gradients[i, j] = gradient_2d(f_example, X[i, j], Y[i, j])

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, gradients[:, :, 0], gradients[:, :, 1], 
           angles='xy', scale_units='xy', scale=1)
plt.title('Gradient Field of f(x,y) = x² + y²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

## 2.6 Applications in Machine Learning

### Gradient Descent

```python
# Simple gradient descent implementation
def gradient_descent_2d(f, grad_f, start_point, learning_rate=0.1, iterations=100):
    """Gradient descent for 2D function"""
    x, y = start_point
    history = [(x, y)]
    
    for i in range(iterations):
        gradient = grad_f(x, y)
        x = x - learning_rate * gradient[0]
        y = y - learning_rate * gradient[1]
        history.append((x, y))
    
    return np.array(history)

def rosenbrock(x, y):
    """Rosenbrock function: common test function for optimization"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Run gradient descent
start_point = (-1, -1)
path = gradient_descent_2d(rosenbrock, rosenbrock_gradient, start_point, 
                          learning_rate=0.001, iterations=1000)

# Visualize optimization path
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=20)
plt.clabel(contour, inline=True, fontsize=8)
plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Optimization path')
plt.scatter(path[0, 0], path[0, 1], c='red', s=100, label='Start')
plt.scatter(path[-1, 0], path[-1, 1], c='green', s=100, label='End')
plt.title('Gradient Descent on Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final point: ({path[-1, 0]:.6f}, {path[-1, 1]:.6f})")
print(f"Final value: {rosenbrock(path[-1, 0], path[-1, 1]):.6f}")
```

### Loss Function Derivatives

```python
# Common loss function derivatives
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

# Demonstrate loss function derivatives
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.3, 0.9])

print("MSE Loss and Derivative:")
print(f"Loss: {mse_loss(y_pred, y_true):.4f}")
print(f"Derivative: {mse_derivative(y_pred, y_true)}")

print("\nCross-Entropy Loss and Derivative:")
print(f"Loss: {cross_entropy_loss(y_pred, y_true):.4f}")
print(f"Derivative: {cross_entropy_derivative(y_pred, y_true)}")
```

## Summary

- **Derivatives** measure rate of change and are fundamental to optimization
- **Basic rules** (power, sum, product, quotient) simplify differentiation
- **Chain rule** is essential for composite functions and backpropagation
- **Partial derivatives** handle multivariable functions
- **Gradient** provides direction of steepest ascent
- **Applications** include gradient descent, loss function optimization, and neural network training

## Next Steps

Understanding derivatives enables us to explore their applications in optimization, curve sketching, and machine learning algorithms in the next section. 