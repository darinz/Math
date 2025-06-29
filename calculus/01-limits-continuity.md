# Limits and Continuity

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)

## Introduction

Limits and continuity form the foundation of calculus. Understanding these concepts is crucial for grasping derivatives, integrals, and their applications in machine learning and data science.

Limits are fundamental to calculus and form the foundation for derivatives and integrals. In AI/ML, understanding limits helps with convergence analysis, optimization algorithms, and understanding model behavior.

## 1.1 Definition of Limits

A limit describes the behavior of a function as the input approaches a specific value.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define a function
def f(x):
    return (x**2 - 1) / (x - 1)

# Calculate limit using SymPy
x = sp.Symbol('x')
limit_expr = (x**2 - 1) / (x - 1)
limit_value = sp.limit(limit_expr, x, 1)
print(f"Limit as x approaches 1: {limit_value}")

# Visualize the function
x_vals = np.linspace(0.5, 1.5, 1000)
y_vals = [f(x) for x in x_vals if x != 1]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = (x²-1)/(x-1)')
plt.axhline(y=2, color='r', linestyle='--', label='Limit = 2')
plt.axvline(x=1, color='g', linestyle='--', label='x = 1')
plt.scatter(1, 2, color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Limit Example: (x²-1)/(x-1) as x → 1')
plt.legend()
plt.grid(True)
plt.show()
```

## 1.2 One-Sided Limits

```python
# One-sided limits
def g(x):
    return np.where(x < 0, -1, 1)

x_vals = np.linspace(-2, 2, 1000)
y_vals = g(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', label='x = 0')
plt.scatter(0, -1, color='red', s=100, zorder=5)
plt.scatter(0, 1, color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('One-Sided Limits: Step Function')
plt.legend()
plt.grid(True)
plt.show()

# Calculate one-sided limits
left_limit = sp.limit(sp.Piecewise((sp.Symbol('x'), sp.Symbol('x') < 0), (1, True)), x, 0, dir='-')
right_limit = sp.limit(sp.Piecewise((sp.Symbol('x'), sp.Symbol('x') < 0), (1, True)), x, 0, dir='+')
print(f"Left limit: {left_limit}")
print(f"Right limit: {right_limit}")
```

## 1.3 Continuity

A function is continuous at a point if the limit exists and equals the function value.

```python
# Continuous vs discontinuous functions
def continuous_func(x):
    return x**2

def discontinuous_func(x):
    return np.where(x != 0, 1/x, 0)

x_vals = np.linspace(-2, 2, 1000)
y1_vals = continuous_func(x_vals)
y2_vals = discontinuous_func(x_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Continuous function
ax1.plot(x_vals, y1_vals, 'b-', linewidth=2)
ax1.set_title('Continuous Function: f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True)

# Discontinuous function
ax2.plot(x_vals, y2_vals, 'r-', linewidth=2)
ax2.set_title('Discontinuous Function: f(x) = 1/x')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True)
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.show()
```

## 1.4 Limits at Infinity

Important for understanding asymptotic behavior in algorithms.

```python
# Limits at infinity
def rational_func(x):
    return (3*x**2 + 2*x + 1) / (x**2 + 1)

x_vals = np.linspace(0, 100, 1000)
y_vals = rational_func(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(y=3, color='r', linestyle='--', label='Limit = 3')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Limit at Infinity: (3x²+2x+1)/(x²+1)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate limit at infinity
limit_inf = sp.limit((3*x**2 + 2*x + 1) / (x**2 + 1), x, sp.oo)
print(f"Limit as x → ∞: {limit_inf}")
```

## 1.5 Applications in AI/ML

### Convergence Analysis

```python
# Convergence of gradient descent
def gradient_descent_convergence(learning_rate=0.1, iterations=100):
    x = 2.0  # Starting point
    history = [x]
    
    for i in range(iterations):
        # Gradient of f(x) = x²
        gradient = 2 * x
        x = x - learning_rate * gradient
        history.append(x)
    
    return history

# Analyze convergence
iterations = 100
history = gradient_descent_convergence(learning_rate=0.1, iterations=iterations)

plt.figure(figsize=(10, 6))
plt.plot(range(iterations + 1), history, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Optimal value = 0')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final value: {history[-1]:.6f}")
print(f"Convergence limit: 0")
```

### Loss Function Behavior

```python
# Loss function limits
def mse_loss(predictions, targets):
    return np.mean((predictions - targets)**2)

# Simulate training process
epochs = 100
loss_history = []
for epoch in range(epochs):
    # Simulate decreasing loss
    loss = 10 * np.exp(-epoch / 20) + 0.1
    loss_history.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history, 'b-', linewidth=2)
plt.axhline(y=0.1, color='r', linestyle='--', label='Minimum loss = 0.1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.legend()
plt.grid(True)
plt.show()

print(f"Loss limit as epochs → ∞: 0.1")
```

## Summary

- **Limits** describe function behavior as inputs approach specific values
- **Continuity** ensures smooth function behavior
- **One-sided limits** are important for piecewise functions
- **Limits at infinity** help understand asymptotic behavior
- **Applications** include convergence analysis and algorithm behavior

## Next Steps

Understanding limits is crucial for derivatives, which we'll explore in the next section. The concept of limits forms the foundation for all calculus operations used in machine learning algorithms. 