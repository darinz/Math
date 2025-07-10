# Python code extracted from 01-limits-continuity.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, limit, simplify

# Define a function with removable discontinuity
def f(x):
    return (x**2 - 1) / (x - 1)

# Calculate limit using SymPy
x = sp.Symbol('x')
limit_expr = (x**2 - 1) / (x - 1)

# Simplify the expression to understand the limit
simplified_expr = sp.simplify(limit_expr)
print(f"Original expression: {limit_expr}")
print(f"Simplified expression: {simplified_expr}")

# Calculate the limit
limit_value = sp.limit(limit_expr, x, 1)
print(f"Limit as x approaches 1: {limit_value}")

# Visualize the function with detailed analysis
x_vals = np.linspace(0.5, 1.5, 1000)
y_vals = [f(x) for x in x_vals if x != 1]

plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x²-1)/(x-1)')
plt.axhline(y=2, color='r', linestyle='--', linewidth=2, label='Limit = 2')
plt.axvline(x=1, color='g', linestyle='--', linewidth=2, label='x = 1')
plt.scatter(1, 2, color='red', s=100, zorder=5, label='Limit point')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Limit Example: (x²-1)/(x-1) as x → 1')
plt.legend()
plt.grid(True, alpha=0.3)

# Zoomed view around x = 1
plt.subplot(2, 1, 2)
x_zoom = np.linspace(0.9, 1.1, 200)
y_zoom = [f(x) for x in x_zoom if x != 1]
plt.plot(x_zoom, y_zoom, 'b-', linewidth=2)
plt.axhline(y=2, color='r', linestyle='--', linewidth=2)
plt.axvline(x=1, color='g', linestyle='--', linewidth=2)
plt.scatter(1, 2, color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Zoomed View Around x = 1')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate limit calculation numerically
print("\nNumerical verification:")
for h in [0.1, 0.01, 0.001, 0.0001]:
    left_val = f(1 - h)
    right_val = f(1 + h)
    print(f"f(1-{h}) = {left_val:.6f}, f(1+{h}) = {right_val:.6f}")

# Code Block 2
# One-sided limits with detailed analysis
def step_function(x):
    """Heaviside step function: returns -1 for x < 0, 1 for x ≥ 0"""
    return np.where(x < 0, -1, 1)

def sign_function(x):
    """Sign function: returns -1 for x < 0, 0 for x = 0, 1 for x > 0"""
    return np.where(x < 0, -1, np.where(x > 0, 1, 0))

# Create visualization
x_vals = np.linspace(-2, 2, 1000)
y_step = step_function(x_vals)
y_sign = sign_function(x_vals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Step function
ax1.plot(x_vals, y_step, 'b-', linewidth=3, label='Step Function')
ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='x = 0')
ax1.scatter(0, -1, color='red', s=100, zorder=5, label='Left limit = -1')
ax1.scatter(0, 1, color='green', s=100, zorder=5, label='Right limit = 1')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('One-Sided Limits: Step Function')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.5, 1.5)

# Sign function
ax2.plot(x_vals, y_sign, 'b-', linewidth=3, label='Sign Function')
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='x = 0')
ax2.scatter(0, -1, color='red', s=100, zorder=5, label='Left limit = -1')
ax2.scatter(0, 1, color='green', s=100, zorder=5, label='Right limit = 1')
ax2.scatter(0, 0, color='purple', s=100, zorder=5, label='f(0) = 0')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('One-Sided Limits: Sign Function')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# Calculate one-sided limits using SymPy
print("One-sided limits analysis:")
print("Step function:")
print(f"  Left limit: {sp.limit(sp.Piecewise((-1, x < 0), (1, True)), x, 0, dir='-')}")
print(f"  Right limit: {sp.limit(sp.Piecewise((-1, x < 0), (1, True)), x, 0, dir='+')}")
print(f"  Two-sided limit exists: {sp.limit(sp.Piecewise((-1, x < 0), (1, True)), x, 0) == sp.limit(sp.Piecewise((-1, x < 0), (1, True)), x, 0, dir='-') == sp.limit(sp.Piecewise((-1, x < 0), (1, True)), x, 0, dir='+')}")

# Numerical verification
print("\nNumerical verification:")
for h in [0.1, 0.01, 0.001]:
    left_val = step_function(-h)
    right_val = step_function(h)
    print(f"f(-{h}) = {left_val}, f({h}) = {right_val}")

# Code Block 3
# Comprehensive continuity analysis
def continuous_func(x):
    """Continuous function: f(x) = x²"""
    return x**2

def removable_discontinuity(x):
    """Function with removable discontinuity at x = 0"""
    return np.where(x != 0, np.sin(x)/x, 1)

def jump_discontinuity(x):
    """Function with jump discontinuity at x = 0"""
    return np.where(x < 0, x, x + 1)

def infinite_discontinuity(x):
    """Function with infinite discontinuity at x = 0"""
    return np.where(x != 0, 1/x, 0)

# Create comprehensive visualization
x_vals = np.linspace(-2, 2, 1000)
y1_vals = continuous_func(x_vals)
y2_vals = removable_discontinuity(x_vals)
y3_vals = jump_discontinuity(x_vals)
y4_vals = infinite_discontinuity(x_vals)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Continuous function
ax1.plot(x_vals, y1_vals, 'b-', linewidth=2)
ax1.set_title('Continuous Function: f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)

# Removable discontinuity
ax2.plot(x_vals, y2_vals, 'g-', linewidth=2)
ax2.scatter(0, 1, color='red', s=100, zorder=5, label='f(0) = 1')
ax2.set_title('Removable Discontinuity: f(x) = sin(x)/x')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Jump discontinuity
ax3.plot(x_vals, y3_vals, 'r-', linewidth=2)
ax3.scatter(0, 0, color='red', s=100, zorder=5, label='Left limit = 0')
ax3.scatter(0, 1, color='blue', s=100, zorder=5, label='Right limit = 1')
ax3.set_title('Jump Discontinuity: f(x) = x for x < 0, x+1 for x ≥ 0')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Infinite discontinuity
ax4.plot(x_vals, y4_vals, 'purple', linewidth=2)
ax4.set_title('Infinite Discontinuity: f(x) = 1/x')
ax4.set_xlabel('x')
ax4.set_ylabel('f(x)')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-5, 5)

plt.tight_layout()
plt.show()

# Mathematical analysis of continuity
print("Continuity Analysis:")
print("1. Continuous function: f(x) = x²")
print(f"   f(0) = {continuous_func(0)}")
print(f"   lim(x→0) f(x) = {sp.limit(x**2, x, 0)}")
print(f"   Continuous at x = 0: {continuous_func(0) == sp.limit(x**2, x, 0)}")

print("\n2. Removable discontinuity: f(x) = sin(x)/x")
print(f"   f(0) = 1 (defined)")
print(f"   lim(x→0) sin(x)/x = {sp.limit(sp.sin(x)/x, x, 0)}")
print(f"   Continuous at x = 0: {1 == sp.limit(sp.sin(x)/x, x, 0)}")

# Code Block 4
# Comprehensive analysis of limits at infinity
def rational_func(x):
    """Rational function: (3x² + 2x + 1)/(x² + 1)"""
    return (3*x**2 + 2*x + 1) / (x**2 + 1)

def exponential_func(x):
    """Exponential function: e^x"""
    return np.exp(x)

def logarithmic_func(x):
    """Logarithmic function: ln(x)"""
    return np.where(x > 0, np.log(x), np.nan)

def polynomial_func(x):
    """Polynomial function: x³ - 2x² + x"""
    return x**3 - 2*x**2 + x

# Create comprehensive visualization
x_vals = np.linspace(0, 10, 1000)
y1_vals = rational_func(x_vals)
y2_vals = exponential_func(x_vals)
y3_vals = logarithmic_func(x_vals)
y4_vals = polynomial_func(x_vals)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Rational function
ax1.plot(x_vals, y1_vals, 'b-', linewidth=2)
ax1.axhline(y=3, color='r', linestyle='--', linewidth=2, label='Horizontal asymptote y = 3')
ax1.set_title('Rational Function: (3x²+2x+1)/(x²+1)')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Exponential function
ax2.plot(x_vals, y2_vals, 'g-', linewidth=2)
ax2.set_title('Exponential Function: e^x')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True, alpha=0.3)

# Logarithmic function
ax3.plot(x_vals, y3_vals, 'r-', linewidth=2)
ax3.set_title('Logarithmic Function: ln(x)')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.grid(True, alpha=0.3)

# Polynomial function
ax4.plot(x_vals, y4_vals, 'purple', linewidth=2)
ax4.set_title('Polynomial Function: x³ - 2x² + x')
ax4.set_xlabel('x')
ax4.set_ylabel('f(x)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate limits at infinity
print("Limits at Infinity Analysis:")
print("1. Rational function: (3x² + 2x + 1)/(x² + 1)")
print(f"   lim(x→∞) = {sp.limit((3*x**2 + 2*x + 1) / (x**2 + 1), x, sp.oo)}")

print("\n2. Exponential function: e^x")
print(f"   lim(x→∞) = {sp.limit(sp.exp(x), x, sp.oo)}")

print("\n3. Logarithmic function: ln(x)")
print(f"   lim(x→∞) = {sp.limit(sp.log(x), x, sp.oo)}")

print("\n4. Polynomial function: x³ - 2x² + x")
print(f"   lim(x→∞) = {sp.limit(x**3 - 2*x**2 + x, x, sp.oo)}")

# Growth rate comparison
print("\nGrowth Rate Comparison (numerical):")
x_large = 1000
print(f"At x = {x_large}:")
print(f"  Rational: {rational_func(x_large):.2f}")
print(f"  Polynomial: {polynomial_func(x_large):.2e}")
print(f"  Exponential: {exponential_func(x_large):.2e}")
print(f"  Logarithmic: {logarithmic_func(x_large):.2f}")

# Code Block 5
# Advanced convergence analysis
def gradient_descent_convergence(learning_rate=0.1, iterations=100, starting_point=2.0):
    """
    Analyze gradient descent convergence for f(x) = x²
    This function has a global minimum at x = 0
    """
    x = starting_point
    history = [x]
    gradients = []
    
    for i in range(iterations):
        # Gradient of f(x) = x² is f'(x) = 2x
        gradient = 2 * x
        gradients.append(gradient)
        
        # Update rule: x = x - α * ∇f(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return history, gradients

# Analyze convergence with different learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]
iterations = 50

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

for i, lr in enumerate(learning_rates):
    history, gradients = gradient_descent_convergence(learning_rate=lr, iterations=iterations)
    
    row = i // 2
    col = i % 2
    
    ax = [ax1, ax2, ax3, ax4][i]
    
    # Plot convergence
    ax.plot(range(iterations + 1), history, 'b-', linewidth=2, label=f'x values')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Optimal value = 0')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('x value')
    ax.set_title(f'Gradient Descent (α = {lr})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Learning rate {lr}: Final value = {history[-1]:.6f}")

plt.tight_layout()
plt.show()

# Mathematical analysis of convergence
print("\nMathematical Analysis:")
print("For f(x) = x²:")
print("  - Gradient: f'(x) = 2x")
print("  - Update rule: x_{n+1} = x_n - α * 2x_n = x_n(1 - 2α)")
print("  - Convergence condition: |1 - 2α| < 1")
print("  - Optimal learning rate: α = 0.5")

for lr in learning_rates:
    convergence_rate = abs(1 - 2 * lr)
    print(f"  α = {lr}: convergence rate = {convergence_rate:.3f} {'(converges)' if convergence_rate < 1 else '(diverges)'}")

# Code Block 6
# Comprehensive loss function analysis
def mse_loss(predictions, targets):
    """Mean Squared Error loss"""
    return np.mean((predictions - targets)**2)

def cross_entropy_loss(predictions, targets):
    """Cross-entropy loss (simplified)"""
    epsilon = 1e-15  # Avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

def hinge_loss(predictions, targets):
    """Hinge loss for binary classification"""
    return np.mean(np.maximum(0, 1 - targets * predictions))

# Simulate different training scenarios
epochs = 100

# Scenario 1: Well-behaved convergence
loss_history_1 = []
for epoch in range(epochs):
    # Exponential decay with noise
    base_loss = 10 * np.exp(-epoch / 20) + 0.1
    noise = np.random.normal(0, 0.01)
    loss = max(0.1, base_loss + noise)
    loss_history_1.append(loss)

# Scenario 2: Oscillating convergence
loss_history_2 = []
for epoch in range(epochs):
    base_loss = 10 * np.exp(-epoch / 30) + 0.2
    oscillation = 0.1 * np.sin(epoch * 0.5)
    loss = max(0.2, base_loss + oscillation)
    loss_history_2.append(loss)

# Scenario 3: Plateau then convergence
loss_history_3 = []
for epoch in range(epochs):
    if epoch < 30:
        loss = 5.0  # Plateau
    else:
        loss = 5.0 * np.exp(-(epoch - 30) / 15) + 0.1
    loss_history_3.append(loss)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Loss curves
ax1.plot(range(epochs), loss_history_1, 'b-', linewidth=2, label='Smooth convergence')
ax1.plot(range(epochs), loss_history_2, 'g-', linewidth=2, label='Oscillating convergence')
ax1.plot(range(epochs), loss_history_3, 'r-', linewidth=2, label='Plateau then convergence')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Behavior')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log scale for better visualization
ax2.semilogy(range(epochs), loss_history_1, 'b-', linewidth=2, label='Smooth convergence')
ax2.semilogy(range(epochs), loss_history_2, 'g-', linewidth=2, label='Oscillating convergence')
ax2.semilogy(range(epochs), loss_history_3, 'r-', linewidth=2, label='Plateau then convergence')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Training Loss Behavior (Log Scale)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mathematical analysis
print("Loss Function Analysis:")
print("1. Smooth convergence:")
print(f"   Initial loss: {loss_history_1[0]:.3f}")
print(f"   Final loss: {loss_history_1[-1]:.3f}")
print(f"   Convergence rate: {loss_history_1[0]/loss_history_1[-1]:.1f}x reduction")

print("\n2. Oscillating convergence:")
print(f"   Average loss (last 20 epochs): {np.mean(loss_history_2[-20:]):.3f}")
print(f"   Standard deviation (last 20 epochs): {np.std(loss_history_2[-20:]):.3f}")

print("\n3. Plateau then convergence:")
print(f"   Plateau duration: 30 epochs")
print(f"   Final convergence rate: {loss_history_3[30]/loss_history_3[-1]:.1f}x reduction")

# Code Block 7
# Numerical stability examples
def unstable_division(x, y):
    """Demonstrate numerical instability in division"""
    return x / y

def stable_division(x, y, epsilon=1e-15):
    """Stable division with protection against division by zero"""
    return x / (y + epsilon)

# Test numerical stability
x_vals = np.linspace(0.1, 1.0, 10)
y_vals = np.linspace(1e-15, 1e-10, 10)

print("Numerical Stability Analysis:")
print("Testing division near zero:")

for x, y in zip(x_vals, y_vals):
    try:
        unstable_result = unstable_division(x, y)
        stable_result = stable_division(x, y)
        print(f"x={x:.1e}, y={y:.1e}: unstable={unstable_result:.6f}, stable={stable_result:.6f}")
    except:
        print(f"x={x:.1e}, y={y:.1e}: unstable=ERROR, stable={stable_division(x, y):.6f}")

# Demonstrate limit concepts in numerical computation
print("\nLimit concepts in numerical computation:")
print("Computing lim(x→0) sin(x)/x numerically:")

for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    result = np.sin(h) / h
    print(f"h = {h:.1e}: sin({h})/{h} = {result:.10f}")

