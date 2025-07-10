"""
Derivatives - Python Implementation
==================================

This file demonstrates the key concepts of derivatives using Python.
It includes comprehensive examples, visualizations, and practical applications
in machine learning and data science.

Key Concepts Covered:
- Definition and interpretation of derivatives
- Differentiation rules and techniques
- Higher-order derivatives
- Applications in optimization and machine learning
- Numerical vs symbolic differentiation
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, simplify
import warnings
warnings.filterwarnings('ignore')

# Set up SymPy symbols for symbolic computation
x, y, z = symbols('x y z')

print("=" * 60)
print("DERIVATIVES - PYTHON IMPLEMENTATION")
print("=" * 60)

# =============================================================================
# SECTION 1: DEFINITION OF DERIVATIVES
# =============================================================================

print("\n1. DEFINITION OF DERIVATIVES")
print("-" * 30)

def demonstrate_derivative_definition():
    """
    Demonstrate the definition of derivatives using the limit concept.
    
    The derivative of f at a point a is defined as:
    f'(a) = lim_{h→0} [f(a+h) - f(a)]/h
    
    This represents the instantaneous rate of change of f at a.
    """
    print("Demonstrating the definition of derivatives...")
    
    # Example function: f(x) = x²
    def f(x):
        """Function to analyze: f(x) = x²"""
        return x**2
    
    # Numerical derivative using finite differences
    def numerical_derivative(f, x, h=1e-7):
        """
        Compute numerical derivative using central difference method.
        This is more accurate than forward difference for most functions.
        """
        return (f(x + h) - f(x - h)) / (2 * h)
    
    def forward_difference(f, x, h):
        """Forward difference approximation: [f(x+h) - f(x)]/h"""
        return (f(x + h) - f(x)) / h
    
    def backward_difference(f, x, h):
        """Backward difference approximation: [f(x) - f(x-h)]/h"""
        return (f(x) - f(x - h)) / h
    
    def central_difference(f, x, h):
        """Central difference approximation: [f(x+h) - f(x-h)]/(2h)"""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    # Symbolic derivative using SymPy
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
    
    # Visual demonstration
    x_vals = np.linspace(-2, 2, 100)
    y_vals = [f(x) for x in x_vals]
    
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
    numerical_derivatives = [numerical_derivative(f, x) for x in x_vals]
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
        numerical_val = numerical_derivative(f, x_test, h_optimal)
        symbolic_val = 2 * x_test
        error = abs(numerical_val - symbolic_val)
        print(f"x = {x_test}: numerical = {numerical_val:.10f}, symbolic = {symbolic_val:.10f}, error = {error:.2e}")

demonstrate_derivative_definition()

# =============================================================================
# SECTION 2: DIFFERENTIATION RULES
# =============================================================================

print("\n2. DIFFERENTIATION RULES")
print("-" * 30)

def demonstrate_differentiation_rules():
    """
    Demonstrate the fundamental differentiation rules.
    
    These rules form the foundation for computing derivatives
    of complex functions by breaking them down into simpler parts.
    """
    print("Demonstrating differentiation rules...")
    
    # Basic rules demonstration
    print("\n=== BASIC DIFFERENTIATION RULES ===\n")
    
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
    
    # Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)
    print("\n7. CHAIN RULE")
    chain_expr = sp.sin(x**2)
    chain_deriv = sp.diff(chain_expr, x)
    print(f"   d/dx({chain_expr}) = {chain_deriv}")
    
    # Visual demonstration of rules
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
    
    ax2.plot(x_vals, y2, 'b-', linewidth=2, label='f(x) = x² sin(x)')
    ax2.plot(x_vals, dy2, 'r--', linewidth=2, label='f\'(x) = 2x sin(x) + x² cos(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Product Rule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Chain rule visualization
    y3 = np.sin(x_vals**2)
    dy3 = 2 * x_vals * np.cos(x_vals**2)
    
    ax3.plot(x_vals, y3, 'b-', linewidth=2, label='f(x) = sin(x²)')
    ax3.plot(x_vals, dy3, 'r--', linewidth=2, label='f\'(x) = 2x cos(x²)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Chain Rule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Quotient rule visualization
    y4 = x_vals**2 / (x_vals + 1)
    dy4 = (2 * x_vals * (x_vals + 1) - x_vals**2) / (x_vals + 1)**2
    
    ax4.plot(x_vals, y4, 'b-', linewidth=2, label='f(x) = x²/(x+1)')
    ax4.plot(x_vals, dy4, 'r--', linewidth=2, label='f\'(x)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Quotient Rule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

demonstrate_differentiation_rules()

# =============================================================================
# SECTION 3: HIGHER-ORDER DERIVATIVES
# =============================================================================

print("\n3. HIGHER-ORDER DERIVATIVES")
print("-" * 30)

def demonstrate_higher_order_derivatives():
    """
    Demonstrate higher-order derivatives and their applications.
    
    Higher-order derivatives provide information about:
    - Rate of change of rate of change (second derivative)
    - Concavity and inflection points
    - Taylor series approximations
    """
    print("Demonstrating higher-order derivatives...")
    
    # Example function: f(x) = x³ - 3x
    f_expr = x**3 - 3*x
    
    # Compute derivatives symbolically
    f_prime = sp.diff(f_expr, x)
    f_double_prime = sp.diff(f_prime, x)
    f_triple_prime = sp.diff(f_double_prime, x)
    
    print(f"Original function: f(x) = {f_expr}")
    print(f"First derivative: f'(x) = {f_prime}")
    print(f"Second derivative: f''(x) = {f_double_prime}")
    print(f"Third derivative: f'''(x) = {f_triple_prime}")
    
    # Visual demonstration
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = x_vals**3 - 3*x_vals
    dy_vals = 3*x_vals**2 - 3
    d2y_vals = 6*x_vals
    d3y_vals = 6 * np.ones_like(x_vals)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original function
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 3x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Original Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # First derivative
    ax2.plot(x_vals, dy_vals, 'r-', linewidth=2, label='f\'(x) = 3x² - 3')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f\'(x)')
    ax2.set_title('First Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Second derivative
    ax3.plot(x_vals, d2y_vals, 'g-', linewidth=2, label='f\'\'(x) = 6x')
    ax3.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('f\'\'(x)')
    ax3.set_title('Second Derivative')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Third derivative
    ax4.plot(x_vals, d3y_vals, 'purple', linewidth=2, label='f\'\'\'(x) = 6')
    ax4.set_xlabel('x')
    ax4.set_ylabel('f\'\'\'(x)')
    ax4.set_title('Third Derivative')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Critical points analysis
    print("\nCritical Points Analysis:")
    print("First derivative zeroes (critical points):")
    critical_points = sp.solve(f_prime, x)
    for point in critical_points:
        print(f"  x = {point}")
        # Evaluate second derivative at critical points
        d2_at_point = f_double_prime.subs(x, point)
        print(f"    f''({point}) = {d2_at_point}")
        if d2_at_point > 0:
            print(f"    → Local minimum")
        elif d2_at_point < 0:
            print(f"    → Local maximum")
        else:
            print(f"    → Need higher derivatives or other test")

demonstrate_higher_order_derivatives()

# =============================================================================
# SECTION 4: APPLICATIONS IN MACHINE LEARNING
# =============================================================================

print("\n4. APPLICATIONS IN MACHINE LEARNING")
print("-" * 40)

def demonstrate_ml_applications():
    """
    Demonstrate how derivatives are used in machine learning.
    """
    print("Demonstrating machine learning applications...")
    
    # 1. Gradient Descent Optimization
    def gradient_descent_demo():
        """
        Demonstrate gradient descent optimization.
        """
        print("\nGradient Descent Optimization:")
        
        # Objective function: f(x) = x² + 2x + 1
        def objective_function(x):
            return x**2 + 2*x + 1
        
        def gradient(x):
            return 2*x + 2
        
        # Gradient descent
        x_current = 3.0
        learning_rate = 0.1
        iterations = 20
        trajectory = [x_current]
        
        print(f"Starting point: x = {x_current}")
        print(f"Learning rate: α = {learning_rate}")
        print(f"Objective function: f(x) = x² + 2x + 1")
        print(f"Gradient: f'(x) = 2x + 2")
        print("\nIteration\tx\t\tf(x)\t\tf'(x)")
        print("-" * 50)
        
        for i in range(iterations):
            grad = gradient(x_current)
            x_new = x_current - learning_rate * grad
            trajectory.append(x_new)
            
            if i % 5 == 0:
                print(f"{i}\t\t{x_current:.6f}\t{objective_function(x_current):.6f}\t{grad:.6f}")
            
            x_current = x_new
        
        print(f"{iterations}\t\t{x_current:.6f}\t{objective_function(x_current):.6f}\t{gradient(x_current):.6f}")
        
        # Visualization
        x_vals = np.linspace(-2, 4, 1000)
        y_vals = objective_function(x_vals)
        
        plt.figure(figsize=(12, 8))
        
        # Function and trajectory
        plt.subplot(2, 1, 1)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
        plt.plot(trajectory, [objective_function(x) for x in trajectory], 'ro-', linewidth=1, markersize=4, label='Gradient Descent')
        plt.axvline(x=-1, color='g', linestyle='--', linewidth=2, label='Optimal point x = -1')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gradient Descent Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convergence plot
        plt.subplot(2, 1, 2)
        plt.plot(range(len(trajectory)), trajectory, 'ro-', linewidth=1, markersize=4)
        plt.axhline(y=-1, color='g', linestyle='--', linewidth=2, label='Optimal value')
        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.title('Convergence to Optimal Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    gradient_descent_demo()
    
    # 2. Neural Network Backpropagation
    def neural_network_demo():
        """
        Demonstrate derivatives in neural network backpropagation.
        """
        print("\nNeural Network Backpropagation:")
        
        # Simple neural network: y = σ(wx + b)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        # Forward pass
        w = 2.0
        b = 1.0
        x_input = 0.5
        
        z = w * x_input + b
        y_output = sigmoid(z)
        
        print(f"Forward pass:")
        print(f"  Input: x = {x_input}")
        print(f"  Weights: w = {w}, b = {b}")
        print(f"  Linear combination: z = wx + b = {w}*{x_input} + {b} = {z}")
        print(f"  Output: y = σ(z) = {y_output:.6f}")
        
        # Backward pass (gradient computation)
        target = 0.8
        loss = 0.5 * (y_output - target)**2
        
        # Chain rule for derivatives
        dL_dy = y_output - target  # Derivative of loss w.r.t. output
        dy_dz = sigmoid_derivative(z)  # Derivative of sigmoid
        dz_dw = x_input  # Derivative of linear combination w.r.t. w
        dz_db = 1.0  # Derivative of linear combination w.r.t. b
        
        # Total derivatives using chain rule
        dL_dw = dL_dy * dy_dz * dz_dw
        dL_db = dL_dy * dy_dz * dz_db
        
        print(f"\nBackward pass (gradient computation):")
        print(f"  Target: {target}")
        print(f"  Loss: L = 0.5(y - target)² = {loss:.6f}")
        print(f"  ∂L/∂y = y - target = {dL_dy:.6f}")
        print(f"  ∂y/∂z = σ'(z) = {dy_dz:.6f}")
        print(f"  ∂z/∂w = x = {dz_dw}")
        print(f"  ∂z/∂b = 1")
        print(f"  ∂L/∂w = ∂L/∂y * ∂y/∂z * ∂z/∂w = {dL_dw:.6f}")
        print(f"  ∂L/∂b = ∂L/∂y * ∂y/∂z * ∂z/∂b = {dL_db:.6f}")
        
        # Visualization
        x_vals = np.linspace(-5, 5, 1000)
        y_sigmoid = sigmoid(x_vals)
        y_deriv = sigmoid_derivative(x_vals)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sigmoid function
        ax1.plot(x_vals, y_sigmoid, 'b-', linewidth=2, label='σ(x)')
        ax1.scatter(z, y_output, color='red', s=100, zorder=5, label=f'Current point: ({z:.2f}, {y_output:.3f})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('σ(x)')
        ax1.set_title('Sigmoid Activation Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sigmoid derivative
        ax2.plot(x_vals, y_deriv, 'g-', linewidth=2, label='σ\'(x)')
        ax2.scatter(z, dy_dz, color='red', s=100, zorder=5, label=f'Derivative at z: {dy_dz:.3f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('σ\'(x)')
        ax2.set_title('Sigmoid Derivative')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    neural_network_demo()
    
    # 3. Loss Function Derivatives
    def loss_function_demo():
        """
        Demonstrate derivatives of common loss functions.
        """
        print("\nLoss Function Derivatives:")
        
        # Mean Squared Error
        def mse_loss(predictions, targets):
            return np.mean((predictions - targets)**2)
        
        def mse_derivative(predictions, targets):
            return 2 * (predictions - targets) / len(predictions)
        
        # Cross-entropy loss
        def cross_entropy_loss(predictions, targets):
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
        def cross_entropy_derivative(predictions, targets):
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -(targets / predictions - (1 - targets) / (1 - predictions)) / len(predictions)
        
        # Test data
        predictions = np.array([0.2, 0.7, 0.1, 0.9])
        targets = np.array([0, 1, 0, 1])
        
        print(f"Predictions: {predictions}")
        print(f"Targets: {targets}")
        
        mse_val = mse_loss(predictions, targets)
        mse_grad = mse_derivative(predictions, targets)
        
        ce_val = cross_entropy_loss(predictions, targets)
        ce_grad = cross_entropy_derivative(predictions, targets)
        
        print(f"\nMSE Loss: {mse_val:.4f}")
        print(f"MSE Gradient: {mse_grad}")
        print(f"Cross-Entropy Loss: {ce_val:.4f}")
        print(f"Cross-Entropy Gradient: {ce_grad}")
        
        # Visualization
        x_vals = np.linspace(0.01, 0.99, 100)
        target = 1.0
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE
        mse_vals = [(x - target)**2 for x in x_vals]
        mse_grad_vals = [2*(x - target) for x in x_vals]
        
        ax1.plot(x_vals, mse_vals, 'b-', linewidth=2, label='MSE Loss')
        ax1.plot(x_vals, mse_grad_vals, 'r--', linewidth=2, label='MSE Gradient')
        ax1.axvline(x=target, color='g', linestyle=':', linewidth=2, label='Target')
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('Loss/Gradient')
        ax1.set_title('Mean Squared Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cross-entropy
        ce_vals = [-target * np.log(x) - (1-target) * np.log(1-x) for x in x_vals]
        ce_grad_vals = [-target/x + (1-target)/(1-x) for x in x_vals]
        
        ax2.plot(x_vals, ce_vals, 'b-', linewidth=2, label='Cross-Entropy Loss')
        ax2.plot(x_vals, ce_grad_vals, 'r--', linewidth=2, label='Cross-Entropy Gradient')
        ax2.axvline(x=target, color='g', linestyle=':', linewidth=2, label='Target')
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Loss/Gradient')
        ax2.set_title('Cross-Entropy Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    loss_function_demo()

demonstrate_ml_applications()

# =============================================================================
# SECTION 5: ADVANCED CONCEPTS
# =============================================================================

print("\n5. ADVANCED CONCEPTS")
print("-" * 30)

def demonstrate_advanced_concepts():
    """
    Demonstrate advanced derivative concepts.
    """
    print("Demonstrating advanced concepts...")
    
    # 1. Implicit Differentiation
    def implicit_differentiation_demo():
        """
        Demonstrate implicit differentiation.
        """
        print("\nImplicit Differentiation:")
        print("Example: x² + y² = 25 (circle)")
        
        # Using SymPy for implicit differentiation
        x, y = symbols('x y')
        equation = x**2 + y**2 - 25
        
        # Differentiate both sides with respect to x
        # d/dx(x² + y²) = d/dx(25)
        # 2x + 2y * dy/dx = 0
        # dy/dx = -x/y
        
        print(f"Original equation: {equation} = 0")
        print(f"Differentiating both sides with respect to x:")
        print(f"  2x + 2y * dy/dx = 0")
        print(f"  dy/dx = -x/y")
        
        # Visual demonstration
        theta = np.linspace(0, 2*np.pi, 1000)
        r = 5
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        
        # Compute derivatives at specific points
        points = [(3, 4), (-3, 4), (0, 5), (5, 0)]
        
        plt.figure(figsize=(10, 8))
        plt.plot(x_circle, y_circle, 'b-', linewidth=2, label='x² + y² = 25')
        
        for x_point, y_point in points:
            if y_point != 0:
                slope = -x_point / y_point
                # Plot tangent line
                t = np.linspace(-2, 2, 100)
                tangent_x = x_point + t
                tangent_y = y_point + slope * t
                plt.plot(tangent_x, tangent_y, 'r--', linewidth=1, alpha=0.7)
                plt.scatter(x_point, y_point, color='red', s=100, zorder=5)
                plt.text(x_point + 0.2, y_point + 0.2, f'({x_point}, {y_point})\ndy/dx = {slope:.2f}', 
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Implicit Differentiation: Circle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    implicit_differentiation_demo()
    
    # 2. Related Rates
    def related_rates_demo():
        """
        Demonstrate related rates problems.
        """
        print("\nRelated Rates:")
        print("Example: A ladder sliding down a wall")
        
        # Ladder problem: 13-foot ladder sliding down a wall
        # When x = 5 feet from wall, dx/dt = 2 ft/s
        # Find dy/dt when x = 5
        
        # x² + y² = 13² (Pythagorean theorem)
        # Differentiating: 2x*dx/dt + 2y*dy/dt = 0
        # dy/dt = -x*dx/dt / y
        
        x = 5
        dx_dt = 2
        y = np.sqrt(13**2 - x**2)
        dy_dt = -x * dx_dt / y
        
        print(f"Ladder length: 13 feet")
        print(f"Distance from wall: x = {x} feet")
        print(f"Rate of change of x: dx/dt = {dx_dt} ft/s")
        print(f"Height on wall: y = √(13² - {x}²) = {y:.2f} feet")
        print(f"Rate of change of y: dy/dt = -x*dx/dt / y = {dy_dt:.2f} ft/s")
        
        # Visualization
        t_vals = np.linspace(0, 5, 100)
        x_vals = 5 + 2 * t_vals  # x = x₀ + dx/dt * t
        y_vals = np.sqrt(13**2 - x_vals**2)
        
        plt.figure(figsize=(12, 8))
        
        # Wall and ground
        plt.axvline(x=0, color='k', linewidth=2, label='Wall')
        plt.axhline(y=0, color='k', linewidth=2, label='Ground')
        
        # Ladder positions
        for i in range(0, len(t_vals), 10):
            x_pos = x_vals[i]
            y_pos = y_vals[i]
            plt.plot([0, x_pos], [y_pos, 0], 'b-', linewidth=2, alpha=0.7)
            plt.scatter(x_pos, 0, color='red', s=50, zorder=5)
            plt.scatter(0, y_pos, color='red', s=50, zorder=5)
        
        plt.xlabel('Distance from wall (feet)')
        plt.ylabel('Height on wall (feet)')
        plt.title('Ladder Sliding Down Wall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-1, 15)
        plt.ylim(-1, 15)
        plt.show()
    
    related_rates_demo()

demonstrate_advanced_concepts()

# =============================================================================
# SUMMARY AND CONCLUSION
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY AND CONCLUSION")
print("=" * 60)

print("""
This implementation has demonstrated the key concepts of derivatives:

1. DEFINITION OF DERIVATIVES
   - Limit definition and geometric interpretation
   - Numerical vs symbolic computation
   - Tangent lines and instantaneous rates of change

2. DIFFERENTIATION RULES
   - Power, constant, sum, product, quotient, and chain rules
   - Systematic approach to computing derivatives
   - Visual verification of rules

3. HIGHER-ORDER DERIVATIVES
   - Second and third derivatives
   - Applications in optimization and curve analysis
   - Critical points and inflection points

4. MACHINE LEARNING APPLICATIONS
   - Gradient descent optimization
   - Neural network backpropagation
   - Loss function derivatives
   - Training algorithms and convergence

5. ADVANCED CONCEPTS
   - Implicit differentiation
   - Related rates problems
   - Applications in physics and engineering

These concepts form the foundation for:
- Optimization algorithms in machine learning
- Neural network training and backpropagation
- Loss function design and analysis
- Algorithm convergence analysis
- Mathematical modeling and analysis
""")

print("\nAll concepts have been implemented with:")
print("- Clear mathematical explanations")
print("- Comprehensive visualizations")
print("- Practical machine learning applications")
print("- Numerical verification of theoretical results")
print("- Code annotations and documentation")

# 1. Limit Definition of Derivative (already present, but add more explicit annotation)
def demonstrate_limit_definition_of_derivative():
    """
    Demonstrate the limit definition of the derivative and compare forward, backward, and central differences.
    """
    print("\n--- Limit Definition of Derivative ---")
    def f(x): return x**3 - 2*x + 1
    x0 = 1.0
    h_vals = [1e-1, 1e-2, 1e-3, 1e-4]
    for h in h_vals:
        fwd = (f(x0 + h) - f(x0)) / h
        bwd = (f(x0) - f(x0 - h)) / h
        cent = (f(x0 + h) - f(x0 - h)) / (2*h)
        print(f"h={h:.0e}: forward={fwd:.6f}, backward={bwd:.6f}, central={cent:.6f}")
    # Symbolic
    f_expr = x**3 - 2*x + 1
    print("Symbolic derivative:", sp.diff(f_expr, x).subs(x, x0))

# 2. Differentiation Rules (expand with explicit examples)
def demonstrate_differentiation_rules_expanded():
    """
    Demonstrate power, product, quotient, and chain rules with explicit examples and symbolic/numeric checks.
    """
    print("\n--- Differentiation Rules ---")
    # Power rule
    print("Power rule: d/dx x^n = n*x^(n-1)")
    for n in [2, 3, 0.5, -1]:
        expr = x**n
        print(f"d/dx x^{n}: {sp.diff(expr, x)}")
    # Product rule
    print("\nProduct rule: d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)")
    f1, g1 = x**2, sp.sin(x)
    prod = f1 * g1
    print(f"d/dx[x^2*sin(x)] = {sp.diff(prod, x)}")
    # Quotient rule
    print("\nQuotient rule: d/dx[f(x)/g(x)] = (f'g - fg')/g^2")
    f2, g2 = sp.exp(x), x**2 + 1
    quot = f2 / g2
    print(f"d/dx[exp(x)/(x^2+1)] = {sp.diff(quot, x)}")
    # Chain rule
    print("\nChain rule: d/dx f(g(x)) = f'(g(x))*g'(x)")
    outer, inner = sp.sin, x**2 + 1
    chain = outer(inner)
    print(f"d/dx[sin(x^2+1)] = {sp.diff(chain, x)}")
    # Numeric check for chain rule
    def f(x): return np.sin(x**2 + 1)
    def f_prime(x): return np.cos(x**2 + 1) * 2 * x
    x0 = 1.0
    h = 1e-5
    num = (f(x0 + h) - f(x0 - h)) / (2*h)
    print(f"Numeric chain rule at x=1: {num:.6f}, analytic: {f_prime(x0):.6f}")

# 3. Higher-Order and Partial Derivatives
def demonstrate_higher_and_partial_derivatives():
    """
    Show higher-order and partial derivatives, with symbolic and numeric examples.
    """
    print("\n--- Higher-Order and Partial Derivatives ---")
    # Higher-order
    expr = x**4 - 3*x**2 + 2
    print("First derivative:", sp.diff(expr, x))
    print("Second derivative:", sp.diff(expr, x, 2))
    # Partial derivatives
    fxy = x**2 * y + sp.sin(x*y)
    print("\nFunction: f(x, y) = x^2*y + sin(xy)")
    print("∂f/∂x:", sp.diff(fxy, x))
    print("∂f/∂y:", sp.diff(fxy, y))
    # Numeric partials
    def f(x, y): return x**2 * y + np.sin(x*y)
    x0, y0 = 1.0, 2.0
    h = 1e-5
    dfdx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2*h)
    dfdy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2*h)
    print(f"Numeric ∂f/∂x at (1,2): {dfdx:.6f}, ∂f/∂y: {dfdy:.6f}")

# 4. Implicit Differentiation
def demonstrate_implicit_differentiation():
    """
    Show implicit differentiation for x^2 + y^2 = 1.
    """
    print("\n--- Implicit Differentiation ---")
    y_sym = sp.Function('y')(x)
    eq = x**2 + y_sym**2 - 1
    dydx = sp.diff(eq, x).subs(sp.Derivative(y_sym, x), sp.Symbol('yprime'))
    print("d/dx[x^2 + y^2 = 1]:", dydx)
    # Solve for dy/dx
    yprime = sp.solve(dydx, sp.Symbol('yprime'))[0]
    print("dy/dx =", yprime)

# 5. Applications in ML: Gradient Descent, Backprop, Sensitivity
def demonstrate_ml_derivative_applications():
    """
    Show gradient descent, backpropagation, and feature sensitivity using derivatives.
    """
    print("\n--- ML Applications: Gradient Descent, Backprop, Sensitivity ---")
    # Gradient descent for f(x) = (x-3)^2
    def f(x): return (x-3)**2
    def grad(x): return 2*(x-3)
    xvals = [0.0]
    for _ in range(20):
        xvals.append(xvals[-1] - 0.1*grad(xvals[-1]))
    plt.plot(xvals, label='x values')
    plt.title('Gradient Descent on f(x)=(x-3)^2'); plt.xlabel('Iteration'); plt.ylabel('x'); plt.legend(); plt.show()
    # Backpropagation: chain rule for simple NN
    def sigmoid(x): return 1/(1+np.exp(-x))
    def sigmoid_prime(x): return sigmoid(x)*(1-sigmoid(x))
    x0 = 0.5
    z = 2*x0 + 1
    a = sigmoid(z)
    loss = (a - 1)**2
    dloss_da = 2*(a-1)
    da_dz = sigmoid_prime(z)
    dz_dx = 2
    dloss_dx = dloss_da * da_dz * dz_dx
    print(f"Backprop example: dloss/dx = {dloss_dx:.6f}")
    # Feature sensitivity
    def model(x, y): return 2*x + 3*y
    print(f"Feature sensitivity: ∂model/∂x = {sp.diff(2*x+3*y, x)}, ∂model/∂y = {sp.diff(2*x+3*y, y)}")

# Run all demonstrations
if __name__ == "__main__":
    demonstrate_limit_definition_of_derivative()
    demonstrate_differentiation_rules_expanded()
    demonstrate_higher_and_partial_derivatives()
    demonstrate_implicit_differentiation()
    demonstrate_ml_derivative_applications()

