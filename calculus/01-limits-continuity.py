"""
Limits and Continuity - Python Implementation
============================================

This file demonstrates the key concepts of limits and continuity using Python.
It includes comprehensive examples, visualizations, and practical applications
in machine learning and data science.

Key Concepts Covered:
- Definition and computation of limits
- One-sided limits and their applications
- Continuity and types of discontinuities
- Limits at infinity and asymptotic behavior
- Applications in machine learning and optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, limit, simplify, oo
import warnings
warnings.filterwarnings('ignore')

# Set up SymPy symbols for symbolic computation
x, y, z = symbols('x y z')

print("=" * 60)
print("LIMITS AND CONTINUITY - PYTHON IMPLEMENTATION")
print("=" * 60)

# =============================================================================
# SECTION 1: DEFINITION OF LIMITS
# =============================================================================

print("\n1. DEFINITION OF LIMITS")
print("-" * 30)

def demonstrate_limit_definition():
    """
    Demonstrate the ε-δ definition of limits using a simple example.
    
    The limit lim_{x→a} f(x) = L means that for every ε > 0, 
    there exists a δ > 0 such that whenever 0 < |x - a| < δ, 
    we have |f(x) - L| < ε.
    """
    print("Demonstrating the ε-δ definition of limits...")
    
    # Example: lim_{x→2} (x² - 4)/(x - 2) = 4
    def f(x):
        """Function with removable discontinuity at x = 2"""
        return (x**2 - 4) / (x - 2) if x != 2 else 4
    
    # Symbolic computation using SymPy
    limit_expr = (x**2 - 4) / (x - 2)
    symbolic_limit = sp.limit(limit_expr, x, 2)
    print(f"Symbolic limit: lim_{{x→2}} (x² - 4)/(x - 2) = {symbolic_limit}")
    
    # Numerical verification
    print("\nNumerical verification:")
    for h in [0.1, 0.01, 0.001, 0.0001]:
        left_val = f(2 - h)
        right_val = f(2 + h)
        print(f"f(2-{h}) = {left_val:.6f}, f(2+{h}) = {right_val:.6f}")
    
    # Visual demonstration
    x_vals = np.linspace(1.5, 2.5, 1000)
    y_vals = [f(x) for x in x_vals if x != 2]
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x²-4)/(x-2)')
    plt.axhline(y=4, color='r', linestyle='--', linewidth=2, label='Limit = 4')
    plt.axvline(x=2, color='g', linestyle='--', linewidth=2, label='x = 2')
    plt.scatter(2, 4, color='red', s=100, zorder=5, label='Limit point')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Limit Example: (x²-4)/(x-2) as x → 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed view
    plt.subplot(2, 1, 2)
    x_zoom = np.linspace(1.9, 2.1, 200)
    y_zoom = [f(x) for x in x_zoom if x != 2]
    plt.plot(x_zoom, y_zoom, 'b-', linewidth=2)
    plt.axhline(y=4, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=2, color='g', linestyle='--', linewidth=2)
    plt.scatter(2, 4, color='red', s=100, zorder=5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Zoomed View Around x = 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

demonstrate_limit_definition()

# 1. Epsilon-Delta Definition (already present, but add more explicit annotation)
def demonstrate_epsilon_delta_definition():
    """
    Demonstrate the ε-δ definition of limits using a simple example.
    The limit lim_{x→a} f(x) = L means that for every ε > 0, there exists a δ > 0 such that whenever 0 < |x - a| < δ, we have |f(x) - L| < ε.
    """
    print("\n--- Epsilon-Delta Definition of Limit ---")
    # Example: lim_{x→1} (x^2 - 1)/(x - 1) = 2
    def f(x):
        return (x**2 - 1) / (x - 1) if x != 1 else 2
    limit_expr = (x**2 - 1) / (x - 1)
    symbolic_limit = sp.limit(limit_expr, x, 1)
    print(f"Symbolic limit: lim_{{x→1}} (x² - 1)/(x - 1) = {symbolic_limit}")
    # Epsilon-delta demonstration
    for epsilon in [0.1, 0.01, 0.001]:
        delta = epsilon
        print(f"For ε={epsilon}, δ={delta}")
        for test_x in [1 - delta/2, 1 + delta/2]:
            print(f"  f({test_x:.4f}) = {f(test_x):.6f}, |f(x)-2| = {abs(f(test_x)-2):.6f}")
    # Visual plot
    x_vals = np.linspace(0.5, 1.5, 400)
    y_vals = [f(x) for x in x_vals if x != 1]
    plt.figure(figsize=(8,4))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.scatter([1], [2], color='red', label='Removable discontinuity')
    plt.axhline(2, color='orange', linestyle='--', label='Limit')
    plt.title('Removable Discontinuity and Limit')
    plt.legend(); plt.show()

# =============================================================================
# SECTION 2: ONE-SIDED LIMITS
# =============================================================================

print("\n2. ONE-SIDED LIMITS")
print("-" * 30)

def demonstrate_one_sided_limits():
    """
    Demonstrate one-sided limits using step, sign, and ReLU functions.
    """
    print("\n--- One-Sided Limits ---")
    def step(x): return np.where(x < 0, -1, 1)
    def relu(x): return np.where(x < 0, 0, x)
    x_vals = np.linspace(-2, 2, 400)
    plt.plot(x_vals, step(x_vals), label='Step')
    plt.plot(x_vals, relu(x_vals), label='ReLU')
    plt.axvline(0, color='k', linestyle=':')
    plt.legend(); plt.title('Step and ReLU Functions'); plt.show()
    # Symbolic one-sided limits
    step_expr = sp.Piecewise((-1, x < 0), (1, True))
    print('Step left:', sp.limit(step_expr, x, 0, dir='-'), 'right:', sp.limit(step_expr, x, 0, dir='+'))
    relu_expr = sp.Piecewise((0, x < 0), (x, x >= 0))
    print('ReLU left:', sp.limit(relu_expr, x, 0, dir='-'), 'right:', sp.limit(relu_expr, x, 0, dir='+'))

# 2. One-Sided Limits (expand with ML context)
def demonstrate_one_sided_limits():
    """
    Demonstrate one-sided limits using piecewise functions.
    
    One-sided limits are crucial for understanding functions that behave
    differently from the left and right sides of a point.
    """
    print("Demonstrating one-sided limits...")
    
    # Define piecewise functions
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
    
    # Symbolic analysis
    print("\nSymbolic analysis of one-sided limits:")
    
    # Step function limits
    step_expr = sp.Piecewise((-1, x < 0), (1, True))
    left_limit = sp.limit(step_expr, x, 0, dir='-')
    right_limit = sp.limit(step_expr, x, 0, dir='+')
    print(f"Step function:")
    print(f"  Left limit: {left_limit}")
    print(f"  Right limit: {right_limit}")
    print(f"  Two-sided limit exists: {left_limit == right_limit}")
    
    # Numerical verification
    print("\nNumerical verification:")
    for h in [0.1, 0.01, 0.001]:
        left_val = step_function(-h)
        right_val = step_function(h)
        print(f"f(-{h}) = {left_val}, f({h}) = {right_val}")

demonstrate_one_sided_limits()

# 3. Continuity (expand with types and ML relevance)
def demonstrate_continuity_types():
    """
    Show continuous, removable, jump, and infinite discontinuity with plots and checks.
    """
    print("\n--- Continuity and Discontinuities ---")
    def cont(x): return x**2
    def removable(x): return (x**2 - 1)/(x - 1) if x != 1 else 2
    def jump(x): return np.where(x < 0, -1, np.where(x > 0, 1, 0))
    def infinite(x): return 1/x if x != 0 else np.nan
    x_vals = np.linspace(-2, 2, 400)
    plt.figure(figsize=(10,6))
    plt.subplot(221); plt.plot(x_vals, cont(x_vals)); plt.title('Continuous')
    plt.subplot(222); plt.plot([x for x in x_vals if x != 1], [removable(x) for x in x_vals if x != 1]); plt.scatter(1,2,color='r'); plt.title('Removable')
    plt.subplot(223); plt.plot(x_vals, jump(x_vals)); plt.title('Jump')
    plt.subplot(224); plt.plot([x for x in x_vals if x != 0], [infinite(x) for x in x_vals if x != 0]); plt.title('Infinite')
    plt.tight_layout(); plt.show()
    # Print checks
    for name, func, pt in [('Cont', cont, 0), ('Removable', removable, 1), ('Jump', jump, 0), ('Infinite', infinite, 0)]:
        try:
            val = func(pt)
            left = func(pt-1e-4)
            right = func(pt+1e-4)
            print(f"{name} at {pt}: f({pt})={val}, left={left}, right={right}")
        except Exception as e:
            print(f"{name} at {pt}: Error {e}")

# =============================================================================
# SECTION 3: CONTINUITY
# =============================================================================

print("\n3. CONTINUITY")
print("-" * 30)

def demonstrate_continuity():
    """
    Demonstrate different types of continuity and discontinuities.
    
    A function f is continuous at a point a if:
    1. f(a) is defined
    2. lim_{x→a} f(x) exists
    3. lim_{x→a} f(x) = f(a)
    """
    print("Demonstrating continuity and discontinuities...")
    
    # Define different types of functions
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
    
    # Mathematical analysis
    print("\nMathematical analysis of continuity:")
    print("1. Continuous function: f(x) = x²")
    print(f"   f(0) = {continuous_func(0)}")
    print(f"   lim(x→0) f(x) = {sp.limit(x**2, x, 0)}")
    print(f"   Continuous at x = 0: {continuous_func(0) == sp.limit(x**2, x, 0)}")
    
    print("\n2. Removable discontinuity: f(x) = sin(x)/x")
    print(f"   f(0) = 1 (defined)")
    print(f"   lim(x→0) sin(x)/x = {sp.limit(sp.sin(x)/x, x, 0)}")
    print(f"   Continuous at x = 0: {1 == sp.limit(sp.sin(x)/x, x, 0)}")

demonstrate_continuity()

# 4. Limits at Infinity (expand with asymptotic behavior)
def demonstrate_limits_at_infinity():
    """
    Show limits at infinity and relate to algorithmic growth rates.
    """
    print("\n--- Limits at Infinity and Asymptotics ---")
    def f1(x): return 1/x
    def f2(x): return x**2/np.exp(x)
    def f3(x): return np.log(x)/x
    def f4(x): return (x**2 + 3*x + 1)/(x**2 + 1)
    x_vals = np.linspace(1, 20, 200)
    plt.figure(figsize=(10,6))
    plt.subplot(221); plt.plot(x_vals, [f1(x) for x in x_vals]); plt.title('1/x')
    plt.subplot(222); plt.plot(x_vals, [f2(x) for x in x_vals]); plt.title('x^2/e^x')
    plt.subplot(223); plt.plot(x_vals, [f3(x) for x in x_vals]); plt.title('ln(x)/x')
    plt.subplot(224); plt.plot(x_vals, [f4(x) for x in x_vals]); plt.title('(x^2+3x+1)/(x^2+1)')
    plt.tight_layout(); plt.show()

# =============================================================================
# SECTION 4: LIMITS AT INFINITY
# =============================================================================

print("\n4. LIMITS AT INFINITY")
print("-" * 30)

def demonstrate_limits_at_infinity():
    """
    Demonstrate limits at infinity and asymptotic behavior.
    
    Limits at infinity describe the long-term behavior of functions
    and are essential for understanding asymptotic behavior in algorithms.
    """
    print("Demonstrating limits at infinity...")
    
    # Common limits at infinity
    print("\nCommon limits at infinity:")
    
    # 1/x as x → ∞
    limit_1_over_x = sp.limit(1/x, x, oo)
    print(f"lim(x→∞) 1/x = {limit_1_over_x}")
    
    # x^n / e^x as x → ∞
    limit_power_over_exp = sp.limit(x**3 / sp.exp(x), x, oo)
    print(f"lim(x→∞) x³/e^x = {limit_power_over_exp}")
    
    # ln(x) / x as x → ∞
    limit_log_over_x = sp.limit(sp.log(x) / x, x, oo)
    print(f"lim(x→∞) ln(x)/x = {limit_log_over_x}")
    
    # Visual demonstration
    x_vals = np.linspace(1, 20, 1000)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1/x
    ax1.plot(x_vals, 1/x_vals, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='y = 0')
    ax1.set_title('f(x) = 1/x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # x³/e^x
    ax2.plot(x_vals, x_vals**3 / np.exp(x_vals), 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='y = 0')
    ax2.set_title('f(x) = x³/e^x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ln(x)/x
    ax3.plot(x_vals, np.log(x_vals) / x_vals, 'r-', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='y = 0')
    ax3.set_title('f(x) = ln(x)/x')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Growth rate comparison
    ax4.plot(x_vals, x_vals, 'b-', linewidth=2, label='x')
    ax4.plot(x_vals, x_vals**2, 'g-', linewidth=2, label='x²')
    ax4.plot(x_vals, np.exp(x_vals/5), 'r-', linewidth=2, label='e^(x/5)')
    ax4.set_title('Growth Rate Comparison')
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

demonstrate_limits_at_infinity()

# 5. Applications in AI/ML: Convergence and Loss Functions
def demonstrate_convergence_and_loss():
    """
    Show gradient descent convergence and loss function behavior as limits.
    """
    print("\n--- ML Applications: Convergence and Loss ---")
    # Gradient descent on f(x)=x^2
    def grad_desc(lr=0.1, iters=30, x0=2.0):
        xvals = [x0]
        for _ in range(iters):
            xvals.append(xvals[-1] - lr*2*xvals[-1])
        return xvals
    for lr in [0.1, 0.5, 1.0]:
        xs = grad_desc(lr)
        plt.plot(xs, label=f'lr={lr}')
    plt.title('Gradient Descent Convergence'); plt.xlabel('Iteration'); plt.ylabel('x'); plt.legend(); plt.show()
    # Loss function behavior
    def mse(pred, target): return np.mean((pred-target)**2)
    def ce(pred, target):
        pred = np.clip(pred, 1e-8, 1-1e-8)
        return -np.mean(target*np.log(pred)+(1-target)*np.log(1-pred))
    preds = np.linspace(0.01, 0.99, 100)
    plt.plot(preds, [mse(p,1) for p in preds], label='MSE (target=1)')
    plt.plot(preds, [ce(p,1) for p in preds], label='Cross-Entropy (target=1)')
    plt.title('Loss Function Behavior as Prediction→Target'); plt.xlabel('Prediction'); plt.ylabel('Loss'); plt.legend(); plt.show()

# =============================================================================
# SECTION 5: APPLICATIONS IN MACHINE LEARNING
# =============================================================================

print("\n5. APPLICATIONS IN MACHINE LEARNING")
print("-" * 40)

def demonstrate_ml_applications():
    """
    Demonstrate how limits and continuity concepts apply to machine learning.
    """
    print("Demonstrating machine learning applications...")
    
    # 1. Gradient Descent Convergence
    def gradient_descent_convergence(learning_rate=0.1, iterations=100, starting_point=2.0):
        """
        Demonstrate convergence analysis using gradient descent.
        
        This shows how limits help us understand whether optimization
        algorithms will converge to a solution.
        """
        print(f"\nGradient Descent Convergence Analysis:")
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations: {iterations}")
        print(f"Starting point: {starting_point}")
        
        # Simple quadratic function: f(x) = x²
        def f(x): return x**2
        def f_prime(x): return 2*x
        
        # Gradient descent
        x = starting_point
        trajectory = [x]
        
        for i in range(iterations):
            x = x - learning_rate * f_prime(x)
            trajectory.append(x)
            
            if i % 20 == 0:
                print(f"Iteration {i}: x = {x:.6f}, f(x) = {f(x):.6f}")
        
        print(f"Final: x = {x:.6f}, f(x) = {f(x):.6f}")
        print(f"Limit: x → 0, f(x) → 0")
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        # Function and trajectory
        x_vals = np.linspace(-3, 3, 1000)
        y_vals = f(x_vals)
        
        plt.subplot(1, 2, 1)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
        plt.plot(trajectory, [f(x) for x in trajectory], 'ro-', linewidth=1, markersize=4, label='Gradient Descent')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gradient Descent on f(x) = x²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convergence plot
        plt.subplot(1, 2, 2)
        plt.plot(range(len(trajectory)), trajectory, 'ro-', linewidth=1, markersize=4)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Optimal value')
        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.title('Convergence to Optimal Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    gradient_descent_convergence()
    
    # 2. Loss Function Analysis
    def analyze_loss_functions():
        """
        Analyze different loss functions and their properties.
        """
        print(f"\nLoss Function Analysis:")
        
        # Define loss functions
        def mse_loss(predictions, targets):
            """Mean Squared Error loss"""
            return np.mean((predictions - targets)**2)
        
        def cross_entropy_loss(predictions, targets):
            """Cross-entropy loss for binary classification"""
            epsilon = 1e-15  # Avoid log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
        def hinge_loss(predictions, targets):
            """Hinge loss for support vector machines"""
            return np.mean(np.maximum(0, 1 - targets * predictions))
        
        # Test data
        targets = np.array([0, 1, 0, 1])
        predictions = np.array([0.1, 0.8, 0.3, 0.9])
        
        print(f"Targets: {targets}")
        print(f"Predictions: {predictions}")
        print(f"MSE Loss: {mse_loss(predictions, targets):.4f}")
        print(f"Cross-Entropy Loss: {cross_entropy_loss(predictions, targets):.4f}")
        print(f"Hinge Loss: {hinge_loss(predictions, targets):.4f}")
        
        # Visualize loss functions
        x_vals = np.linspace(0.01, 0.99, 100)
        target = 1.0
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # MSE
        mse_vals = [(x - target)**2 for x in x_vals]
        ax1.plot(x_vals, mse_vals, 'b-', linewidth=2)
        ax1.axvline(x=target, color='r', linestyle='--', linewidth=2, label='Target')
        ax1.set_title('Mean Squared Error')
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cross-Entropy
        ce_vals = [-target * np.log(x) - (1-target) * np.log(1-x) for x in x_vals]
        ax2.plot(x_vals, ce_vals, 'g-', linewidth=2)
        ax2.axvline(x=target, color='r', linestyle='--', linewidth=2, label='Target')
        ax2.set_title('Cross-Entropy Loss')
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Hinge
        hinge_vals = [max(0, 1 - target * x) for x in x_vals]
        ax3.plot(x_vals, hinge_vals, 'r-', linewidth=2)
        ax3.axvline(x=target, color='r', linestyle='--', linewidth=2, label='Target')
        ax3.set_title('Hinge Loss')
        ax3.set_xlabel('Prediction')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    analyze_loss_functions()
    
    # 3. Numerical Stability
    def demonstrate_numerical_stability():
        """
        Demonstrate numerical stability issues and solutions.
        """
        print(f"\nNumerical Stability Analysis:")
        
        def unstable_division(x, y):
            """Unstable division that can cause issues"""
            return x / y
        
        def stable_division(x, y, epsilon=1e-15):
            """Stable division with protection against division by zero"""
            return x / (y + epsilon)
        
        # Test cases
        test_cases = [
            (1.0, 1e-10),
            (1.0, 1e-15),
            (1.0, 0.0)
        ]
        
        print("Division stability test:")
        for x, y in test_cases:
            try:
                unstable_result = unstable_division(x, y)
                print(f"Unstable: {x} / {y} = {unstable_result}")
            except:
                print(f"Unstable: {x} / {y} = Error (division by zero)")
            
            stable_result = stable_division(x, y)
            print(f"Stable:   {x} / {y} = {stable_result}")
            print()
    
    demonstrate_numerical_stability()

demonstrate_ml_applications()

# 6. Advanced Concepts (Squeeze Theorem, L'Hôpital's Rule)
def demonstrate_advanced_concepts():
    """
    Demonstrate advanced concepts like the squeeze theorem and L'Hôpital's rule.
    """
    print("Demonstrating advanced concepts...")
    
    # Squeeze Theorem
    def squeeze_theorem_demo():
        """
        Demonstrate the squeeze theorem using a classic example.
        """
        print("\nSqueeze Theorem Demonstration:")
        print("If g(x) ≤ f(x) ≤ h(x) for all x near a, and")
        print("lim_{x→a} g(x) = lim_{x→a} h(x) = L,")
        print("then lim_{x→a} f(x) = L")
        
        # Example: lim_{x→0} x² sin(1/x)
        x_vals = np.linspace(-0.1, 0.1, 1000)
        x_vals = x_vals[x_vals != 0]  # Remove zero
        
        # Define functions
        def f(x): return x**2 * np.sin(1/x)
        def g(x): return -x**2  # Lower bound
        def h(x): return x**2   # Upper bound
        
        y_f = [f(x) for x in x_vals]
        y_g = [g(x) for x in x_vals]
        y_h = [h(x) for x in x_vals]
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_vals, y_f, 'b-', linewidth=2, label='f(x) = x² sin(1/x)')
        plt.plot(x_vals, y_g, 'r--', linewidth=2, label='g(x) = -x²')
        plt.plot(x_vals, y_h, 'g--', linewidth=2, label='h(x) = x²')
        plt.axhline(y=0, color='k', linestyle=':', linewidth=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Squeeze Theorem: x² sin(1/x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Symbolic verification
        limit_result = sp.limit(x**2 * sp.sin(1/x), x, 0)
        print(f"Symbolic limit: lim_{{x→0}} x² sin(1/x) = {limit_result}")
    
    squeeze_theorem_demo()
    
    # L'Hôpital's Rule
    def lhopital_rule_demo():
        """
        Demonstrate L'Hôpital's rule for indeterminate forms.
        """
        print("\nL'Hôpital's Rule Demonstration:")
        print("If lim_{x→a} f(x) = 0 and lim_{x→a} g(x) = 0,")
        print("and lim_{x→a} f'(x)/g'(x) exists, then")
        print("lim_{x→a} f(x)/g(x) = lim_{x→a} f'(x)/g'(x)")
        
        # Example: lim_{x→0} sin(x)/x
        f_expr = sp.sin(x)
        g_expr = x
        
        # Direct substitution gives 0/0
        print(f"Direct substitution: lim_{{x→0}} sin(x)/x = 0/0 (indeterminate)")
        
        # Using L'Hôpital's rule
        f_prime = sp.diff(f_expr, x)
        g_prime = sp.diff(g_expr, x)
        
        print(f"f'(x) = {f_prime}")
        print(f"g'(x) = {g_prime}")
        
        limit_lhopital = sp.limit(f_prime / g_prime, x, 0)
        print(f"Using L'Hôpital's rule: lim_{{x→0}} cos(x)/1 = {limit_lhopital}")
        
        # Numerical verification
        x_vals = np.linspace(-0.1, 0.1, 1000)
        x_vals = x_vals[x_vals != 0]
        
        y_vals = [np.sin(x)/x for x in x_vals]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='sin(x)/x')
        plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Limit = 1')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('L\'Hôpital\'s Rule: sin(x)/x')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    lhopital_rule_demo()

demonstrate_advanced_concepts()

# =============================================================================
# SUMMARY AND CONCLUSION
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY AND CONCLUSION")
print("=" * 60)

print("""
This implementation has demonstrated the key concepts of limits and continuity:

1. DEFINITION OF LIMITS
   - ε-δ definition and its practical interpretation
   - Numerical and symbolic computation of limits
   - Visual verification using plots

2. ONE-SIDED LIMITS
   - Understanding piecewise functions
   - Left and right limits
   - Applications in machine learning activation functions

3. CONTINUITY
   - Three conditions for continuity
   - Types of discontinuities (removable, jump, infinite)
   - Importance for gradient-based optimization

4. LIMITS AT INFINITY
   - Asymptotic behavior
   - Growth rate comparisons
   - Applications in algorithm complexity

5. MACHINE LEARNING APPLICATIONS
   - Gradient descent convergence analysis
   - Loss function properties
   - Numerical stability considerations

6. ADVANCED CONCEPTS
   - Squeeze theorem for difficult limits
   - L'Hôpital's rule for indeterminate forms

These concepts form the foundation for understanding:
- Optimization algorithms in machine learning
- Convergence analysis of iterative methods
- Numerical stability in computations
- Model behavior and performance analysis
""")

print("\nAll concepts have been implemented with:")
print("- Clear mathematical explanations")
print("- Comprehensive visualizations")
print("- Practical machine learning applications")
print("- Numerical verification of theoretical results")
print("- Code annotations and documentation")

