"""
Derivative Applications - Python Implementation
=============================================

This file demonstrates the key applications of derivatives using Python.
It includes comprehensive examples, visualizations, and practical applications
in optimization, economics, physics, and machine learning.

Key Concepts Covered:
- Curve sketching and function analysis
- Optimization problems and techniques
- Related rates problems
- Economic applications (marginal analysis)
- Motion analysis and physics applications
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, solve, limit, oo
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set up SymPy symbols for symbolic computation
x, y, z, t = symbols('x y z t')

print("=" * 60)
print("DERIVATIVE APPLICATIONS - PYTHON IMPLEMENTATION")
print("=" * 60)

# =============================================================================
# SECTION 1: CURVE SKETCHING AND FUNCTION ANALYSIS
# =============================================================================

print("\n1. CURVE SKETCHING AND FUNCTION ANALYSIS")
print("-" * 40)

def demonstrate_curve_sketching():
    """
    Demonstrate comprehensive curve sketching using derivatives.
    
    This involves analyzing:
    - Critical points (where f'(x) = 0)
    - Inflection points (where f''(x) = 0)
    - Concavity and increasing/decreasing behavior
    - Asymptotes and end behavior
    """
    print("Demonstrating curve sketching techniques...")
    
    # Example function: f(x) = x³ - 6x² + 9x + 1
    f_expr = x**3 - 6*x**2 + 9*x + 1
    f_prime = sp.diff(f_expr, x)
    f_double_prime = sp.diff(f_prime, x)
    
    print(f"Function: f(x) = {f_expr}")
    print(f"First derivative: f'(x) = {f_prime}")
    print(f"Second derivative: f''(x) = {f_double_prime}")
    
    # Find critical points (solve f'(x) = 0)
    critical_points = sp.solve(f_prime, x)
    print(f"\nCritical points: {critical_points}")
    
    # Classify critical points using the second derivative test
    print("\nCritical point classification:")
    for point in critical_points:
        second_deriv = f_double_prime.subs(x, point)
        first_deriv = f_prime.subs(x, point)
        func_value = f_expr.subs(x, point)
        
        print(f"x = {point}:")
        print(f"  f({point}) = {func_value}")
        print(f"  f'({point}) = {first_deriv}")
        print(f"  f''({point}) = {second_deriv}")
        
        if second_deriv > 0:
            print(f"  → Local minimum (concave up)")
        elif second_deriv < 0:
            print(f"  → Local maximum (concave down)")
        else:
            print(f"  → Need first derivative test or higher derivatives")
    
    # Find inflection points (solve f''(x) = 0)
    inflection_points = sp.solve(f_double_prime, x)
    print(f"\nInflection points: {inflection_points}")
    
    # Analyze intervals of increase/decrease and concavity
    print("\nInterval analysis:")
    
    # Test points in different intervals
    test_points = [-1, 0, 1, 2, 3, 4, 5]
    for test_point in test_points:
        first_deriv = f_prime.subs(x, test_point)
        second_deriv = f_double_prime.subs(x, test_point)
        
        behavior = "increasing" if first_deriv > 0 else "decreasing"
        concavity = "concave up" if second_deriv > 0 else "concave down"
        
        print(f"x = {test_point}: {behavior}, {concavity}")
    
    # Visual demonstration
    x_vals = np.linspace(-1, 5, 1000)
    y_vals = [f_expr.subs(x, val) for val in x_vals]
    dy_vals = [f_prime.subs(x, val) for val in x_vals]
    ddy_vals = [f_double_prime.subs(x, val) for val in x_vals]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original function with critical and inflection points
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 6x² + 9x + 1')
    
    # Mark critical points
    for point in critical_points:
        y_point = f_expr.subs(x, point)
        ax1.scatter(point, y_point, c='red', s=100, zorder=5, label='Critical Point' if point == critical_points[0] else "")
    
    # Mark inflection points
    for point in inflection_points:
        y_point = f_expr.subs(x, point)
        ax1.scatter(point, y_point, c='green', s=100, marker='s', zorder=5, label='Inflection Point' if point == inflection_points[0] else "")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function with Critical and Inflection Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # First derivative
    ax2.plot(x_vals, dy_vals, 'r-', linewidth=2, label="f'(x)")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in critical_points:
        ax2.scatter(point, 0, c='red', s=100, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title('First Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Second derivative
    ax3.plot(x_vals, ddy_vals, 'g-', linewidth=2, label="f''(x)")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in inflection_points:
        second_deriv = f_double_prime.subs(x, point)
        ax3.scatter(point, second_deriv, c='green', s=100, marker='s', zorder=5)
    ax3.set_xlabel('x')
    ax3.set_ylabel("f''(x)")
    ax3.set_title('Second Derivative')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Concavity analysis
    ax4.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    
    # Color regions by concavity
    for i in range(len(x_vals)-1):
        if ddy_vals[i] > 0:
            ax4.fill_between(x_vals[i:i+2], y_vals[i:i+2], alpha=0.3, color='green')
        else:
            ax4.fill_between(x_vals[i:i+2], y_vals[i:i+2], alpha=0.3, color='red')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Concavity Analysis (Green: Concave Up, Red: Concave Down)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

demonstrate_curve_sketching()

# 1. Enhanced Curve Sketching with Asymptotes and End Behavior
def demonstrate_asymptotes_and_end_behavior():
    """
    Show how to find vertical, horizontal, and slant asymptotes, plus end behavior analysis.
    """
    print("\n--- Asymptotes and End Behavior ---")
    # Rational function: f(x) = (x^2 + 2x + 1)/(x - 1)
    f_expr = (x**2 + 2*x + 1)/(x - 1)
    print(f"Function: f(x) = {f_expr}")
    # Vertical asymptote: x = 1 (denominator = 0)
    print("Vertical asymptote: x = 1")
    # Horizontal asymptote: degree comparison
    print("Horizontal asymptote: y = x + 3 (slant, since deg(num) = deg(den) + 1)")
    # End behavior
    print("End behavior: f(x) → ∞ as x → ∞, f(x) → -∞ as x → -∞")
    # Plot
    x_vals = np.linspace(-5, 5, 1000)
    x_vals = x_vals[x_vals != 1]  # Remove vertical asymptote
    y_vals = [(val**2 + 2*val + 1)/(val - 1) for val in x_vals]
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axvline(1, color='r', linestyle='--', label='Vertical asymptote')
    plt.axline((0, 3), slope=1, color='g', linestyle='--', label='Slant asymptote')
    plt.title('Rational Function with Asymptotes'); plt.legend(); plt.show()

# 2. Enhanced Optimization with Constraints
def demonstrate_constrained_optimization():
    """
    Show optimization with constraints using Lagrange multipliers and substitution.
    """
    print("\n--- Constrained Optimization ---")
    # Example: Maximize f(x,y) = xy subject to x + y = 10
    # Method 1: Substitution
    # y = 10 - x, so f(x) = x(10-x) = 10x - x^2
    def f(x): return 10*x - x**2
    def f_prime(x): return 10 - 2*x
    critical_point = 5  # f'(x) = 0 → x = 5
    print(f"Critical point: x = {critical_point}")
    print(f"Maximum value: f(5) = {f(5)}")
    # Method 2: Lagrange multipliers (symbolic)
    from sympy import symbols, solve
    x, y, lam = symbols('x y lam')
    # Lagrangian: L = xy + lam*(x + y - 10)
    # ∂L/∂x = y + lam = 0
    # ∂L/∂y = x + lam = 0
    # ∂L/∂lam = x + y - 10 = 0
    eq1 = y + lam
    eq2 = x + lam
    eq3 = x + y - 10
    solution = solve([eq1, eq2, eq3], [x, y, lam])
    print(f"Lagrange solution: {solution}")

# 3. Related Rates with ML Context
def demonstrate_related_rates_ml():
    """
    Show related rates problems in ML context: learning rate scheduling, convergence rates.
    """
    print("\n--- Related Rates in ML ---")
    # Learning rate scheduling: how fast should learning rate decay?
    def learning_rate_schedule(t, initial_lr=0.1, decay_rate=0.95):
        return initial_lr * (decay_rate ** t)
    def lr_derivative(t, initial_lr=0.1, decay_rate=0.95):
        return initial_lr * np.log(decay_rate) * (decay_rate ** t)
    t_vals = np.arange(0, 20)
    lr_vals = [learning_rate_schedule(t) for t in t_vals]
    lr_deriv_vals = [lr_derivative(t) for t in t_vals]
    plt.plot(t_vals, lr_vals, label='Learning Rate')
    plt.plot(t_vals, lr_deriv_vals, label='Rate of Change')
    plt.title('Learning Rate Scheduling'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.legend(); plt.show()
    # Convergence analysis
    def convergence_rate(iterations, initial_error=1.0, rate=0.9):
        return initial_error * (rate ** iterations)
    def convergence_derivative(iterations, initial_error=1.0, rate=0.9):
        return initial_error * np.log(rate) * (rate ** iterations)
    iter_vals = np.arange(0, 50)
    error_vals = [convergence_rate(i) for i in iter_vals]
    error_deriv_vals = [convergence_derivative(i) for i in iter_vals]
    plt.plot(iter_vals, error_vals, label='Error')
    plt.plot(iter_vals, error_deriv_vals, label='Rate of Convergence')
    plt.title('Convergence Analysis'); plt.xlabel('Iteration'); plt.ylabel('Error'); plt.legend(); plt.show()

# 4. Economic Applications with Marginal Analysis
def demonstrate_marginal_analysis():
    """
    Show marginal cost, revenue, and profit analysis with derivatives.
    """
    print("\n--- Marginal Analysis ---")
    # Cost function: C(q) = 100 + 10q + 0.1q^2
    def cost(q): return 100 + 10*q + 0.1*q**2
    def marginal_cost(q): return 10 + 0.2*q
    # Revenue function: R(q) = 50q - 0.05q^2
    def revenue(q): return 50*q - 0.05*q**2
    def marginal_revenue(q): return 50 - 0.1*q
    # Profit function: P(q) = R(q) - C(q)
    def profit(q): return revenue(q) - cost(q)
    def marginal_profit(q): return marginal_revenue(q) - marginal_cost(q)
    q_vals = np.linspace(0, 100, 100)
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.plot(q_vals, [cost(q) for q in q_vals], label='Cost'); plt.plot(q_vals, [marginal_cost(q) for q in q_vals], label='Marginal Cost'); plt.legend(); plt.title('Cost Analysis')
    plt.subplot(132); plt.plot(q_vals, [revenue(q) for q in q_vals], label='Revenue'); plt.plot(q_vals, [marginal_revenue(q) for q in q_vals], label='Marginal Revenue'); plt.legend(); plt.title('Revenue Analysis')
    plt.subplot(133); plt.plot(q_vals, [profit(q) for q in q_vals], label='Profit'); plt.plot(q_vals, [marginal_profit(q) for q in q_vals], label='Marginal Profit'); plt.legend(); plt.title('Profit Analysis')
    plt.tight_layout(); plt.show()
    # Optimal production level: where marginal profit = 0
    from scipy.optimize import fsolve
    optimal_q = fsolve(marginal_profit, 50)[0]
    print(f"Optimal production level: q = {optimal_q:.2f}")

# 5. ML Applications: Loss Landscape Analysis
def demonstrate_loss_landscape_analysis():
    """
    Show how derivatives help analyze loss landscapes and optimization behavior.
    """
    print("\n--- Loss Landscape Analysis ---")
    # Simple loss function: L(w) = (w - 2)^2 + 0.1*w^4
    def loss(w): return (w - 2)**2 + 0.1*w**4
    def loss_prime(w): return 2*(w - 2) + 0.4*w**3
    def loss_double_prime(w): return 2 + 1.2*w**2
    w_vals = np.linspace(-3, 5, 100)
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.plot(w_vals, [loss(w) for w in w_vals]); plt.title('Loss Function'); plt.xlabel('w'); plt.ylabel('Loss')
    plt.subplot(132); plt.plot(w_vals, [loss_prime(w) for w in w_vals]); plt.axhline(0, color='k', linestyle='--'); plt.title('Gradient'); plt.xlabel('w'); plt.ylabel('dL/dw')
    plt.subplot(133); plt.plot(w_vals, [loss_double_prime(w) for w in w_vals]); plt.title('Hessian'); plt.xlabel('w'); plt.ylabel('d²L/dw²')
    plt.tight_layout(); plt.show()
    # Critical points
    from scipy.optimize import fsolve
    critical_points = fsolve(loss_prime, [-1, 0, 3])
    print(f"Critical points: {critical_points}")
    for cp in critical_points:
        hessian = loss_double_prime(cp)
        print(f"w = {cp:.3f}: Hessian = {hessian:.3f} ({'min' if hessian > 0 else 'max'})")

# Run all demonstrations
if __name__ == "__main__":
    demonstrate_asymptotes_and_end_behavior()
    demonstrate_constrained_optimization()
    demonstrate_related_rates_ml()
    demonstrate_marginal_analysis()
    demonstrate_loss_landscape_analysis()

# =============================================================================
# SECTION 2: OPTIMIZATION PROBLEMS
# =============================================================================

print("\n2. OPTIMIZATION PROBLEMS")
print("-" * 30)

def demonstrate_optimization():
    """
    Demonstrate optimization problems using derivatives.
    
    Optimization involves finding maximum or minimum values
    of functions, often subject to constraints.
    """
    print("Demonstrating optimization techniques...")
    
    # 1. Rectangle optimization
    def rectangle_optimization():
        """
        Find the rectangle with maximum area given a fixed perimeter.
        """
        print("\nRectangle Optimization:")
        print("Find the rectangle with maximum area given perimeter P = 20")
        
        # Let x = width, y = height
        # Perimeter: 2x + 2y = 20 → y = 10 - x
        # Area: A = xy = x(10 - x) = 10x - x²
        
        def area_function(x):
            return x * (10 - x)
        
        def area_derivative(x):
            return 10 - 2*x
        
        # Find critical points
        critical_point = 5  # From A'(x) = 10 - 2x = 0
        
        print(f"Area function: A(x) = x(10-x) = 10x - x²")
        print(f"Derivative: A'(x) = 10 - 2x")
        print(f"Critical point: A'(x) = 0 → 10 - 2x = 0 → x = 5")
        print(f"Optimal width: x = {critical_point}")
        print(f"Optimal height: y = 10 - {critical_point} = {10 - critical_point}")
        print(f"Maximum area: A({critical_point}) = {area_function(critical_point)}")
        
        # Visualization
        x_vals = np.linspace(0, 10, 1000)
        y_vals = [area_function(x) for x in x_vals]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='A(x) = x(10-x)')
        plt.scatter(critical_point, area_function(critical_point), color='red', s=100, zorder=5, label='Maximum')
        plt.axvline(x=critical_point, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Width (x)')
        plt.ylabel('Area')
        plt.title('Rectangle Area Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show the optimal rectangle
        plt.subplot(2, 1, 2)
        rect = plt.Rectangle((0, 0), critical_point, 10-critical_point, 
                           fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(critical_point/2, (10-critical_point)/2, f'{critical_point} × {10-critical_point}', 
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        plt.xlim(-1, 11)
        plt.ylim(-1, 11)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Optimal Rectangle')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    rectangle_optimization()
    
    # 2. Distance optimization
    def distance_optimization():
        """
        Find the minimum distance from a point to a line.
        """
        print("\nDistance Optimization:")
        print("Find minimum distance from point (2, 3) to line y = 2x + 1")
        
        # Point: (2, 3)
        # Line: y = 2x + 1
        # Distance from point (x₀, y₀) to line ax + by + c = 0:
        # d = |ax₀ + by₀ + c| / √(a² + b²)
        
        # Convert line to standard form: 2x - y + 1 = 0
        # So a = 2, b = -1, c = 1
        # Point: (x₀, y₀) = (2, 3)
        
        a, b, c = 2, -1, 1
        x0, y0 = 2, 3
        
        distance = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
        
        print(f"Line in standard form: 2x - y + 1 = 0")
        print(f"Point: ({x0}, {y0})")
        print(f"Distance formula: d = |ax₀ + by₀ + c| / √(a² + b²)")
        print(f"d = |{a}×{x0} + {b}×{y0} + {c}| / √({a}² + {b}²)")
        print(f"d = |{a*x0 + b*y0 + c}| / √{a**2 + b**2}")
        print(f"d = {abs(a*x0 + b*y0 + c)} / {np.sqrt(a**2 + b**2)}")
        print(f"d = {distance:.4f}")
        
        # Visualization
        x_vals = np.linspace(-2, 4, 1000)
        y_line = 2*x_vals + 1
        
        plt.figure(figsize=(10, 8))
        plt.plot(x_vals, y_line, 'b-', linewidth=2, label='y = 2x + 1')
        plt.scatter(x0, y0, color='red', s=100, zorder=5, label=f'Point ({x0}, {y0})')
        
        # Find closest point on line
        # The closest point is where the perpendicular line intersects
        # Perpendicular slope = -1/2
        # Equation: y - 3 = (-1/2)(x - 2)
        # y = (-1/2)x + 1 + 3 = (-1/2)x + 4
        
        # Intersection: 2x + 1 = (-1/2)x + 4
        # 2x + (1/2)x = 4 - 1
        # (5/2)x = 3
        # x = 6/5 = 1.2
        
        closest_x = 6/5
        closest_y = 2*closest_x + 1
        
        plt.scatter(closest_x, closest_y, color='green', s=100, zorder=5, label='Closest point on line')
        plt.plot([x0, closest_x], [y0, closest_y], 'r--', linewidth=2, label=f'Distance = {distance:.4f}')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Minimum Distance from Point to Line')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
    distance_optimization()

demonstrate_optimization()

# =============================================================================
# SECTION 3: RELATED RATES PROBLEMS
# =============================================================================

print("\n3. RELATED RATES PROBLEMS")
print("-" * 30)

def demonstrate_related_rates():
    """
    Demonstrate related rates problems using derivatives.
    
    Related rates problems involve finding how one quantity
    changes with respect to time when another quantity is changing.
    """
    print("Demonstrating related rates problems...")
    
    # 1. Ladder sliding down a wall
    def ladder_problem():
        """
        A 10-foot ladder is sliding down a wall.
        When the bottom is 6 feet from the wall, it's moving at 2 ft/s.
        How fast is the top sliding down?
        """
        print("\nLadder Problem:")
        print("A 10-foot ladder is sliding down a wall.")
        print("When the bottom is 6 feet from the wall, it's moving at 2 ft/s.")
        print("How fast is the top sliding down?")
        
        # Variables
        ladder_length = 10  # feet
        x = 6  # distance from wall
        dx_dt = 2  # rate of change of x
        
        # Relationship: x² + y² = 10² (Pythagorean theorem)
        # Differentiate with respect to time: 2x(dx/dt) + 2y(dy/dt) = 0
        # Therefore: dy/dt = -(x/y)(dx/dt)
        
        y = np.sqrt(ladder_length**2 - x**2)
        dy_dt = -(x / y) * dx_dt
        
        print(f"\nGiven:")
        print(f"  Ladder length: {ladder_length} feet")
        print(f"  Distance from wall: x = {x} feet")
        print(f"  Rate of change of x: dx/dt = {dx_dt} ft/s")
        print(f"  Height on wall: y = √({ladder_length}² - {x}²) = {y:.2f} feet")
        print(f"\nUsing the relationship x² + y² = {ladder_length}²")
        print(f"Differentiating with respect to time:")
        print(f"  2x(dx/dt) + 2y(dy/dt) = 0")
        print(f"  dy/dt = -(x/y)(dx/dt)")
        print(f"  dy/dt = -({x}/{y:.2f})({dx_dt})")
        print(f"  dy/dt = {dy_dt:.2f} ft/s")
        
        # Visualization
        t_vals = np.linspace(0, 3, 100)
        x_vals = 6 + 2 * t_vals  # x = x₀ + dx/dt * t
        y_vals = np.sqrt(10**2 - x_vals**2)
        
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
    
    ladder_problem()
    
    # 2. Expanding sphere
    def expanding_sphere_problem():
        """
        A spherical balloon is being inflated.
        When the radius is 3 cm, it's increasing at 2 cm/s.
        How fast is the volume increasing?
        """
        print("\nExpanding Sphere Problem:")
        print("A spherical balloon is being inflated.")
        print("When the radius is 3 cm, it's increasing at 2 cm/s.")
        print("How fast is the volume increasing?")
        
        # Variables
        r = 3  # radius in cm
        dr_dt = 2  # rate of change of radius
        
        # Volume: V = (4/3)πr³
        # Differentiate: dV/dt = 4πr²(dr/dt)
        
        dV_dt = 4 * np.pi * r**2 * dr_dt
        
        print(f"\nGiven:")
        print(f"  Radius: r = {r} cm")
        print(f"  Rate of change of radius: dr/dt = {dr_dt} cm/s")
        print(f"\nVolume: V = (4/3)πr³")
        print(f"Differentiating with respect to time:")
        print(f"  dV/dt = 4πr²(dr/dt)")
        print(f"  dV/dt = 4π({r})²({dr_dt})")
        print(f"  dV/dt = 4π({r**2})({dr_dt})")
        print(f"  dV/dt = {dV_dt:.2f} cm³/s")
        
        # Visualization
        t_vals = np.linspace(0, 2, 100)
        r_vals = 3 + 2 * t_vals
        V_vals = (4/3) * np.pi * r_vals**3
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Radius over time
        ax1.plot(t_vals, r_vals, 'b-', linewidth=2, label='r(t) = 3 + 2t')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Radius (cm)')
        ax1.set_title('Radius vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume over time
        ax2.plot(t_vals, V_vals, 'r-', linewidth=2, label='V(t) = (4/3)πr³')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Volume (cm³)')
        ax2.set_title('Volume vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    expanding_sphere_problem()

demonstrate_related_rates()

# =============================================================================
# SECTION 4: ECONOMIC APPLICATIONS
# =============================================================================

print("\n4. ECONOMIC APPLICATIONS")
print("-" * 30)

def demonstrate_economic_applications():
    """
    Demonstrate economic applications of derivatives.
    
    Derivatives are crucial in economics for:
    - Marginal analysis (cost, revenue, profit)
    - Elasticity calculations
    - Optimization of economic functions
    """
    print("Demonstrating economic applications...")
    
    # 1. Marginal analysis
    def marginal_analysis():
        """
        Demonstrate marginal cost, revenue, and profit analysis.
        """
        print("\nMarginal Analysis:")
        print("Cost function: C(q) = 100 + 10q + 0.1q²")
        print("Revenue function: R(q) = 50q - 0.05q²")
        print("Profit function: P(q) = R(q) - C(q)")
        
        # Define functions
        def cost_function(q):
            return 100 + 10*q + 0.1*q**2
        
        def revenue_function(q):
            return 50*q - 0.05*q**2
        
        def profit_function(q):
            return revenue_function(q) - cost_function(q)
        
        # Derivatives (marginal functions)
        def marginal_cost(q):
            return 10 + 0.2*q
        
        def marginal_revenue(q):
            return 50 - 0.1*q
        
        def marginal_profit(q):
            return marginal_revenue(q) - marginal_cost(q)
        
        # Find optimal production level
        # Set marginal profit = 0
        # 50 - 0.1q - (10 + 0.2q) = 0
        # 40 - 0.3q = 0
        # q = 40/0.3 ≈ 133.33
        
        optimal_q = 40 / 0.3
        
        print(f"\nMarginal functions:")
        print(f"  Marginal Cost: C'(q) = 10 + 0.2q")
        print(f"  Marginal Revenue: R'(q) = 50 - 0.1q")
        print(f"  Marginal Profit: P'(q) = R'(q) - C'(q) = 40 - 0.3q")
        
        print(f"\nOptimal production level:")
        print(f"  Set P'(q) = 0: 40 - 0.3q = 0")
        print(f"  q = 40/0.3 = {optimal_q:.2f}")
        
        print(f"\nAt optimal level:")
        print(f"  Cost: C({optimal_q:.2f}) = {cost_function(optimal_q):.2f}")
        print(f"  Revenue: R({optimal_q:.2f}) = {revenue_function(optimal_q):.2f}")
        print(f"  Profit: P({optimal_q:.2f}) = {profit_function(optimal_q):.2f}")
        
        # Visualization
        q_vals = np.linspace(0, 200, 1000)
        C_vals = [cost_function(q) for q in q_vals]
        R_vals = [revenue_function(q) for q in q_vals]
        P_vals = [profit_function(q) for q in q_vals]
        MC_vals = [marginal_cost(q) for q in q_vals]
        MR_vals = [marginal_revenue(q) for q in q_vals]
        MP_vals = [marginal_profit(q) for q in q_vals]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total functions
        ax1.plot(q_vals, C_vals, 'r-', linewidth=2, label='C(q)')
        ax1.plot(q_vals, R_vals, 'g-', linewidth=2, label='R(q)')
        ax1.plot(q_vals, P_vals, 'b-', linewidth=2, label='P(q)')
        ax1.axvline(x=optimal_q, color='k', linestyle='--', alpha=0.5, label='Optimal q')
        ax1.set_xlabel('Quantity (q)')
        ax1.set_ylabel('Dollars')
        ax1.set_title('Total Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Marginal functions
        ax2.plot(q_vals, MC_vals, 'r-', linewidth=2, label='MC(q)')
        ax2.plot(q_vals, MR_vals, 'g-', linewidth=2, label='MR(q)')
        ax2.plot(q_vals, MP_vals, 'b-', linewidth=2, label='MP(q)')
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax2.axvline(x=optimal_q, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Quantity (q)')
        ax2.set_ylabel('Dollars per unit')
        ax2.set_title('Marginal Functions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Profit maximization
        ax3.plot(q_vals, P_vals, 'b-', linewidth=2, label='P(q)')
        ax3.scatter(optimal_q, profit_function(optimal_q), color='red', s=100, zorder=5, label='Maximum profit')
        ax3.set_xlabel('Quantity (q)')
        ax3.set_ylabel('Profit')
        ax3.set_title('Profit Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Marginal profit
        ax4.plot(q_vals, MP_vals, 'b-', linewidth=2, label='MP(q)')
        ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax4.scatter(optimal_q, 0, color='red', s=100, zorder=5, label='MP(q) = 0')
        ax4.set_xlabel('Quantity (q)')
        ax4.set_ylabel('Marginal Profit')
        ax4.set_title('Marginal Profit')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    marginal_analysis()

demonstrate_economic_applications()

# =============================================================================
# SECTION 5: MOTION ANALYSIS
# =============================================================================

print("\n5. MOTION ANALYSIS")
print("-" * 30)

def demonstrate_motion_analysis():
    """
    Demonstrate motion analysis using derivatives.
    
    In physics, derivatives are used to analyze:
    - Position, velocity, and acceleration
    - Projectile motion
    - Harmonic motion
    """
    print("Demonstrating motion analysis...")
    
    # 1. Position, velocity, and acceleration
    def motion_analysis():
        """
        Analyze the motion of an object with position function s(t) = t³ - 6t² + 9t.
        """
        print("\nMotion Analysis:")
        print("Position function: s(t) = t³ - 6t² + 9t")
        
        # Define functions
        def position(t):
            return t**3 - 6*t**2 + 9*t
        
        def velocity(t):
            return 3*t**2 - 12*t + 9
        
        def acceleration(t):
            return 6*t - 12
        
        # Find when object is at rest (velocity = 0)
        # 3t² - 12t + 9 = 0
        # t² - 4t + 3 = 0
        # (t - 1)(t - 3) = 0
        # t = 1 or t = 3
        
        rest_times = [1, 3]
        
        print(f"\nVelocity: v(t) = s'(t) = 3t² - 12t + 9")
        print(f"Acceleration: a(t) = v'(t) = 6t - 12")
        print(f"\nObject is at rest when v(t) = 0:")
        print(f"  3t² - 12t + 9 = 0")
        print(f"  t² - 4t + 3 = 0")
        print(f"  (t - 1)(t - 3) = 0")
        print(f"  t = {rest_times[0]} or t = {rest_times[1]}")
        
        print(f"\nAt t = {rest_times[0]}:")
        print(f"  Position: s({rest_times[0]}) = {position(rest_times[0])}")
        print(f"  Acceleration: a({rest_times[0]}) = {acceleration(rest_times[0])}")
        
        print(f"\nAt t = {rest_times[1]}:")
        print(f"  Position: s({rest_times[1]}) = {position(rest_times[1])}")
        print(f"  Acceleration: a({rest_times[1]}) = {acceleration(rest_times[1])}")
        
        # Visualization
        t_vals = np.linspace(0, 4, 1000)
        s_vals = [position(t) for t in t_vals]
        v_vals = [velocity(t) for t in t_vals]
        a_vals = [acceleration(t) for t in t_vals]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Position
        ax1.plot(t_vals, s_vals, 'b-', linewidth=2, label='s(t)')
        for t in rest_times:
            ax1.scatter(t, position(t), color='red', s=100, zorder=5)
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('Position')
        ax1.set_title('Position vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity
        ax2.plot(t_vals, v_vals, 'g-', linewidth=2, label='v(t)')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        for t in rest_times:
            ax2.scatter(t, 0, color='red', s=100, zorder=5)
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Acceleration
        ax3.plot(t_vals, a_vals, 'r-', linewidth=2, label='a(t)')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        for t in rest_times:
            ax3.scatter(t, acceleration(t), color='red', s=100, zorder=5)
        ax3.set_xlabel('Time (t)')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    motion_analysis()

demonstrate_motion_analysis()

# =============================================================================
# SUMMARY AND CONCLUSION
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY AND CONCLUSION")
print("=" * 60)

print("""
This implementation has demonstrated the key applications of derivatives:

1. CURVE SKETCHING AND FUNCTION ANALYSIS
   - Critical points and inflection points
   - Increasing/decreasing behavior
   - Concavity analysis
   - Complete function behavior

2. OPTIMIZATION PROBLEMS
   - Finding maximum and minimum values
   - Constrained optimization
   - Geometric optimization
   - Economic optimization

3. RELATED RATES PROBLEMS
   - Ladder sliding down a wall
   - Expanding sphere
   - Time-dependent relationships
   - Chain rule applications

4. ECONOMIC APPLICATIONS
   - Marginal cost, revenue, and profit
   - Profit maximization
   - Cost-benefit analysis
   - Economic optimization

5. MOTION ANALYSIS
   - Position, velocity, and acceleration
   - When objects are at rest
   - Motion characteristics
   - Physics applications

These applications form the foundation for:
- Business and economic decision making
- Engineering design and optimization
- Physics and motion analysis
- Mathematical modeling and analysis
- Real-world problem solving
""")

print("\nAll concepts have been implemented with:")
print("- Clear mathematical explanations")
print("- Comprehensive visualizations")
print("- Practical real-world applications")
print("- Numerical verification of theoretical results")
print("- Code annotations and documentation")

