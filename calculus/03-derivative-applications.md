# Applications of Derivatives

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Derivatives have numerous practical applications beyond basic differentiation. In this section, we explore curve sketching, related rates, optimization problems, and real-world applications that are fundamental to understanding function behavior and solving practical problems in AI/ML and data science.

## 3.1 Curve Sketching and Function Analysis

### First and Second Derivative Tests

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize_scalar

def analyze_function():
    """Analyze a function using derivatives for curve sketching"""
    x = sp.Symbol('x')
    
    # Example function: f(x) = x³ - 6x² + 9x + 1
    f = x**3 - 6*x**2 + 9*x + 1
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    
    print(f"Function: f(x) = {f}")
    print(f"First derivative: f'(x) = {f_prime}")
    print(f"Second derivative: f''(x) = {f_double_prime}")
    
    # Find critical points
    critical_points = sp.solve(f_prime, x)
    print(f"\nCritical points: {critical_points}")
    
    # Classify critical points
    for point in critical_points:
        second_deriv = f_double_prime.subs(x, point)
        if second_deriv > 0:
            print(f"x = {point}: Local minimum (f''({point}) = {second_deriv})")
        elif second_deriv < 0:
            print(f"x = {point}: Local maximum (f''({point}) = {second_deriv})")
        else:
            print(f"x = {point}: Saddle point or inflection point")
    
    # Find inflection points
    inflection_points = sp.solve(f_double_prime, x)
    print(f"\nInflection points: {inflection_points}")
    
    return f, f_prime, f_double_prime, critical_points, inflection_points

f, f_prime, f_double_prime, critical_points, inflection_points = analyze_function()

# Visualize the function and its derivatives
x_vals = np.linspace(-1, 5, 1000)
y_vals = [f.subs(sp.Symbol('x'), x) for x in x_vals]
dy_vals = [f_prime.subs(sp.Symbol('x'), x) for x in x_vals]
ddy_vals = [f_double_prime.subs(sp.Symbol('x'), x) for x in x_vals]

plt.figure(figsize=(15, 10))

# Original function
plt.subplot(2, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 6x² + 9x + 1')
for point in critical_points:
    y_point = f.subs(sp.Symbol('x'), point)
    plt.scatter(point, y_point, c='red', s=100, zorder=5)
for point in inflection_points:
    y_point = f.subs(sp.Symbol('x'), point)
    plt.scatter(point, y_point, c='green', s=100, marker='s', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function with Critical and Inflection Points')
plt.legend()
plt.grid(True)

# First derivative
plt.subplot(2, 2, 2)
plt.plot(x_vals, dy_vals, 'r-', linewidth=2, label="f'(x)")
plt.axhline(y=0, color='k', linestyle='--')
for point in critical_points:
    plt.scatter(point, 0, c='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('First Derivative')
plt.legend()
plt.grid(True)

# Second derivative
plt.subplot(2, 2, 3)
plt.plot(x_vals, ddy_vals, 'g-', linewidth=2, label="f''(x)")
plt.axhline(y=0, color='k', linestyle='--')
for point in inflection_points:
    second_deriv = f_double_prime.subs(sp.Symbol('x'), point)
    plt.scatter(point, second_deriv, c='green', s=100, marker='s', zorder=5)
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.title('Second Derivative')
plt.legend()
plt.grid(True)

# Concavity analysis
plt.subplot(2, 2, 4)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
# Color regions by concavity
for i in range(len(x_vals)-1):
    if ddy_vals[i] > 0:
        plt.fill_between(x_vals[i:i+2], y_vals[i:i+2], alpha=0.3, color='green')
    else:
        plt.fill_between(x_vals[i:i+2], y_vals[i:i+2], alpha=0.3, color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Concavity Analysis (Green: Concave Up, Red: Concave Down)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Asymptotes and End Behavior

```python
def analyze_asymptotes():
    """Analyze asymptotes and end behavior of rational functions"""
    x = sp.Symbol('x')
    
    # Example: f(x) = (x² + 2x + 1) / (x + 1)
    f = (x**2 + 2*x + 1) / (x + 1)
    
    # Find vertical asymptotes (where denominator = 0)
    denominator = x + 1
    vertical_asymptotes = sp.solve(denominator, x)
    print(f"Vertical asymptotes: x = {vertical_asymptotes}")
    
    # Find horizontal asymptotes (limit as x → ±∞)
    limit_pos_inf = sp.limit(f, x, sp.oo)
    limit_neg_inf = sp.limit(f, x, -sp.oo)
    print(f"Horizontal asymptote as x → ∞: {limit_pos_inf}")
    print(f"Horizontal asymptote as x → -∞: {limit_neg_inf}")
    
    # Check for slant asymptotes
    if limit_pos_inf == sp.oo or limit_pos_inf == -sp.oo:
        # Perform polynomial long division
        quotient = sp.div(x**2 + 2*x + 1, x + 1)[0]
        print(f"Slant asymptote: y = {quotient}")
    
    return f, vertical_asymptotes, limit_pos_inf

f, vertical_asymptotes, horizontal_asymptote = analyze_asymptotes()

# Visualize with asymptotes
x_vals = np.linspace(-5, 5, 1000)
# Remove points near vertical asymptote
x_vals = x_vals[np.abs(x_vals + 1) > 0.1]
y_vals = [f.subs(sp.Symbol('x'), x) for x in x_vals]

plt.figure(figsize=(12, 8))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x² + 2x + 1)/(x + 1)')

# Vertical asymptote
for asymptote in vertical_asymptotes:
    plt.axvline(x=asymptote, color='r', linestyle='--', label=f'Vertical asymptote: x = {asymptote}')

# Horizontal asymptote
if horizontal_asymptote != sp.oo and horizontal_asymptote != -sp.oo:
    plt.axhline(y=horizontal_asymptote, color='g', linestyle='--', 
                label=f'Horizontal asymptote: y = {horizontal_asymptote}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Rational Function with Asymptotes')
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-10, 10)
plt.show()
```

## 3.2 Related Rates Problems

### Classic Related Rates Examples

```python
def related_rates_example():
    """Example: A ladder sliding down a wall"""
    # A 10-foot ladder is sliding down a wall
    # When the bottom is 6 feet from the wall, it's moving at 2 ft/s
    # How fast is the top sliding down?
    
    # Variables
    ladder_length = 10  # feet
    x = 6  # distance from wall
    dx_dt = 2  # rate of change of x
    
    # Relationship: x² + y² = 10² (Pythagorean theorem)
    # Differentiate with respect to time: 2x(dx/dt) + 2y(dy/dt) = 0
    # Therefore: dy/dt = -(x/y)(dx/dt)
    
    y = np.sqrt(ladder_length**2 - x**2)
    dy_dt = -(x / y) * dx_dt
    
    print(f"Ladder length: {ladder_length} feet")
    print(f"Distance from wall (x): {x} feet")
    print(f"Height on wall (y): {y:.2f} feet")
    print(f"Rate of change of x: {dx_dt} ft/s")
    print(f"Rate of change of y: {dy_dt:.2f} ft/s")
    
    return x, y, dx_dt, dy_dt

x, y, dx_dt, dy_dt = related_rates_example()

# Visualize the ladder problem
def visualize_ladder_problem():
    # Create animation-like visualization
    x_vals = np.linspace(0, 10, 20)
    y_vals = np.sqrt(10**2 - x_vals**2)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ladder positions
    for i in range(len(x_vals)):
        plt.plot([0, x_vals[i]], [y_vals[i], 0], 'b-', alpha=0.3)
    
    # Highlight current position
    plt.plot([0, x], [y, 0], 'r-', linewidth=3, label=f'Current position: x={x}, y={y:.2f}')
    
    # Velocity vectors
    plt.arrow(x, 0, dx_dt, 0, head_width=0.2, head_length=0.2, fc='red', ec='red', 
              label=f'dx/dt = {dx_dt} ft/s')
    plt.arrow(0, y, 0, dy_dt, head_width=0.2, head_length=0.2, fc='red', ec='red', 
              label=f'dy/dt = {dy_dt:.2f} ft/s')
    
    plt.xlabel('Distance from wall (feet)')
    plt.ylabel('Height on wall (feet)')
    plt.title('Ladder Sliding Down Wall - Related Rates Problem')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

visualize_ladder_problem()
```

### Volume and Surface Area Related Rates

```python
def volume_related_rates():
    """Example: Expanding sphere"""
    # A spherical balloon is being inflated at a rate of 10π cubic inches per second
    # How fast is the radius increasing when the radius is 5 inches?
    
    # Variables
    dV_dt = 10 * np.pi  # rate of change of volume
    r = 5  # current radius
    
    # Volume of sphere: V = (4/3)πr³
    # Differentiate: dV/dt = 4πr²(dr/dt)
    # Therefore: dr/dt = (dV/dt)/(4πr²)
    
    dr_dt = dV_dt / (4 * np.pi * r**2)
    
    print(f"Rate of change of volume: {dV_dt:.2f} cubic inches/second")
    print(f"Current radius: {r} inches")
    print(f"Rate of change of radius: {dr_dt:.4f} inches/second")
    
    # Surface area rate of change
    # Surface area: A = 4πr²
    # dA/dt = 8πr(dr/dt)
    dA_dt = 8 * np.pi * r * dr_dt
    
    print(f"Rate of change of surface area: {dA_dt:.2f} square inches/second")
    
    return r, dV_dt, dr_dt, dA_dt

r, dV_dt, dr_dt, dA_dt = volume_related_rates()

# Visualize expanding sphere
def visualize_expanding_sphere():
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D visualization
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, alpha=0.7, color='blue')
    ax1.set_title(f'Sphere with radius {r} inches')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # 2D cross-section with velocity vectors
    ax2 = fig.add_subplot(122)
    circle = plt.Circle((0, 0), r, fill=False, color='blue', linewidth=2)
    ax2.add_patch(circle)
    
    # Velocity vector
    ax2.arrow(0, 0, dr_dt, 0, head_width=0.2, head_length=0.2, fc='red', ec='red',
              label=f'dr/dt = {dr_dt:.4f} in/s')
    
    ax2.set_xlim(-r-1, r+1)
    ax2.set_ylim(-r-1, r+1)
    ax2.set_title('Cross-section with radial velocity')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

visualize_expanding_sphere()
```

## 3.3 Optimization Problems

### Classic Optimization Examples

```python
def optimization_examples():
    """Solve classic optimization problems"""
    
    # Example 1: Maximum area rectangle with fixed perimeter
    def rectangle_optimization():
        # Perimeter P = 2x + 2y = 20 (fixed)
        # Area A = xy
        # From perimeter: y = 10 - x
        # Area: A = x(10 - x) = 10x - x²
        
        x = sp.Symbol('x')
        A = x * (10 - x)
        A_prime = sp.diff(A, x)
        
        # Find critical points
        critical_points = sp.solve(A_prime, x)
        print(f"Critical points: {critical_points}")
        
        # Evaluate area at critical points
        for point in critical_points:
            area = A.subs(x, point)
            print(f"x = {point}, Area = {area}")
        
        return critical_points[0], A.subs(x, critical_points[0])
    
    # Example 2: Minimum distance from point to line
    def distance_optimization():
        # Find minimum distance from point (2, 3) to line y = 2x + 1
        
        x = sp.Symbol('x')
        point_x, point_y = 2, 3
        line_y = 2*x + 1
        
        # Distance formula: d = √[(x-2)² + (y-3)²]
        # Substitute y = 2x + 1
        d_squared = (x - point_x)**2 + (line_y - point_y)**2
        d = sp.sqrt(d_squared)
        
        # Minimize distance (or distance squared)
        d_squared_prime = sp.diff(d_squared, x)
        critical_point = sp.solve(d_squared_prime, x)[0]
        
        min_distance = d.subs(x, critical_point)
        closest_point_y = line_y.subs(x, critical_point)
        
        print(f"Closest point on line: ({critical_point}, {closest_point_y})")
        print(f"Minimum distance: {min_distance}")
        
        return critical_point, closest_point_y, min_distance
    
    print("Optimization Example 1: Maximum Area Rectangle")
    print("-" * 50)
    opt_x, opt_area = rectangle_optimization()
    print(f"Optimal dimensions: x = {opt_x}, y = {10 - opt_x}")
    print(f"Maximum area: {opt_area}")
    
    print("\nOptimization Example 2: Minimum Distance")
    print("-" * 50)
    closest_x, closest_y, min_dist = distance_optimization()
    print(f"Closest point: ({closest_x}, {closest_y})")
    print(f"Minimum distance: {min_dist}")
    
    return opt_x, opt_area, closest_x, closest_y, min_dist

opt_x, opt_area, closest_x, closest_y, min_dist = optimization_examples()

# Visualize optimization results
def visualize_optimization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Rectangle optimization
    x_vals = np.linspace(0, 10, 100)
    y_vals = 10 - x_vals
    areas = x_vals * y_vals
    
    ax1.plot(x_vals, areas, 'b-', linewidth=2, label='Area = x(10-x)')
    ax1.scatter(opt_x, opt_area, c='red', s=100, zorder=5, 
                label=f'Maximum: ({opt_x}, {opt_area})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Area')
    ax1.set_title('Rectangle Area Optimization')
    ax1.legend()
    ax1.grid(True)
    
    # Distance optimization
    line_x = np.linspace(-2, 4, 100)
    line_y = 2 * line_x + 1
    
    ax2.plot(line_x, line_y, 'b-', linewidth=2, label='y = 2x + 1')
    ax2.scatter(2, 3, c='red', s=100, label='Point (2, 3)')
    ax2.scatter(closest_x, closest_y, c='green', s=100, label=f'Closest point ({closest_x:.2f}, {closest_y:.2f})')
    ax2.plot([2, closest_x], [3, closest_y], 'r--', linewidth=2, label=f'Distance = {min_dist:.2f}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Minimum Distance from Point to Line')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

visualize_optimization()
```

## 3.4 Applications in Economics and Business

### Marginal Analysis

```python
def marginal_analysis():
    """Economic applications of derivatives: marginal cost, revenue, profit"""
    
    # Example: Cost function C(x) = 1000 + 10x + 0.1x²
    # Revenue function R(x) = 50x - 0.05x²
    # Profit function P(x) = R(x) - C(x)
    
    x = sp.Symbol('x')
    C = 1000 + 10*x + 0.1*x**2  # Cost function
    R = 50*x - 0.05*x**2        # Revenue function
    P = R - C                    # Profit function
    
    # Derivatives (marginal functions)
    C_prime = sp.diff(C, x)      # Marginal cost
    R_prime = sp.diff(R, x)      # Marginal revenue
    P_prime = sp.diff(P, x)      # Marginal profit
    
    print("Economic Functions:")
    print(f"Cost: C(x) = {C}")
    print(f"Revenue: R(x) = {R}")
    print(f"Profit: P(x) = {P}")
    print(f"\nMarginal Functions:")
    print(f"Marginal Cost: C'(x) = {C_prime}")
    print(f"Marginal Revenue: R'(x) = {R_prime}")
    print(f"Marginal Profit: P'(x) = {P_prime}")
    
    # Find maximum profit
    critical_points = sp.solve(P_prime, x)
    print(f"\nCritical points for profit: {critical_points}")
    
    # Evaluate profit at critical points
    for point in critical_points:
        if point > 0:  # Only positive production makes sense
            profit = P.subs(x, point)
            print(f"Production level: {point}, Profit: {profit}")
    
    return C, R, P, C_prime, R_prime, P_prime, critical_points

C, R, P, C_prime, R_prime, P_prime, critical_points = marginal_analysis()

# Visualize economic functions
def visualize_economic_functions():
    x_vals = np.linspace(0, 100, 1000)
    
    # Convert symbolic expressions to numerical functions
    C_vals = [C.subs(sp.Symbol('x'), x) for x in x_vals]
    R_vals = [R.subs(sp.Symbol('x'), x) for x in x_vals]
    P_vals = [P.subs(sp.Symbol('x'), x) for x in x_vals]
    C_prime_vals = [C_prime.subs(sp.Symbol('x'), x) for x in x_vals]
    R_prime_vals = [R_prime.subs(sp.Symbol('x'), x) for x in x_vals]
    P_prime_vals = [P_prime.subs(sp.Symbol('x'), x) for x in x_vals]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total functions
    ax1.plot(x_vals, C_vals, 'r-', linewidth=2, label='Cost C(x)')
    ax1.plot(x_vals, R_vals, 'g-', linewidth=2, label='Revenue R(x)')
    ax1.plot(x_vals, P_vals, 'b-', linewidth=2, label='Profit P(x)')
    ax1.set_xlabel('Quantity (x)')
    ax1.set_ylabel('Dollars')
    ax1.set_title('Total Cost, Revenue, and Profit')
    ax1.legend()
    ax1.grid(True)
    
    # Marginal functions
    ax2.plot(x_vals, C_prime_vals, 'r-', linewidth=2, label="Marginal Cost C'(x)")
    ax2.plot(x_vals, R_prime_vals, 'g-', linewidth=2, label="Marginal Revenue R'(x)")
    ax2.plot(x_vals, P_prime_vals, 'b-', linewidth=2, label="Marginal Profit P'(x)")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Quantity (x)')
    ax2.set_ylabel('Dollars per unit')
    ax2.set_title('Marginal Functions')
    ax2.legend()
    ax2.grid(True)
    
    # Profit maximization
    ax3.plot(x_vals, P_vals, 'b-', linewidth=2, label='Profit P(x)')
    for point in critical_points:
        if point > 0:
            profit = P.subs(sp.Symbol('x'), point)
            ax3.scatter(point, profit, c='red', s=100, zorder=5,
                       label=f'Max profit: ({point:.1f}, {profit:.1f})')
    ax3.set_xlabel('Quantity (x)')
    ax3.set_ylabel('Profit')
    ax3.set_title('Profit Maximization')
    ax3.legend()
    ax3.grid(True)
    
    # Break-even analysis
    ax4.plot(x_vals, C_vals, 'r-', linewidth=2, label='Cost C(x)')
    ax4.plot(x_vals, R_vals, 'g-', linewidth=2, label='Revenue R(x)')
    ax4.fill_between(x_vals, C_vals, R_vals, where=(np.array(R_vals) > np.array(C_vals)), 
                     alpha=0.3, color='green', label='Profit region')
    ax4.fill_between(x_vals, C_vals, R_vals, where=(np.array(R_vals) < np.array(C_vals)), 
                     alpha=0.3, color='red', label='Loss region')
    ax4.set_xlabel('Quantity (x)')
    ax4.set_ylabel('Dollars')
    ax4.set_title('Break-even Analysis')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_economic_functions()
```

## 3.5 Applications in Physics and Engineering

### Motion and Velocity

```python
def motion_analysis():
    """Analyze motion using derivatives: position, velocity, acceleration"""
    
    # Example: Position function s(t) = t³ - 6t² + 9t + 1
    t = sp.Symbol('t')
    s = t**3 - 6*t**2 + 9*t + 1  # Position function
    v = sp.diff(s, t)            # Velocity function
    a = sp.diff(v, t)            # Acceleration function
    
    print("Motion Analysis:")
    print(f"Position: s(t) = {s}")
    print(f"Velocity: v(t) = s'(t) = {v}")
    print(f"Acceleration: a(t) = v'(t) = s''(t) = {a}")
    
    # Find when velocity is zero (stationary points)
    stationary_points = sp.solve(v, t)
    print(f"\nStationary points (v = 0): t = {stationary_points}")
    
    # Find when acceleration is zero
    acceleration_zeros = sp.solve(a, t)
    print(f"Acceleration zeros (a = 0): t = {acceleration_zeros}")
    
    # Analyze motion at specific times
    times = [0, 1, 2, 3, 4]
    print(f"\nMotion Analysis at Specific Times:")
    for time in times:
        position = s.subs(t, time)
        velocity = v.subs(t, time)
        acceleration = a.subs(t, time)
        print(f"t = {time}: s = {position:.2f}, v = {velocity:.2f}, a = {acceleration:.2f}")
    
    return s, v, a, stationary_points, acceleration_zeros

s, v, a, stationary_points, acceleration_zeros = motion_analysis()

# Visualize motion
def visualize_motion():
    t_vals = np.linspace(0, 4, 1000)
    
    s_vals = [s.subs(sp.Symbol('t'), t) for t in t_vals]
    v_vals = [v.subs(sp.Symbol('t'), t) for t in t_vals]
    a_vals = [a.subs(sp.Symbol('t'), t) for t in t_vals]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Position
    ax1.plot(t_vals, s_vals, 'b-', linewidth=2, label='Position s(t)')
    for point in stationary_points:
        if 0 <= point <= 4:
            pos = s.subs(sp.Symbol('t'), point)
            ax1.scatter(point, pos, c='red', s=100, zorder=5)
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position')
    ax1.set_title('Position vs Time')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity
    ax2.plot(t_vals, v_vals, 'g-', linewidth=2, label='Velocity v(t)')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in stationary_points:
        if 0 <= point <= 4:
            ax2.scatter(point, 0, c='red', s=100, zorder=5)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Acceleration
    ax3.plot(t_vals, a_vals, 'r-', linewidth=2, label='Acceleration a(t)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in acceleration_zeros:
        if 0 <= point <= 4:
            ax3.scatter(point, 0, c='red', s=100, zorder=5)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Acceleration')
    ax3.set_title('Acceleration vs Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_motion()
```

## Summary

- **Curve Sketching**: Use derivatives to understand function behavior, find extrema, and analyze concavity
- **Related Rates**: Solve problems involving changing quantities using the chain rule
- **Optimization**: Find maximum/minimum values using critical points and derivative tests
- **Economic Applications**: Marginal analysis for cost, revenue, and profit optimization
- **Physical Applications**: Motion analysis using position, velocity, and acceleration functions

## Next Steps

Understanding derivative applications enables you to solve real-world optimization problems, analyze function behavior, and model dynamic systems. The next section explores integration applications for calculating areas, volumes, and cumulative effects. 