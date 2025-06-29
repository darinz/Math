# Multivariable Calculus

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Multivariable calculus generalizes the concepts of single-variable calculus to functions of several variables. This is essential for understanding high-dimensional spaces, which are ubiquitous in AI/ML and data science. Many models, such as neural networks, operate in spaces with thousands or millions of dimensions, making multivariable calculus foundational for:
- Optimization of loss functions with many parameters
- Sensitivity analysis and feature importance
- Modeling complex systems with multiple inputs

## 6.1 Functions of Multiple Variables

### Mathematical Foundations and Visualization

A function of two variables, \( f(x, y) \), assigns a real number to each point \( (x, y) \) in its domain. The graph of such a function is a surface in three-dimensional space. Key concepts include:
- **Level curves (contours):** Curves where \( f(x, y) = c \) for constant \( c \). These help visualize the function's behavior in the plane.
- **Surfaces:** The set of points \( (x, y, f(x, y)) \) forms a surface, which can be visualized in 3D.

**Relevance to AI/ML:**
- Loss landscapes in neural networks are high-dimensional surfaces.
- Contour plots help visualize optimization paths and convergence.
- Understanding the geometry of multivariable functions aids in interpreting model behavior and feature interactions.

### Python Implementation: Multivariable Functions

The following code demonstrates how to define and visualize several multivariable functions, with commentary on their geometric and practical significance.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def multivariable_functions():
    """
    Explore functions of multiple variables.
    Examples:
    1. Paraboloid: f(x, y) = x^2 + y^2 (convex surface, unique minimum)
    2. Oscillatory: f(x, y) = sin(x) * cos(y) (multiple local extrema)
    3. Saddle: f(x, y) = x*y (saddle point at origin)
    """
    
    # Example 1: f(x,y) = x^2 + y^2 (paraboloid)
    x, y = sp.symbols('x y')
    f1 = x**2 + y**2
    
    # Example 2: f(x,y) = sin(x) * cos(y)
    f2 = sp.sin(x) * sp.cos(y)
    
    # Example 3: f(x,y) = x*y (saddle surface)
    f3 = x * y
    
    print("Multivariable Functions:")
    print(f"f1(x,y) = {f1}")
    print(f"f2(x,y) = {f2}")
    print(f"f3(x,y) = {f3}")
    
    return f1, f2, f3

f1, f2, f3 = multivariable_functions()

# Visualize multivariable functions
def visualize_multivariable_functions():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Function 1: f(x,y) = x^2 + y^2
    Z1 = X**2 + Y**2
    
    # Function 2: f(x,y) = sin(x) * cos(y)
    Z2 = np.sin(X) * np.cos(Y)
    
    # Function 3: f(x,y) = x*y
    Z3 = X * Y
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plots
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
    ax1.set_title('f(x,y) = x^2 + y^2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
    ax2.set_title('f(x,y) = sin(x) * cos(y)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z3, cmap='coolwarm', alpha=0.8)
    ax3.set_title('f(x,y) = x*y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    plt.show()
    
    # Contour plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    contour1 = ax1.contour(X, Y, Z1, levels=10)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.set_title('Contours: f(x,y) = x^2 + y^2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    contour2 = ax2.contour(X, Y, Z2, levels=10)
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.set_title('Contours: f(x,y) = sin(x) * cos(y)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    
    contour3 = ax3.contour(X, Y, Z3, levels=10)
    ax3.clabel(contour3, inline=True, fontsize=8)
    ax3.set_title('Contours: f(x,y) = x*y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_multivariable_functions()
```

**Explanation:**
- The code defines and visualizes three types of surfaces: convex (paraboloid), oscillatory, and saddle.
- 3D plots show the geometry of each function, while contour plots reveal level curves and critical points.
- These visualizations are directly relevant to understanding optimization landscapes and feature interactions in AI/ML.

## 6.2 Partial Derivatives

### Mathematical Foundations and Computation

A partial derivative measures how a multivariable function changes as one variable varies, holding the others constant. For \( f(x, y) \):
\[
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
\]
\[
\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y + h) - f(x, y)}{h}
\]
Partial derivatives are the building blocks for gradients, Jacobians, and optimization in high-dimensional spaces.

**Relevance to AI/ML:**
- Gradients (vectors of partial derivatives) are used in gradient descent and backpropagation.
- Sensitivity analysis: Partial derivatives quantify how sensitive a model's output is to each input feature.

### Python Implementation: Partial Derivatives

```python
def partial_derivatives():
    """
    Compute partial derivatives of multivariable functions.
    Steps:
    1. Define the function symbolically.
    2. Compute partial derivatives with respect to each variable.
    3. Interpret the results.
    """
    x, y = sp.symbols('x y')
    
    # Example 1: f(x,y) = x^2 + y^2
    f1 = x**2 + y**2
    df1_dx = sp.diff(f1, x)
    df1_dy = sp.diff(f1, y)
    
    print("Partial Derivatives:")
    print(f"f(x,y) = {f1}")
    print(f"∂f/∂x = {df1_dx}")
    print(f"∂f/∂y = {df1_dy}")
    
    # Example 2: f(x,y) = x*y + sin(x)
    f2 = x*y + sp.sin(x)
    df2_dx = sp.diff(f2, x)
    df2_dy = sp.diff(f2, y)
    
    print(f"\nf(x,y) = {f2}")
    print(f"∂f/∂x = {df2_dx}")
    print(f"∂f/∂y = {df2_dy}")
    
    # Example 3: f(x,y) = e^(x*y)
    f3 = sp.exp(x*y)
    df3_dx = sp.diff(f3, x)
    df3_dy = sp.diff(f3, y)
    
    print(f"\nf(x,y) = {f3}")
    print(f"∂f/∂x = {df3_dx}")
    print(f"∂f/∂y = {df3_dy}")
    
    return f1, f2, f3, df1_dx, df1_dy, df2_dx, df2_dy, df3_dx, df3_dy

f1, f2, f3, df1_dx, df1_dy, df2_dx, df2_dy, df3_dx, df3_dy = partial_derivatives()

# Visualize partial derivatives
def visualize_partial_derivatives():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Function and its partial derivatives
    Z = X**2 + Y**2
    dZ_dx = 2 * X
    dZ_dy = 2 * Y
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original function
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('f(x,y) = x^2 + y^2')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    
    # ∂f/∂x
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, dZ_dx, cmap='plasma', alpha=0.8)
    ax2.set_title('∂f/∂x = 2x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('∂f/∂x')
    
    # ∂f/∂y
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, dZ_dy, cmap='coolwarm', alpha=0.8)
    ax3.set_title('∂f/∂y = 2y')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('∂f/∂y')
    
    plt.tight_layout()
    plt.show()

visualize_partial_derivatives()
```

**Explanation:**
- The code computes partial derivatives for several functions, illustrating how each variable affects the output.
- 3D plots show the original function and the effect of changing each variable independently.
- These concepts are foundational for gradient-based optimization and feature sensitivity in AI/ML.

## 6.3 Gradient and Directional Derivatives

### Gradient Vector

```python
def gradient_calculations():
    """Calculate gradients of multivariable functions"""
    
    x, y = sp.symbols('x y')
    
    # Example: f(x,y) = x² + y²
    f = x**2 + y**2
    grad_f = [sp.diff(f, x), sp.diff(f, y)]
    
    print("Gradient:")
    print(f"f(x,y) = {f}")
    print(f"∇f = [{grad_f[0]}, {grad_f[1]}]")
    
    # Evaluate gradient at specific points
    points = [(0, 0), (1, 1), (-1, 0), (2, -1)]
    print(f"\nGradient at specific points:")
    for point in points:
        grad_at_point = [grad_f[0].subs([(x, point[0]), (y, point[1])]),
                        grad_f[1].subs([(x, point[0]), (y, point[1])])]
        print(f"∇f({point}) = {grad_at_point}")
    
    return f, grad_f

f, grad_f = gradient_calculations()

# Visualize gradient field
def visualize_gradient_field():
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # Gradient components
    dF_dx = 2 * X
    dF_dy = 2 * Y
    
    plt.figure(figsize=(10, 8))
    
    # Gradient field
    plt.quiver(X, Y, dF_dx, dF_dy, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    
    # Contour plot
    contour = plt.contour(X, Y, X**2 + Y**2, levels=10, alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8)
    
    plt.title('Gradient Field of f(x,y) = x² + y²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

visualize_gradient_field()
```

### Directional Derivatives

```python
def directional_derivative():
    """Calculate directional derivatives"""
    
    x, y = sp.symbols('x y')
    
    # Function: f(x,y) = x² + y²
    f = x**2 + y**2
    
    # Direction vector: u = [cos(θ), sin(θ)]
    theta = sp.Symbol('theta')
    u_x = sp.cos(theta)
    u_y = sp.sin(theta)
    
    # Gradient
    grad_f = [sp.diff(f, x), sp.diff(f, y)]
    
    # Directional derivative: D_u f = ∇f · u
    directional_deriv = grad_f[0] * u_x + grad_f[1] * u_y
    directional_deriv = sp.simplify(directional_deriv)
    
    print("Directional Derivative:")
    print(f"f(x,y) = {f}")
    print(f"Direction vector: u = [cos(θ), sin(θ)]")
    print(f"D_u f = ∇f · u = {directional_deriv}")
    
    # Evaluate at specific point and direction
    point = (1, 1)
    direction_angle = np.pi/4  # 45 degrees
    
    grad_at_point = [grad_f[0].subs([(x, point[0]), (y, point[1])]),
                    grad_f[1].subs([(x, point[0]), (y, point[1])])]
    
    u_at_angle = [np.cos(direction_angle), np.sin(direction_angle)]
    
    directional_deriv_at_point = grad_at_point[0] * u_at_angle[0] + grad_at_point[1] * u_at_angle[1]
    
    print(f"\nAt point {point} in direction θ = {direction_angle:.2f}:")
    print(f"∇f({point}) = {grad_at_point}")
    print(f"u = {u_at_angle}")
    print(f"D_u f({point}) = {directional_deriv_at_point:.4f}")
    
    return f, directional_deriv, point, direction_angle, directional_deriv_at_point

f, directional_deriv, point, direction_angle, directional_deriv_at_point = directional_derivative()

# Visualize directional derivative
def visualize_directional_derivative():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=10)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Point and direction
    plt.scatter(point[0], point[1], c='red', s=100, zorder=5, label=f'Point {point}')
    
    # Direction vector
    u_x = np.cos(direction_angle)
    u_y = np.sin(direction_angle)
    plt.arrow(point[0], point[1], u_x, u_y, head_width=0.1, head_length=0.1, 
              fc='red', ec='red', label=f'Direction θ = {direction_angle:.2f}')
    
    # Gradient at point
    grad_x = 2 * point[0]
    grad_y = 2 * point[1]
    plt.arrow(point[0], point[1], grad_x, grad_y, head_width=0.1, head_length=0.1,
              fc='blue', ec='blue', label=f'Gradient ∇f({point})')
    
    plt.title('Directional Derivative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_directional_derivative()
```

## 6.4 Optimization in Multiple Dimensions

### Critical Points and Classification

```python
def multivariable_optimization():
    """Find and classify critical points of multivariable functions"""
    
    x, y = sp.symbols('x y')
    
    # Example: f(x,y) = x³ + y³ - 3xy
    f = x**3 + y**3 - 3*x*y
    
    # Partial derivatives
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    
    # Second partial derivatives
    d2f_dx2 = sp.diff(df_dx, x)
    d2f_dy2 = sp.diff(df_dy, y)
    d2f_dxdy = sp.diff(df_dx, y)
    
    print("Multivariable Optimization:")
    print(f"f(x,y) = {f}")
    print(f"∂f/∂x = {df_dx}")
    print(f"∂f/∂y = {df_dy}")
    print(f"∂²f/∂x² = {d2f_dx2}")
    print(f"∂²f/∂y² = {d2f_dy2}")
    print(f"∂²f/∂x∂y = {d2f_dxdy}")
    
    # Find critical points
    critical_points = sp.solve([df_dx, df_dy], [x, y])
    print(f"\nCritical points: {critical_points}")
    
    # Classify critical points using second derivative test
    # D = f_xx * f_yy - (f_xy)²
    for point in critical_points:
        x_val, y_val = point
        
        # Evaluate second derivatives at critical point
        f_xx = d2f_dx2.subs([(x, x_val), (y, y_val)])
        f_yy = d2f_dy2.subs([(x, x_val), (y, y_val)])
        f_xy = d2f_dxdy.subs([(x, x_val), (y, y_val)])
        
        D = f_xx * f_yy - f_xy**2
        
        print(f"\nAt point ({x_val}, {y_val}):")
        print(f"f_xx = {f_xx}, f_yy = {f_yy}, f_xy = {f_xy}")
        print(f"D = {D}")
        
        if D > 0:
            if f_xx > 0:
                print("Local minimum")
            else:
                print("Local maximum")
        elif D < 0:
            print("Saddle point")
        else:
            print("Inconclusive (need higher order derivatives)")
    
    return f, critical_points

f, critical_points = multivariable_optimization()

# Visualize optimization
def visualize_multivariable_optimization():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**3 + Y**3 - 3*X*Y
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Mark critical points
    for point in critical_points:
        x_val, y_val = point
        z_val = x_val**3 + y_val**3 - 3*x_val*y_val
        ax1.scatter(x_val, y_val, z_val, c='red', s=100, zorder=5)
    
    ax1.set_title('f(x,y) = x³ + y³ - 3xy')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Mark critical points
    for point in critical_points:
        x_val, y_val = point
        ax2.scatter(x_val, y_val, c='red', s=100, zorder=5)
    
    ax2.set_title('Contour Plot with Critical Points')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    
    # Gradient field
    ax3 = fig.add_subplot(133)
    dF_dx = 3*X**2 - 3*Y
    dF_dy = 3*Y**2 - 3*X
    
    ax3.quiver(X[::5, ::5], Y[::5, ::5], dF_dx[::5, ::5], dF_dy[::5, ::5], 
               angles='xy', scale_units='xy', scale=1, alpha=0.7)
    
    # Mark critical points
    for point in critical_points:
        x_val, y_val = point
        ax3.scatter(x_val, y_val, c='red', s=100, zorder=5)
    
    ax3.set_title('Gradient Field')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_multivariable_optimization()
```

## 6.5 Lagrange Multipliers

### Constrained Optimization

```python
def lagrange_multipliers():
    """Solve constrained optimization problems using Lagrange multipliers"""
    
    x, y, lambda_var = sp.symbols('x y lambda')
    
    # Example: Maximize f(x,y) = xy subject to x + y = 10
    f = x * y
    g = x + y - 10  # constraint: x + y = 10
    
    # Lagrange function: L = f - λg
    L = f - lambda_var * g
    
    # Partial derivatives
    dL_dx = sp.diff(L, x)
    dL_dy = sp.diff(L, y)
    dL_dlambda = sp.diff(L, lambda_var)
    
    print("Lagrange Multipliers:")
    print(f"Objective: maximize f(x,y) = {f}")
    print(f"Constraint: {g} = 0")
    print(f"Lagrange function: L = {L}")
    print(f"∂L/∂x = {dL_dx}")
    print(f"∂L/∂y = {dL_dy}")
    print(f"∂L/∂λ = {dL_dlambda}")
    
    # Solve system of equations
    solution = sp.solve([dL_dx, dL_dy, dL_dlambda], [x, y, lambda_var])
    print(f"\nSolution: {solution}")
    
    # Verify solution
    if solution:
        x_opt, y_opt, lambda_opt = solution[0]
        print(f"Optimal x = {x_opt}")
        print(f"Optimal y = {y_opt}")
        print(f"Optimal value = {f.subs([(x, x_opt), (y, y_opt)])}")
        print(f"Constraint satisfied: {g.subs([(x, x_opt), (y, y_opt)])}")
    
    return f, g, solution

f, g, solution = lagrange_multipliers()

# Visualize constrained optimization
def visualize_constrained_optimization():
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    
    # Constraint line
    constraint_x = np.linspace(0, 10, 100)
    constraint_y = 10 - constraint_x
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot of objective function
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Constraint line
    plt.plot(constraint_x, constraint_y, 'r-', linewidth=3, label='Constraint: x + y = 10')
    
    # Optimal point
    if solution:
        x_opt, y_opt, _ = solution[0]
        plt.scatter(x_opt, y_opt, c='red', s=200, zorder=5, 
                   label=f'Optimal point: ({x_opt}, {y_opt})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Constrained Optimization: Maximize xy subject to x + y = 10')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_constrained_optimization()
```

## 6.6 Applications in Machine Learning

### Gradient Descent in Multiple Dimensions

```python
def gradient_descent_2d():
    """Implement gradient descent for multivariable functions"""
    
    # Objective function: f(x,y) = x² + y²
    def objective_function(x, y):
        return x**2 + y**2
    
    def gradient_function(x, y):
        return np.array([2*x, 2*y])
    
    # Gradient descent implementation
    def gradient_descent_2d(f, grad_f, start_point, learning_rate=0.1, max_iter=100):
        x, y = start_point
        history = [(x, y)]
        
        for i in range(max_iter):
            gradient = grad_f(x, y)
            x = x - learning_rate * gradient[0]
            y = y - learning_rate * gradient[1]
            history.append((x, y))
            
            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                break
        
        return np.array(history)
    
    # Run gradient descent
    start_point = (2.0, 2.0)
    path = gradient_descent_2d(objective_function, gradient_function, start_point)
    
    print("2D Gradient Descent:")
    print(f"Starting point: {start_point}")
    print(f"Final point: {path[-1]}")
    print(f"Final value: {objective_function(path[-1, 0], path[-1, 1]):.6f}")
    print(f"Number of iterations: {len(path)}")
    
    return path, objective_function

path, objective_function = gradient_descent_2d()

# Visualize gradient descent path
def visualize_gradient_descent_2d():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Optimization path
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Gradient descent path')
    plt.scatter(path[0, 0], path[0, 1], c='red', s=100, label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], c='green', s=100, label='End')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Gradient Descent on f(x,y) = x² + y²')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_gradient_descent_2d()
```

## Summary

- **Multivariable Functions**: Functions of multiple variables and their visualization
- **Partial Derivatives**: Derivatives with respect to individual variables
- **Gradient**: Vector of partial derivatives indicating direction of steepest ascent
- **Directional Derivatives**: Rate of change in specific directions
- **Optimization**: Finding critical points and classifying them using second derivative test
- **Lagrange Multipliers**: Constrained optimization technique
- **Applications**: Gradient descent in machine learning and optimization algorithms

## Next Steps

Understanding multivariable calculus enables you to work with high-dimensional optimization problems, understand gradient-based learning algorithms, and model complex systems. The next section explores vector calculus for understanding vector fields and their properties. 