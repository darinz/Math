# Python code extracted from 06-multivariable-calculus.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
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

# Code Block 2
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

# Code Block 3
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

# Code Block 4
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

# Code Block 5
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

# Code Block 6
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

# Code Block 7
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

# --- Advanced Topics: Vector Fields, Divergence, Curl, ML Applications ---

def vector_field_example():
    """Visualize a vector field, its gradient field, and check if it's conservative."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    # Example: F(x, y) = (P, Q) = (y, -x) (rotational field)
    P = Y
    Q = -X
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, P, Q, color='blue')
    plt.title('Vector Field: F(x, y) = (y, -x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    # Example: Gradient field of f(x, y) = x^2 + y^2
    f = lambda x, y: x**2 + y**2
    Px = 2 * X
    Qy = 2 * Y
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, Px, Qy, color='green')
    plt.title('Gradient Field: grad(f), f(x, y) = x^2 + y^2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    # Conservative check: curl of grad(f) should be zero
    # For grad(f), curl = dQ/dx - dP/dy = 0

vector_field_example()


def divergence_and_curl_example():
    """Compute and visualize divergence and curl of a vector field."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    # Example: F(x, y) = (P, Q) = (x, y)
    P = X
    Q = Y
    # Divergence: dP/dx + dQ/dy = 1 + 1 = 2
    div = np.ones_like(X) * 2
    # Curl: dQ/dx - dP/dy = 0 - 0 = 0
    curl = np.zeros_like(X)
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, P, Q, color='purple')
    plt.title('Vector Field: F(x, y) = (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    # Visualize divergence as color
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, div, shading='auto', cmap='Reds')
    plt.colorbar(label='Divergence')
    plt.title('Divergence of F(x, y) = (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()
    # Visualize curl as color
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, curl, shading='auto', cmap='Blues')
    plt.colorbar(label='Curl')
    plt.title('Curl of F(x, y) = (x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    # Example: Rotational field F(x, y) = (y, -x)
    P2 = Y
    Q2 = -X
    # Divergence: dP2/dx + dQ2/dy = 0 + 0 = 0
    div2 = np.zeros_like(X)
    # Curl: dQ2/dx - dP2/dy = -1 - 1 = -2
    curl2 = -2 * np.ones_like(X)
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, curl2, shading='auto', cmap='Blues')
    plt.colorbar(label='Curl')
    plt.title('Curl of F(x, y) = (y, -x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

divergence_and_curl_example()

# --- ML Applications: Gradient Flows, Neural Tangent Kernel, Information Geometry ---

def gradient_flow_example():
    """Simulate gradient flow for a simple quadratic loss (ML context)."""
    # dtheta/dt = -grad L(theta), L(theta) = 0.5 * theta^2
    t = np.linspace(0, 5, 100)
    theta0 = 2.0
    L = lambda theta: 0.5 * theta**2
    grad_L = lambda theta: theta
    # Analytical solution: theta(t) = theta0 * exp(-t)
    theta_t = theta0 * np.exp(-t)
    plt.figure(figsize=(8, 4))
    plt.plot(t, theta_t, label=r'$\theta(t) = 2e^{-t}$')
    plt.title('Gradient Flow: dθ/dt = -∇L(θ), L(θ) = 0.5θ²')
    plt.xlabel('Time t')
    plt.ylabel('θ(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

gradient_flow_example()


def neural_tangent_kernel_example():
    """Symbolic computation of a simple neural tangent kernel (NTK) for a linear model."""
    x1, x2, w = sp.symbols('x1 x2 w')
    # Model: f_w(x) = w * x
    f = w * x1
    grad_f_x1 = sp.diff(f, w)
    grad_f_x2 = sp.diff(f.subs(x1, x2), w)
    # NTK: K(x1, x2) = grad_f(x1) * grad_f(x2)
    K = grad_f_x1 * grad_f_x2
    print(f"Neural Tangent Kernel for linear model: K(x1, x2) = {K}")

neural_tangent_kernel_example()


def information_geometry_example():
    """Compute Fisher information matrix for a normal distribution (ML/statistics context)."""
    mu, sigma, x = sp.symbols('mu sigma x')
    # log-likelihood: log p(x|mu, sigma) for normal
    logp = -0.5 * sp.log(2 * sp.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)
    dlogp_dmu = sp.diff(logp, mu)
    dlogp_dsigma = sp.diff(logp, sigma)
    # Fisher information matrix elements
    I_mu_mu = sp.simplify(sp.integrate(dlogp_dmu**2 * (1/(sp.sqrt(2*sp.pi)*sigma)) * sp.exp(-(x-mu)**2/(2*sigma**2)), (x, -sp.oo, sp.oo)))
    I_sigma_sigma = sp.simplify(sp.integrate(dlogp_dsigma**2 * (1/(sp.sqrt(2*sp.pi)*sigma)) * sp.exp(-(x-mu)**2/(2*sigma**2)), (x, -sp.oo, sp.oo)))
    print(f"Fisher information (mu, mu): {I_mu_mu}")
    print(f"Fisher information (sigma, sigma): {I_sigma_sigma}")

information_geometry_example()

