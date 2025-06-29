# Vector Calculus

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Vector calculus extends calculus to vector fields and provides powerful tools for understanding physical phenomena, fluid dynamics, electromagnetism, and many other applications. It's essential for understanding advanced machine learning concepts like gradient flows and optimization in function spaces.

## 7.1 Vector Fields

### Definition and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

def vector_fields():
    """Explore different types of vector fields"""
    
    # Example 1: Conservative vector field F = [x, y]
    def conservative_field(x, y):
        return x, y
    
    # Example 2: Rotational vector field F = [-y, x]
    def rotational_field(x, y):
        return -y, x
    
    # Example 3: Radial field F = [x/r, y/r] where r = sqrt(x² + y²)
    def radial_field(x, y):
        r = np.sqrt(x**2 + y**2)
        if r == 0:
            return 0, 0
        return x/r, y/r
    
    return conservative_field, rotational_field, radial_field

conservative_field, rotational_field, radial_field = vector_fields()

# Visualize vector fields
def visualize_vector_fields():
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Conservative field: F = [x, y]
    U1, V1 = conservative_field(X, Y)
    ax1.quiver(X, Y, U1, V1, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax1.set_title('Conservative Field: F = [x, y]')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Rotational field: F = [-y, x]
    U2, V2 = rotational_field(X, Y)
    ax2.quiver(X, Y, U2, V2, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax2.set_title('Rotational Field: F = [-y, x]')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    
    # Radial field: F = [x/r, y/r]
    U3, V3 = radial_field(X, Y)
    ax3.quiver(X, Y, U3, V3, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax3.set_title('Radial Field: F = [x/r, y/r]')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    
    # Gradient field of f(x,y) = x² + y²
    U4 = 2 * X
    V4 = 2 * Y
    ax4.quiver(X, Y, U4, V4, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax4.set_title('Gradient Field: ∇f where f = x² + y²')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_vector_fields()
```

### 3D Vector Fields

```python
def vector_fields_3d():
    """Visualize 3D vector fields"""
    
    # Example: F = [x, y, z] (radial field in 3D)
    def radial_field_3d(x, y, z):
        return x, y, z
    
    # Example: F = [-y, x, 0] (rotational field around z-axis)
    def rotational_field_3d(x, y, z):
        return -y, x, 0
    
    return radial_field_3d, rotational_field_3d

radial_field_3d, rotational_field_3d = vector_fields_3d()

# Visualize 3D vector fields
def visualize_3d_vector_fields():
    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 5)
    z = np.linspace(-2, 2, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Radial field
    ax1 = fig.add_subplot(121, projection='3d')
    U1, V1, W1 = radial_field_3d(X, Y, Z)
    ax1.quiver(X, Y, Z, U1, V1, W1, length=0.3, alpha=0.7)
    ax1.set_title('3D Radial Field: F = [x, y, z]')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Rotational field
    ax2 = fig.add_subplot(122, projection='3d')
    U2, V2, W2 = rotational_field_3d(X, Y, Z)
    ax2.quiver(X, Y, Z, U2, V2, W2, length=0.3, alpha=0.7)
    ax2.set_title('3D Rotational Field: F = [-y, x, 0]')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    plt.tight_layout()
    plt.show()

visualize_3d_vector_fields()
```

## 7.2 Divergence and Curl

### Divergence

```python
def divergence_calculations():
    """Calculate divergence of vector fields"""
    
    x, y, z = sp.symbols('x y z')
    
    # Example 1: F = [x, y, 0]
    F1 = [x, y, 0]
    div_F1 = sp.diff(F1[0], x) + sp.diff(F1[1], y) + sp.diff(F1[2], z)
    
    # Example 2: F = [-y, x, 0]
    F2 = [-y, x, 0]
    div_F2 = sp.diff(F2[0], x) + sp.diff(F2[1], y) + sp.diff(F2[2], z)
    
    # Example 3: F = [x², y², z²]
    F3 = [x**2, y**2, z**2]
    div_F3 = sp.diff(F3[0], x) + sp.diff(F3[1], y) + sp.diff(F3[2], z)
    
    print("Divergence Calculations:")
    print(f"F1 = {F1}")
    print(f"∇ · F1 = {div_F1}")
    print(f"\nF2 = {F2}")
    print(f"∇ · F2 = {div_F2}")
    print(f"\nF3 = {F3}")
    print(f"∇ · F3 = {div_F3}")
    
    return F1, F2, F3, div_F1, div_F2, div_F3

F1, F2, F3, div_F1, div_F2, div_F3 = divergence_calculations()

# Visualize divergence
def visualize_divergence():
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Vector field F = [x, y, 0]
    U = X
    V = Y
    
    # Divergence ∇ · F = 2
    divergence = 2 * np.ones_like(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vector field
    ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax1.set_title('Vector Field: F = [x, y, 0]')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Divergence
    im = ax2.imshow(divergence, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu')
    ax2.set_title('Divergence: ∇ · F = 2')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

visualize_divergence()
```

### Curl

```python
def curl_calculations():
    """Calculate curl of vector fields"""
    
    x, y, z = sp.symbols('x y z')
    
    # Example 1: F = [-y, x, 0]
    F1 = [-y, x, 0]
    curl_F1 = [
        sp.diff(F1[2], y) - sp.diff(F1[1], z),  # ∂Fz/∂y - ∂Fy/∂z
        sp.diff(F1[0], z) - sp.diff(F1[2], x),  # ∂Fx/∂z - ∂Fz/∂x
        sp.diff(F1[1], x) - sp.diff(F1[0], y)   # ∂Fy/∂x - ∂Fx/∂y
    ]
    
    # Example 2: F = [0, 0, x*y]
    F2 = [0, 0, x*y]
    curl_F2 = [
        sp.diff(F2[2], y) - sp.diff(F2[1], z),
        sp.diff(F2[0], z) - sp.diff(F2[2], x),
        sp.diff(F2[1], x) - sp.diff(F2[0], y)
    ]
    
    # Example 3: F = [x, y, z]
    F3 = [x, y, z]
    curl_F3 = [
        sp.diff(F3[2], y) - sp.diff(F3[1], z),
        sp.diff(F3[0], z) - sp.diff(F3[2], x),
        sp.diff(F3[1], x) - sp.diff(F3[0], y)
    ]
    
    print("Curl Calculations:")
    print(f"F1 = {F1}")
    print(f"∇ × F1 = {curl_F1}")
    print(f"\nF2 = {F2}")
    print(f"∇ × F2 = {curl_F2}")
    print(f"\nF3 = {F3}")
    print(f"∇ × F3 = {curl_F3}")
    
    return F1, F2, F3, curl_F1, curl_F2, curl_F3

F1, F2, F3, curl_F1, curl_F2, curl_F3 = curl_calculations()

# Visualize curl
def visualize_curl():
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Vector field F = [-y, x, 0]
    U = -Y
    V = X
    
    # Curl ∇ × F = [0, 0, 2]
    curl_z = 2 * np.ones_like(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vector field
    ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax1.set_title('Vector Field: F = [-y, x, 0]')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Curl (z-component)
    im = ax2.imshow(curl_z, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu')
    ax2.set_title('Curl (z-component): ∇ × F = [0, 0, 2]')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

visualize_curl()
```

## 7.3 Line Integrals

### Scalar Line Integrals

```python
def line_integrals():
    """Calculate line integrals"""
    
    # Example: Line integral of f(x,y) = x + y along the curve C: x = t, y = t², 0 ≤ t ≤ 1
    
    t = sp.Symbol('t')
    
    # Parametric equations of the curve
    x = t
    y = t**2
    
    # Function to integrate
    f = x + y
    
    # Arc length element: ds = sqrt((dx/dt)² + (dy/dt)²) dt
    dx_dt = sp.diff(x, t)
    dy_dt = sp.diff(y, t)
    ds = sp.sqrt(dx_dt**2 + dy_dt**2)
    
    # Line integral: ∫f(x,y) ds = ∫f(x(t), y(t)) * ds/dt dt
    integrand = f * ds
    line_integral = sp.integrate(integrand, (t, 0, 1))
    
    print("Line Integral:")
    print(f"Curve C: x = {x}, y = {y}, 0 ≤ t ≤ 1")
    print(f"Function: f(x,y) = {f}")
    print(f"Arc length element: ds = {ds} dt")
    print(f"Integrand: f(x(t), y(t)) * ds/dt = {integrand}")
    print(f"Line integral: ∫f(x,y) ds = {line_integral}")
    
    return x, y, f, line_integral

x, y, f, line_integral = line_integrals()

# Visualize line integral
def visualize_line_integral():
    t_vals = np.linspace(0, 1, 100)
    x_vals = t_vals
    y_vals = t_vals**2
    f_vals = x_vals + y_vals
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Curve in xy-plane
    ax1.plot(x_vals, y_vals, 'b-', linewidth=3, label='Curve C: x = t, y = t²')
    ax1.scatter(x_vals[0], y_vals[0], c='green', s=100, label='Start (0,0)')
    ax1.scatter(x_vals[-1], y_vals[-1], c='red', s=100, label='End (1,1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Curve C in xy-plane')
    ax1.legend()
    ax1.grid(True)
    
    # Function values along the curve
    ax2.plot(t_vals, f_vals, 'r-', linewidth=2, label='f(x(t), y(t)) = t + t²')
    ax2.fill_between(t_vals, f_vals, alpha=0.3, color='red', 
                     label=f'Area = Line integral = {line_integral:.4f}')
    ax2.set_xlabel('Parameter t')
    ax2.set_ylabel('f(x(t), y(t))')
    ax2.set_title('Function Values Along Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_line_integral()
```

### Vector Line Integrals (Work)

```python
def vector_line_integrals():
    """Calculate vector line integrals (work)"""
    
    # Example: Work done by F = [x, y] along the curve C: x = t, y = t², 0 ≤ t ≤ 1
    
    t = sp.Symbol('t')
    
    # Vector field
    F_x = x
    F_y = y
    
    # Parametric curve
    x = t
    y = t**2
    
    # Tangent vector: dr/dt = [dx/dt, dy/dt]
    dx_dt = sp.diff(x, t)
    dy_dt = sp.diff(y, t)
    
    # Vector line integral: ∫F · dr = ∫F(x(t), y(t)) · [dx/dt, dy/dt] dt
    work_integrand = F_x.subs([(sp.Symbol('x'), x), (sp.Symbol('y'), y)]) * dx_dt + \
                    F_y.subs([(sp.Symbol('x'), x), (sp.Symbol('y'), y)]) * dy_dt
    
    work = sp.integrate(work_integrand, (t, 0, 1))
    
    print("Vector Line Integral (Work):")
    print(f"Vector field: F = [{F_x}, {F_y}]")
    print(f"Curve C: x = {x}, y = {y}, 0 ≤ t ≤ 1")
    print(f"Tangent vector: dr/dt = [{dx_dt}, {dy_dt}]")
    print(f"Integrand: F · dr/dt = {work_integrand}")
    print(f"Work: ∫F · dr = {work}")
    
    return F_x, F_y, work

F_x, F_y, work = vector_line_integrals()

# Visualize vector line integral
def visualize_vector_line_integral():
    t_vals = np.linspace(0, 1, 20)
    x_vals = t_vals
    y_vals = t_vals**2
    
    # Vector field values along the curve
    F_x_vals = x_vals
    F_y_vals = y_vals
    
    # Tangent vectors
    dx_dt_vals = np.ones_like(t_vals)
    dy_dt_vals = 2 * t_vals
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vector field and curve
    x_field = np.linspace(0, 1, 10)
    y_field = np.linspace(0, 1, 10)
    X_field, Y_field = np.meshgrid(x_field, y_field)
    U_field = X_field
    V_field = Y_field
    
    ax1.quiver(X_field, Y_field, U_field, V_field, angles='xy', scale_units='xy', scale=1, alpha=0.5)
    ax1.plot(x_vals, y_vals, 'r-', linewidth=3, label='Curve C')
    
    # Vector field at curve points
    ax1.quiver(x_vals, y_vals, F_x_vals, F_y_vals, angles='xy', scale_units='xy', scale=0.5, 
               color='blue', alpha=0.7, label='Vector field F')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Vector Field and Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Work calculation
    work_vals = F_x_vals * dx_dt_vals + F_y_vals * dy_dt_vals
    ax2.plot(t_vals, work_vals, 'g-', linewidth=2, label='F · dr/dt')
    ax2.fill_between(t_vals, work_vals, alpha=0.3, color='green', 
                     label=f'Work = {work:.4f}')
    ax2.set_xlabel('Parameter t')
    ax2.set_ylabel('F · dr/dt')
    ax2.set_title('Work Calculation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_vector_line_integral()
```

## 7.4 Conservative Vector Fields

### Definition and Properties

```python
def conservative_fields():
    """Analyze conservative vector fields"""
    
    x, y = sp.symbols('x y')
    
    # Example 1: Conservative field F = [x, y]
    F1 = [x, y]
    
    # Check if conservative: ∂Fy/∂x = ∂Fx/∂y
    dF1y_dx = sp.diff(F1[1], x)
    dF1x_dy = sp.diff(F1[0], y)
    
    is_conservative1 = dF1y_dx == dF1x_dy
    
    # Potential function: ∇φ = F
    # ∂φ/∂x = x, ∂φ/∂y = y
    # φ = x²/2 + y²/2 + C
    phi1 = x**2/2 + y**2/2
    
    # Example 2: Non-conservative field F = [-y, x]
    F2 = [-y, x]
    dF2y_dx = sp.diff(F2[1], x)
    dF2x_dy = sp.diff(F2[0], y)
    is_conservative2 = dF2y_dx == dF2x_dy
    
    print("Conservative Vector Fields:")
    print(f"F1 = {F1}")
    print(f"∂F1y/∂x = {dF1y_dx}")
    print(f"∂F1x/∂y = {dF1x_dy}")
    print(f"Conservative: {is_conservative1}")
    print(f"Potential function: φ = {phi1}")
    
    print(f"\nF2 = {F2}")
    print(f"∂F2y/∂x = {dF2y_dx}")
    print(f"∂F2x/∂y = {dF2x_dy}")
    print(f"Conservative: {is_conservative2}")
    
    return F1, F2, phi1, is_conservative1, is_conservative2

F1, F2, phi1, is_conservative1, is_conservative2 = conservative_fields()

# Visualize conservative vs non-conservative fields
def visualize_conservative_fields():
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conservative field F = [x, y]
    U1 = X
    V1 = Y
    ax1.quiver(X, Y, U1, V1, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax1.set_title('Conservative Field: F = [x, y]\n∇ × F = 0')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Non-conservative field F = [-y, x]
    U2 = -Y
    V2 = X
    ax2.quiver(X, Y, U2, V2, angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax2.set_title('Non-conservative Field: F = [-y, x]\n∇ × F ≠ 0')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_conservative_fields()
```

## 7.5 Applications in Physics

### Electric and Magnetic Fields

```python
def electromagnetic_fields():
    """Model electromagnetic fields using vector calculus"""
    
    # Example: Electric field of a point charge
    # E = k * q * r / |r|³ where r = [x, y, z]
    
    def electric_field_point_charge(x, y, z, q=1, k=1):
        r_mag = np.sqrt(x**2 + y**2 + z**2)
        if r_mag == 0:
            return 0, 0, 0
        return k * q * x / r_mag**3, k * q * y / r_mag**3, k * q * z / r_mag**3
    
    # Example: Magnetic field of a current-carrying wire
    # B = μ₀ * I / (2πr) * [-y/r, x/r, 0] where r = sqrt(x² + y²)
    
    def magnetic_field_wire(x, y, z, I=1, mu0=1):
        r = np.sqrt(x**2 + y**2)
        if r == 0:
            return 0, 0, 0
        return -mu0 * I * y / (2 * np.pi * r**2), mu0 * I * x / (2 * np.pi * r**2), 0
    
    return electric_field_point_charge, magnetic_field_wire

electric_field_point_charge, magnetic_field_wire = electromagnetic_fields()

# Visualize electromagnetic fields
def visualize_electromagnetic_fields():
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    z = 0  # 2D slice
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Electric field
    ax1 = fig.add_subplot(121)
    E_x, E_y, E_z = electric_field_point_charge(X, Y, z)
    ax1.quiver(X, Y, E_x, E_y, angles='xy', scale_units='xy', scale=0.5, alpha=0.7)
    ax1.scatter(0, 0, c='red', s=200, label='Point charge')
    ax1.set_title('Electric Field of Point Charge')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    
    # Magnetic field
    ax2 = fig.add_subplot(122)
    B_x, B_y, B_z = magnetic_field_wire(X, Y, z)
    ax2.quiver(X, Y, B_x, B_y, angles='xy', scale_units='xy', scale=0.5, alpha=0.7)
    ax2.scatter(0, 0, c='blue', s=200, marker='o', label='Current wire')
    ax2.set_title('Magnetic Field of Current Wire')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_electromagnetic_fields()
```

## 7.6 Applications in Machine Learning

### Gradient Flows

```python
def gradient_flows():
    """Implement gradient flows for optimization"""
    
    # Example: Gradient flow for f(x,y) = x² + y²
    def objective_function(x, y):
        return x**2 + y**2
    
    def gradient_function(x, y):
        return np.array([2*x, 2*y])
    
    # Gradient flow: dx/dt = -∇f(x)
    def gradient_flow(initial_point, dt=0.01, steps=1000):
        x, y = initial_point
        trajectory = [(x, y)]
        
        for i in range(steps):
            grad = gradient_function(x, y)
            x = x - dt * grad[0]
            y = y - dt * grad[1]
            trajectory.append((x, y))
            
            # Stop if gradient is small
            if np.linalg.norm(grad) < 1e-6:
                break
        
        return np.array(trajectory)
    
    # Run gradient flow
    initial_point = (2.0, 2.0)
    trajectory = gradient_flow(initial_point)
    
    print("Gradient Flow:")
    print(f"Initial point: {initial_point}")
    print(f"Final point: {trajectory[-1]}")
    print(f"Final value: {objective_function(trajectory[-1, 0], trajectory[-1, 1]):.6f}")
    print(f"Number of steps: {len(trajectory)}")
    
    return trajectory, objective_function

trajectory, objective_function = gradient_flows()

# Visualize gradient flow
def visualize_gradient_flow():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Gradient flow trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Gradient flow')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='red', s=100, label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='green', s=100, label='End')
    
    # Gradient field
    x_field = np.linspace(-3, 3, 10)
    y_field = np.linspace(-3, 3, 10)
    X_field, Y_field = np.meshgrid(x_field, y_field)
    U_field = 2 * X_field
    V_field = 2 * Y_field
    
    plt.quiver(X_field, Y_field, U_field, V_field, angles='xy', scale_units='xy', scale=1, alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Flow on f(x,y) = x² + y²')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_gradient_flow()
```

## Summary

- **Vector Fields**: Functions that assign vectors to points in space
- **Divergence**: Measures the "outflow" of a vector field at a point
- **Curl**: Measures the "rotation" of a vector field at a point
- **Line Integrals**: Integrals along curves, both scalar and vector
- **Conservative Fields**: Vector fields that are gradients of scalar functions
- **Applications**: Electromagnetism, fluid dynamics, gradient flows in optimization

## Next Steps

Understanding vector calculus enables you to work with complex physical systems, understand advanced optimization algorithms, and model phenomena in multiple dimensions. The next section covers numerical methods for when analytical solutions are not available. 