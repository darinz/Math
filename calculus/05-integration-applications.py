# Python code extracted from 05-integration-applications.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate

def area_between_curves():
    """
    Calculate area between curves using integration.
    Steps:
    1. Define the functions symbolically.
    2. Find intersection points (limits of integration).
    3. Integrate the difference to find the area.
    4. Repeat for trigonometric functions.
    """
    # Example 1: Area between y = x^2 and y = x
    x = sp.Symbol('x')
    f1 = x**2  # Lower function
    f2 = x     # Upper function
    
    # Find intersection points (solve x^2 = x)
    intersection_points = sp.solve(f1 - f2, x)
    print(f"Intersection points: {intersection_points}")
    
    # Calculate area: ∫[f2(x) - f1(x)]dx from a to b
    a, b = intersection_points[0], intersection_points[1]
    area = sp.integrate(f2 - f1, (x, a, b))
    print(f"Area between curves: {area}")
    
    # Example 2: Area between y = sin(x) and y = cos(x) from 0 to π
    f3 = sp.sin(x)
    f4 = sp.cos(x)
    
    # Find where sin(x) = cos(x) in [0, π]
    intersection_points2 = sp.solve(f3 - f4, x)
    intersection_points2 = [p for p in intersection_points2 if 0 <= p <= sp.pi]
    print(f"Intersection points in [0, π]: {intersection_points2}")
    
    # Calculate area
    area2 = sp.integrate(abs(f3 - f4), (x, 0, sp.pi))
    print(f"Area between sin(x) and cos(x) from 0 to π: {area2}")
    
    return f1, f2, f3, f4, intersection_points, area, area2

f1, f2, f3, f4, intersection_points, area, area2 = area_between_curves()

# Visualize area between curves
def visualize_area_between_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example 1: y = x^2 and y = x
    x_vals = np.linspace(-0.5, 1.5, 1000)
    y1_vals = x_vals**2
    y2_vals = x_vals
    
    ax1.plot(x_vals, y1_vals, 'b-', linewidth=2, label='y = x^2')
    ax1.plot(x_vals, y2_vals, 'r-', linewidth=2, label='y = x')
    
    # Fill area between curves
    mask = (x_vals >= intersection_points[0]) & (x_vals <= intersection_points[1])
    ax1.fill_between(x_vals[mask], y1_vals[mask], y2_vals[mask], 
                     alpha=0.3, color='green', label=f'Area = {area}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Area Between y = x^2 and y = x')
    ax1.legend()
    ax1.grid(True)
    
    # Example 2: y = sin(x) and y = cos(x)
    x_vals2 = np.linspace(0, np.pi, 1000)
    y3_vals = np.sin(x_vals2)
    y4_vals = np.cos(x_vals2)
    
    ax2.plot(x_vals2, y3_vals, 'b-', linewidth=2, label='y = sin(x)')
    ax2.plot(x_vals2, y4_vals, 'r-', linewidth=2, label='y = cos(x)')
    
    # Fill area between curves
    ax2.fill_between(x_vals2, np.minimum(y3_vals, y4_vals), np.maximum(y3_vals, y4_vals), 
                     alpha=0.3, color='green', label=f'Area = {area2:.4f}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Area Between y = sin(x) and y = cos(x)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_area_between_curves()

# Code Block 2
def volume_of_revolution():
    """Calculate volumes of solids of revolution"""
    
    # Example 1: Volume of revolution of y = x² from x = 0 to x = 2 around x-axis
    x = sp.Symbol('x')
    f = x**2
    a, b = 0, 2
    
    # Volume: V = π∫[f(x)]²dx
    volume = sp.integrate(sp.pi * f**2, (x, a, b))
    print(f"Volume of revolution of y = x² from x = 0 to x = 2: {volume}")
    
    # Example 2: Volume of revolution of y = √x from x = 0 to x = 4 around y-axis
    f2 = sp.sqrt(x)
    a2, b2 = 0, 4
    
    # Volume around y-axis: V = 2π∫x*f(x)dx
    volume2 = sp.integrate(2 * sp.pi * x * f2, (x, a2, b2))
    print(f"Volume of revolution of y = √x from x = 0 to x = 4 around y-axis: {volume2}")
    
    # Example 3: Volume between two curves rotated around x-axis
    # y = x and y = x² from x = 0 to x = 1
    f3 = x
    f4 = x**2
    a3, b3 = 0, 1
    
    # Volume: V = π∫[f3(x)² - f4(x)²]dx
    volume3 = sp.integrate(sp.pi * (f3**2 - f4**2), (x, a3, b3))
    print(f"Volume between y = x and y = x² rotated around x-axis: {volume3}")
    
    return f, f2, f3, f4, volume, volume2, volume3

f, f2, f3, f4, volume, volume2, volume3 = volume_of_revolution()

# Visualize solids of revolution
def visualize_solids_of_revolution():
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # Example 1: y = x² rotated around x-axis
    ax1 = fig.add_subplot(131, projection='3d')
    
    x = np.linspace(0, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = Y**2
    
    # Create surface of revolution
    theta = np.linspace(0, 2*np.pi, 50)
    X_rev = X
    Y_rev = Y * np.cos(theta[:, np.newaxis, np.newaxis])
    Z_rev = Y * np.sin(theta[:, np.newaxis, np.newaxis])
    
    ax1.plot_surface(X_rev, Y_rev, Z_rev, alpha=0.7, cmap='viridis')
    ax1.set_title(f'Volume = {volume:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Example 2: y = √x rotated around y-axis
    ax2 = fig.add_subplot(132, projection='3d')
    
    x = np.linspace(0, 4, 50)
    y = np.linspace(0, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X)
    
    # Create surface of revolution around y-axis
    theta = np.linspace(0, 2*np.pi, 50)
    X_rev = X * np.cos(theta[:, np.newaxis, np.newaxis])
    Y_rev = Y
    Z_rev = X * np.sin(theta[:, np.newaxis, np.newaxis])
    
    ax2.plot_surface(X_rev, Y_rev, Z_rev, alpha=0.7, cmap='plasma')
    ax2.set_title(f'Volume = {volume2:.2f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    # Example 3: Volume between curves
    ax3 = fig.add_subplot(133, projection='3d')
    
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z1 = X
    Z2 = X**2
    
    # Create surfaces
    theta = np.linspace(0, 2*np.pi, 50)
    X_rev1 = X
    Y_rev1 = Z1 * np.cos(theta[:, np.newaxis, np.newaxis])
    Z_rev1 = Z1 * np.sin(theta[:, np.newaxis, np.newaxis])
    
    X_rev2 = X
    Y_rev2 = Z2 * np.cos(theta[:, np.newaxis, np.newaxis])
    Z_rev2 = Z2 * np.sin(theta[:, np.newaxis, np.newaxis])
    
    ax3.plot_surface(X_rev1, Y_rev1, Z_rev1, alpha=0.7, color='blue')
    ax3.plot_surface(X_rev2, Y_rev2, Z_rev2, alpha=0.7, color='red')
    ax3.set_title(f'Volume = {volume3:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    
    plt.tight_layout()
    plt.show()

visualize_solids_of_revolution()

# Code Block 3
def work_calculations():
    """Calculate work done by variable forces"""
    
    # Example 1: Work done by force F(x) = 2x + 1 from x = 0 to x = 5
    x = sp.Symbol('x')
    F = 2*x + 1
    a, b = 0, 5
    
    # Work: W = ∫F(x)dx
    work = sp.integrate(F, (x, a, b))
    print(f"Work done by F(x) = 2x + 1 from x = 0 to x = 5: {work}")
    
    # Example 2: Work done by spring force F(x) = -kx
    k = 10  # spring constant
    F_spring = -k * x
    displacement = 2  # meters
    
    work_spring = sp.integrate(F_spring, (x, 0, displacement))
    print(f"Work done by spring force F(x) = -{k}x from x = 0 to x = {displacement}: {work_spring}")
    
    # Example 3: Work done against gravity
    # Force = mg, where m = 2 kg, g = 9.8 m/s²
    m, g = 2, 9.8
    F_gravity = m * g
    height = 10  # meters
    
    work_gravity = sp.integrate(F_gravity, (x, 0, height))
    print(f"Work done lifting {m} kg object {height} m against gravity: {work_gravity} J")
    
    return F, F_spring, F_gravity, work, work_spring, work_gravity

F, F_spring, F_gravity, work, work_spring, work_gravity = work_calculations()

# Visualize work calculations
def visualize_work():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example 1: Variable force
    x_vals = np.linspace(0, 5, 1000)
    F_vals = 2 * x_vals + 1
    
    ax1.plot(x_vals, F_vals, 'b-', linewidth=2, label='F(x) = 2x + 1')
    ax1.fill_between(x_vals, F_vals, alpha=0.3, color='blue', label=f'Work = {work}')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Work Done by Variable Force')
    ax1.legend()
    ax1.grid(True)
    
    # Example 2: Spring force
    x_vals2 = np.linspace(0, 2, 1000)
    F_spring_vals = -10 * x_vals2
    
    ax2.plot(x_vals2, F_spring_vals, 'r-', linewidth=2, label='F(x) = -10x')
    ax2.fill_between(x_vals2, F_spring_vals, alpha=0.3, color='red', label=f'Work = {work_spring}')
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Force (N)')
    ax2.set_title('Work Done by Spring Force')
    ax2.legend()
    ax2.grid(True)
    
    # Example 3: Gravitational force
    x_vals3 = np.linspace(0, 10, 1000)
    F_gravity_vals = 2 * 9.8 * np.ones_like(x_vals3)
    
    ax3.plot(x_vals3, F_gravity_vals, 'g-', linewidth=2, label='F = mg = 19.6 N')
    ax3.fill_between(x_vals3, F_gravity_vals, alpha=0.3, color='green', label=f'Work = {work_gravity:.1f} J')
    ax3.set_xlabel('Height (m)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Work Done Against Gravity')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_work()

# Code Block 4
def probability_integration():
    """Calculate probabilities using integration"""
    
    # Example 1: Normal distribution probabilities
    def normal_pdf(x, mu=0, sigma=1):
        return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    
    # P(-1 ≤ X ≤ 1) for standard normal
    prob_1sigma, _ = integrate.quad(normal_pdf, -1, 1)
    print(f"P(-1 ≤ X ≤ 1) for standard normal: {prob_1sigma:.4f}")
    
    # P(-2 ≤ X ≤ 2) for standard normal
    prob_2sigma, _ = integrate.quad(normal_pdf, -2, 2)
    print(f"P(-2 ≤ X ≤ 2) for standard normal: {prob_2sigma:.4f}")
    
    # Example 2: Exponential distribution
    def exponential_pdf(x, lambda_param=1):
        return lambda_param * np.exp(-lambda_param * x)
    
    # P(X > 2) for exponential with λ = 1
    prob_exponential, _ = integrate.quad(exponential_pdf, 2, np.inf, args=(1,))
    print(f"P(X > 2) for exponential(λ=1): {prob_exponential:.4f}")
    
    # Example 3: Expected value calculations
    def expected_value_exponential(lambda_param=1):
        integrand = lambda x: x * exponential_pdf(x, lambda_param)
        result, _ = integrate.quad(integrand, 0, np.inf)
        return result
    
    expected_val = expected_value_exponential(1)
    print(f"E[X] for exponential(λ=1): {expected_val:.4f}")
    
    return prob_1sigma, prob_2sigma, prob_exponential, expected_val

prob_1sigma, prob_2sigma, prob_exponential, expected_val = probability_integration()

# Visualize probability distributions
def visualize_probability_distributions():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Normal distribution
    x_normal = np.linspace(-4, 4, 1000)
    y_normal = normal_pdf(x_normal)
    
    ax1.plot(x_normal, y_normal, 'b-', linewidth=2, label='Standard Normal')
    mask_1sigma = (x_normal >= -1) & (x_normal <= 1)
    ax1.fill_between(x_normal[mask_1sigma], y_normal[mask_1sigma], alpha=0.5, color='red',
                     label=f'P(-1 ≤ X ≤ 1) = {prob_1sigma:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Normal Distribution')
    ax1.legend()
    ax1.grid(True)
    
    # Normal distribution ±2σ
    ax2.plot(x_normal, y_normal, 'b-', linewidth=2, label='Standard Normal')
    mask_2sigma = (x_normal >= -2) & (x_normal <= 2)
    ax2.fill_between(x_normal[mask_2sigma], y_normal[mask_2sigma], alpha=0.5, color='green',
                     label=f'P(-2 ≤ X ≤ 2) = {prob_2sigma:.4f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Normal Distribution ±2σ')
    ax2.legend()
    ax2.grid(True)
    
    # Exponential distribution
    x_exp = np.linspace(0, 5, 1000)
    y_exp = exponential_pdf(x_exp, 1)
    
    ax3.plot(x_exp, y_exp, 'r-', linewidth=2, label='Exponential(λ=1)')
    mask_exp = x_exp > 2
    ax3.fill_between(x_exp[mask_exp], y_exp[mask_exp], alpha=0.5, color='orange',
                     label=f'P(X > 2) = {prob_exponential:.4f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.set_title('Exponential Distribution')
    ax3.legend()
    ax3.grid(True)
    
    # Expected value visualization
    x_expected = np.linspace(0, 5, 1000)
    y_expected = exponential_pdf(x_expected, 1)
    expected_line = expected_val * np.ones_like(x_expected)
    
    ax4.plot(x_expected, y_expected, 'r-', linewidth=2, label='Exponential(λ=1)')
    ax4.axvline(x=expected_val, color='g', linestyle='--', linewidth=2,
                label=f'E[X] = {expected_val:.4f}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Expected Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_probability_distributions()

# Code Block 5
def economic_surplus():
    """Calculate consumer and producer surplus using integration"""
    
    # Example: Market with demand and supply curves
    x = sp.Symbol('x')  # quantity
    p = sp.Symbol('p')  # price
    
    # Demand curve: p = 100 - 2x
    demand = 100 - 2*x
    
    # Supply curve: p = 20 + 3x
    supply = 20 + 3*x
    
    # Find equilibrium
    equilibrium_quantity = sp.solve(demand - supply, x)[0]
    equilibrium_price = demand.subs(x, equilibrium_quantity)
    
    print(f"Equilibrium quantity: {equilibrium_quantity}")
    print(f"Equilibrium price: {equilibrium_price}")
    
    # Consumer surplus: ∫[demand - equilibrium_price]dx from 0 to equilibrium_quantity
    consumer_surplus = sp.integrate(demand - equilibrium_price, (x, 0, equilibrium_quantity))
    
    # Producer surplus: ∫[equilibrium_price - supply]dx from 0 to equilibrium_quantity
    producer_surplus = sp.integrate(equilibrium_price - supply, (x, 0, equilibrium_quantity))
    
    print(f"Consumer surplus: {consumer_surplus}")
    print(f"Producer surplus: {producer_surplus}")
    print(f"Total surplus: {consumer_surplus + producer_surplus}")
    
    return demand, supply, equilibrium_quantity, equilibrium_price, consumer_surplus, producer_surplus

demand, supply, eq_q, eq_p, cs, ps = economic_surplus()

# Visualize economic surplus
def visualize_economic_surplus():
    x_vals = np.linspace(0, 50, 1000)
    demand_vals = 100 - 2 * x_vals
    supply_vals = 20 + 3 * x_vals
    
    plt.figure(figsize=(12, 8))
    
    # Plot demand and supply curves
    plt.plot(x_vals, demand_vals, 'b-', linewidth=2, label='Demand: p = 100 - 2x')
    plt.plot(x_vals, supply_vals, 'r-', linewidth=2, label='Supply: p = 20 + 3x')
    
    # Mark equilibrium point
    plt.scatter(eq_q, eq_p, c='green', s=100, zorder=5, 
                label=f'Equilibrium: ({eq_q}, {eq_p})')
    
    # Fill consumer surplus (area above price, below demand)
    mask = x_vals <= eq_q
    plt.fill_between(x_vals[mask], demand_vals[mask], eq_p, alpha=0.3, color='blue',
                     label=f'Consumer Surplus = {cs:.2f}')
    
    # Fill producer surplus (area below price, above supply)
    plt.fill_between(x_vals[mask], eq_p, supply_vals[mask], alpha=0.3, color='red',
                     label=f'Producer Surplus = {ps:.2f}')
    
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.title('Consumer and Producer Surplus')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.show()

visualize_economic_surplus()

# Code Block 6
def center_of_mass_calculations():
    """Calculate center of mass using integration"""
    
    # Example: Center of mass of a thin rod with variable density
    x = sp.Symbol('x')
    
    # Density function: ρ(x) = 1 + x (kg/m)
    density = 1 + x
    length = 2  # meters
    
    # Total mass: M = ∫ρ(x)dx
    total_mass = sp.integrate(density, (x, 0, length))
    
    # First moment: ∫x*ρ(x)dx
    first_moment = sp.integrate(x * density, (x, 0, length))
    
    # Center of mass: x_cm = (∫x*ρ(x)dx) / (∫ρ(x)dx)
    center_of_mass = first_moment / total_mass
    
    print(f"Total mass: {total_mass} kg")
    print(f"First moment: {first_moment} kg·m")
    print(f"Center of mass: {center_of_mass} m")
    
    # Example: Center of mass of a semicircle
    # For a semicircle of radius R, center of mass is at (0, 4R/(3π))
    R = 1
    y_cm_semicircle = 4 * R / (3 * np.pi)
    print(f"Center of mass of semicircle (radius {R}): (0, {y_cm_semicircle:.4f})")
    
    return density, total_mass, center_of_mass, y_cm_semicircle

density, total_mass, center_of_mass, y_cm_semicircle = center_of_mass_calculations()

# Visualize center of mass
def visualize_center_of_mass():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Thin rod with variable density
    x_vals = np.linspace(0, 2, 1000)
    density_vals = 1 + x_vals
    
    ax1.plot(x_vals, density_vals, 'b-', linewidth=2, label='Density: ρ(x) = 1 + x')
    ax1.axvline(x=center_of_mass, color='red', linestyle='--', linewidth=2,
                label=f'Center of mass: x = {center_of_mass:.3f}')
    ax1.fill_between(x_vals, density_vals, alpha=0.3, color='blue')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Density (kg/m)')
    ax1.set_title('Thin Rod with Variable Density')
    ax1.legend()
    ax1.grid(True)
    
    # Semicircle
    theta = np.linspace(0, np.pi, 1000)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    ax2.plot(x_circle, y_circle, 'b-', linewidth=2, label='Semicircle')
    ax2.scatter(0, y_cm_semicircle, c='red', s=100, zorder=5,
                label=f'Center of mass: (0, {y_cm_semicircle:.3f})')
    ax2.fill(x_circle, y_circle, alpha=0.3, color='blue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Semicircle Center of Mass')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

visualize_center_of_mass()

# --- Advanced Applications: Signal Processing, ML Integrals, Numerical Methods ---
import scipy.fftpack

# 7.1 Signal Processing: Fourier Transform and Convolution

def fourier_transform_example():
    """Compute and visualize the Fourier transform of a signal (ML/signal processing context)."""
    # Example: Signal is a sum of two sine waves
    t = np.linspace(0, 1, 500)
    f1, f2 = 5, 20  # frequencies in Hz
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    # Compute FFT
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(t), d=t[1] - t[0])

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.stem(fft_freqs[:len(t)//2], np.abs(fft_vals)[:len(t)//2], use_line_collection=True)
    plt.title('Fourier Transform (Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

fourier_transform_example()


def convolution_example():
    """Compute and visualize convolution of two functions (signal/ML context)."""
    # Example: f(x) = exp(-x), g(x) = exp(-2x), for x >= 0
    x = np.linspace(0, 5, 500)
    f = np.exp(-x)
    g = np.exp(-2 * x)
    conv = np.convolve(f, g, mode='full') * (x[1] - x[0])
    x_conv = np.linspace(0, 2 * x[-1], len(conv))

    plt.figure(figsize=(10, 5))
    plt.plot(x, f, label='f(x) = exp(-x)')
    plt.plot(x, g, label='g(x) = exp(-2x)')
    plt.plot(x_conv, conv, label='(f * g)(t)', linewidth=2)
    plt.title('Convolution of f and g')
    plt.xlabel('x or t')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

convolution_example()

# 7.2 ML Integrals: Kernel, Activation, Loss

def kernel_integral_example():
    """Compute a simple kernel integral (SVM/ML context)."""
    # Example: Gaussian kernel normalization
    def gaussian_kernel(x, y, sigma=1):
        return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))
    # Integral over z: K(x, y) = ∫ k(x, z) k(z, y) dz
    x, y = 0, 1
    integrand = lambda z: gaussian_kernel(x, z) * gaussian_kernel(z, y)
    result, _ = integrate.quad(integrand, -np.inf, np.inf)
    print(f"Kernel integral (Gaussian, x=0, y=1): {result:.4f}")

kernel_integral_example()


def activation_integral_example():
    """Compute normalization constant for softmax-like activation."""
    # Z = ∫ exp(f(x)) dx, f(x) = x on [0, 1]
    integrand = lambda x: np.exp(x)
    Z, _ = integrate.quad(integrand, 0, 1)
    print(f"Softmax normalization constant Z = ∫₀¹ exp(x) dx: {Z:.4f}")

activation_integral_example()


def expected_loss_example():
    """Compute expected loss for a simple regression loss and normal distribution."""
    # L(y, y_hat) = (y - y_hat)^2, y ~ N(0, 1), y_hat = 0
    def normal_pdf(y):
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * y**2)
    integrand = lambda y: (y ** 2) * normal_pdf(y)
    E_L, _ = integrate.quad(integrand, -np.inf, np.inf)
    print(f"Expected squared loss (y_hat=0, y~N(0,1)): {E_L:.4f}")

expected_loss_example()

# 7.3 Numerical Integration Methods

def numerical_integration_examples():
    """Demonstrate trapezoidal, Simpson's, and Monte Carlo integration with ML context."""
    # Function: f(x) = exp(-x^2), integral over [0, 1]
    f = lambda x: np.exp(-x ** 2)
    a, b = 0, 1
    n = 1000
    x_vals = np.linspace(a, b, n + 1)
    y_vals = f(x_vals)
    # Trapezoidal Rule
    trapz_result = np.trapz(y_vals, x_vals)
    # Simpson's Rule
    from scipy.integrate import simps
    simps_result = simps(y_vals, x_vals)
    # Monte Carlo
    N = 100000
    x_rand = np.random.uniform(a, b, N)
    mc_result = (b - a) * np.mean(f(x_rand))
    print(f"Trapezoidal: {trapz_result:.6f}, Simpson's: {simps_result:.6f}, Monte Carlo: {mc_result:.6f}")
    # Visualization
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label='f(x) = exp(-x^2)')
    plt.fill_between(x_vals, y_vals, alpha=0.3)
    plt.title('Numerical Integration Example')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

numerical_integration_examples()

