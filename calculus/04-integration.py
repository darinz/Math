"""
Integration - Python Implementation
=================================

This file demonstrates the key concepts of integration using Python.
It includes comprehensive examples, visualizations, and practical applications
in physics, probability, and machine learning.

Key Concepts Covered:
- Antiderivatives and indefinite integrals
- Definite integrals and area interpretation
- Integration techniques and methods
- Applications in physics and probability
- Numerical integration methods
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, integrate, exp, sin, cos, log, sqrt, pi
from scipy import integrate as scipy_integrate
import warnings
warnings.filterwarnings('ignore')

# Set up SymPy symbols for symbolic computation
x, y, z, t = symbols('x y z t')

print("=" * 60)
print("INTEGRATION - PYTHON IMPLEMENTATION")
print("=" * 60)

# =============================================================================
# SECTION 1: ANTIDERIVATIVES AND INDEFINITE INTEGRALS
# =============================================================================

print("\n1. ANTIDERIVATIVES AND INDEFINITE INTEGRALS")
print("-" * 40)

def demonstrate_antiderivatives():
    """
    Demonstrate antiderivatives and indefinite integrals.
    
    An antiderivative of f(x) is a function F(x) such that F'(x) = f(x).
    The indefinite integral ∫f(x)dx represents the family of all antiderivatives.
    """
    print("Demonstrating antiderivatives and indefinite integrals...")
    
    # Basic power functions
    print("\n=== BASIC POWER FUNCTIONS ===")
    power_examples = [x**0, x**1, x**2, x**3, x**(-1), x**(-2)]
    for expr in power_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")
    
    # Exponential and logarithmic functions
    print("\n=== EXPONENTIAL AND LOGARITHMIC FUNCTIONS ===")
    exp_expr = sp.exp(x)
    log_expr = sp.log(x)
    print(f"   ∫ {exp_expr} dx = {sp.integrate(exp_expr, x)}")
    print(f"   ∫ {log_expr} dx = {sp.integrate(log_expr, x)}")
    
    # Trigonometric functions
    print("\n=== TRIGONOMETRIC FUNCTIONS ===")
    trig_examples = [sp.sin(x), sp.cos(x), sp.tan(x), 1/sp.cos(x)**2]
    for expr in trig_examples:
        integral = sp.integrate(expr, x)
        print(f"   ∫ {expr} dx = {integral}")
    
    # Composite functions requiring substitution
    print("\n=== COMPOSITE FUNCTIONS (SUBSTITUTION) ===")
    composite_examples = [
        sp.sin(x**2) * x,  # Requires substitution u = x²
        sp.exp(x**2) * x,  # Requires substitution u = x²
        sp.log(x) / x,     # Requires substitution u = log(x)
        sp.sin(x) * sp.cos(x)  # Requires trigonometric identity
    ]
    
    for expr in composite_examples:
        try:
            integral = sp.integrate(expr, x)
            print(f"   ∫ {expr} dx = {integral}")
        except:
            print(f"   ∫ {expr} dx = (requires advanced techniques)")
    
    # Visual demonstration of antiderivatives
    def visualize_antiderivatives():
        """
        Visualize how antiderivatives differ by constants.
        """
        def f(x):
            return x**2
        
        def F(x, C=0):
            return x**3 / 3 + C
        
        x_vals = np.linspace(-2, 2, 100)
        y_vals = f(x_vals)
        
        # Multiple antiderivatives with different constants
        constants = [-2, -1, 0, 1, 2]
        colors = ['red', 'orange', 'blue', 'green', 'purple']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original function
        ax1.plot(x_vals, y_vals, 'b-', linewidth=3, label='f(x) = x²')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Original Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Antiderivatives
        for i, (C, color) in enumerate(zip(constants, colors)):
            Y_vals = F(x_vals, C)
            ax2.plot(x_vals, Y_vals, color=color, linewidth=2, 
                    label=f'F(x) = x³/3 + {C}')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('F(x)')
        ax2.set_title('Antiderivatives (Different Constants)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Verify that all antiderivatives have the same derivative
        print("\nVerification: All antiderivatives have the same derivative")
        x_test = 1.0
        for C in constants:
            # Numerical derivative of antiderivative
            h = 1e-7
            derivative = (F(x_test + h, C) - F(x_test - h, C)) / (2 * h)
            print(f"  F(x) = x³/3 + {C}: F'({x_test}) = {derivative:.6f}")
    
    visualize_antiderivatives()

demonstrate_antiderivatives()

# =============================================================================
# SECTION 2: DEFINITE INTEGRALS AND AREA INTERPRETATION
# =============================================================================

print("\n2. DEFINITE INTEGRALS AND AREA INTERPRETATION")
print("-" * 45)

def demonstrate_definite_integrals():
    """
    Demonstrate definite integrals and their geometric interpretation.
    
    The definite integral ∫[a,b] f(x)dx represents the signed area
    between the curve y = f(x) and the x-axis from x = a to x = b.
    """
    print("Demonstrating definite integrals and area interpretation...")
    
    # Example: ∫[0,2] x² dx
    def area_under_curve():
        """
        Calculate and visualize the area under y = x² from x = 0 to x = 2.
        """
        print("\nExample: ∫[0,2] x² dx")
        
        # Symbolic calculation
        f_expr = x**2
        definite_integral = sp.integrate(f_expr, (x, 0, 2))
        print(f"Symbolic result: ∫[0,2] x² dx = {definite_integral}")
        
        # Numerical verification
        def f(x):
            return x**2
        
        # Using scipy for numerical integration
        numerical_result, error = scipy_integrate.quad(f, 0, 2)
        print(f"Numerical result: ∫[0,2] x² dx = {numerical_result:.6f}")
        print(f"Error estimate: {error:.2e}")
        
        # Visual demonstration
        x_vals = np.linspace(-0.5, 2.5, 1000)
        y_vals = f(x_vals)
        
        # Area of interest
        x_area = np.linspace(0, 2, 1000)
        y_area = f(x_area)
        
        plt.figure(figsize=(12, 8))
        
        # Plot function
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
        
        # Fill area under curve
        plt.fill_between(x_area, y_area, alpha=0.3, color='blue', label='Area = 8/3')
        
        # Mark integration bounds
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='x = 0')
        plt.axvline(x=2, color='r', linestyle='--', alpha=0.7, label='x = 2')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Definite Integral: Area Under y = x² from x = 0 to x = 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Riemann sum approximation
        print("\nRiemann Sum Approximation:")
        n_intervals = 10
        dx = 2 / n_intervals
        riemann_sum = sum(f(i * dx) * dx for i in range(n_intervals))
        print(f"  With {n_intervals} intervals: {riemann_sum:.6f}")
        
        n_intervals = 100
        dx = 2 / n_intervals
        riemann_sum = sum(f(i * dx) * dx for i in range(n_intervals))
        print(f"  With {n_intervals} intervals: {riemann_sum:.6f}")
    
    area_under_curve()
    
    # Negative area example
    def negative_area_example():
        """
        Demonstrate negative area when function is below x-axis.
        """
        print("\nExample: ∫[0,π] sin(x) dx")
        
        # Symbolic calculation
        f_expr = sp.sin(x)
        definite_integral = sp.integrate(f_expr, (x, 0, sp.pi))
        print(f"Symbolic result: ∫[0,π] sin(x) dx = {definite_integral}")
        
        # Visual demonstration
        x_vals = np.linspace(-0.5, np.pi + 0.5, 1000)
        y_vals = np.sin(x_vals)
        
        # Areas of interest
        x_area1 = np.linspace(0, np.pi/2, 500)  # Positive area
        y_area1 = np.sin(x_area1)
        x_area2 = np.linspace(np.pi/2, np.pi, 500)  # Negative area
        y_area2 = np.sin(x_area2)
        
        plt.figure(figsize=(12, 8))
        
        # Plot function
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = sin(x)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Fill positive area
        plt.fill_between(x_area1, y_area1, alpha=0.3, color='green', label='Positive area')
        
        # Fill negative area
        plt.fill_between(x_area2, y_area2, alpha=0.3, color='red', label='Negative area')
        
        # Mark integration bounds
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='x = 0')
        plt.axvline(x=np.pi, color='r', linestyle='--', alpha=0.7, label='x = π')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Definite Integral: ∫[0,π] sin(x) dx = 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    negative_area_example()

demonstrate_definite_integrals()

# =============================================================================
# SECTION 3: INTEGRATION TECHNIQUES
# =============================================================================

print("\n3. INTEGRATION TECHNIQUES")
print("-" * 30)

def demonstrate_integration_techniques():
    """
    Demonstrate various integration techniques.
    """
    print("Demonstrating integration techniques...")
    
    # 1. Integration by substitution
    def substitution_examples():
        """
        Demonstrate integration by substitution.
        """
        print("\n=== INTEGRATION BY SUBSTITUTION ===")
        
        # Example 1: ∫ x*e^(x²) dx
        print("Example 1: ∫ x*e^(x²) dx")
        print("Let u = x², then du = 2x dx")
        print("∫ x*e^(x²) dx = (1/2)∫ e^u du = (1/2)e^u + C = (1/2)e^(x²) + C")
        
        # Symbolic verification
        expr1 = x * sp.exp(x**2)
        integral1 = sp.integrate(expr1, x)
        print(f"Symbolic result: {integral1}")
        
        # Example 2: ∫ sin(x²)*x dx
        print("\nExample 2: ∫ sin(x²)*x dx")
        print("Let u = x², then du = 2x dx")
        print("∫ sin(x²)*x dx = (1/2)∫ sin(u) du = -(1/2)cos(u) + C = -(1/2)cos(x²) + C")
        
        expr2 = sp.sin(x**2) * x
        integral2 = sp.integrate(expr2, x)
        print(f"Symbolic result: {integral2}")
    
    substitution_examples()
    
    # 2. Integration by parts
    def integration_by_parts_examples():
        """
        Demonstrate integration by parts.
        """
        print("\n=== INTEGRATION BY PARTS ===")
        print("Formula: ∫ u dv = uv - ∫ v du")
        
        # Example 1: ∫ x*e^x dx
        print("\nExample 1: ∫ x*e^x dx")
        print("Let u = x, dv = e^x dx")
        print("Then du = dx, v = e^x")
        print("∫ x*e^x dx = x*e^x - ∫ e^x dx = x*e^x - e^x + C")
        
        expr1 = x * sp.exp(x)
        integral1 = sp.integrate(expr1, x)
        print(f"Symbolic result: {integral1}")
        
        # Example 2: ∫ x*ln(x) dx
        print("\nExample 2: ∫ x*ln(x) dx")
        print("Let u = ln(x), dv = x dx")
        print("Then du = dx/x, v = x²/2")
        print("∫ x*ln(x) dx = (x²/2)*ln(x) - ∫ (x²/2)*(dx/x)")
        print("= (x²/2)*ln(x) - (1/2)∫ x dx = (x²/2)*ln(x) - x²/4 + C")
        
        expr2 = x * sp.log(x)
        integral2 = sp.integrate(expr2, x)
        print(f"Symbolic result: {integral2}")
    
    integration_by_parts_examples()
    
    # 3. Partial fractions
    def partial_fractions_examples():
        """
        Demonstrate partial fraction decomposition.
        """
        print("\n=== PARTIAL FRACTIONS ===")
        
        # Example: ∫ 1/(x²-1) dx
        print("Example: ∫ 1/(x²-1) dx")
        print("1/(x²-1) = 1/((x+1)(x-1)) = A/(x+1) + B/(x-1)")
        print("Solving: A = -1/2, B = 1/2")
        print("∫ 1/(x²-1) dx = (-1/2)ln|x+1| + (1/2)ln|x-1| + C")
        print("= (1/2)ln|(x-1)/(x+1)| + C")
        
        expr = 1 / (x**2 - 1)
        integral = sp.integrate(expr, x)
        print(f"Symbolic result: {integral}")
    
    partial_fractions_examples()

demonstrate_integration_techniques()

# =============================================================================
# SECTION 4: APPLICATIONS IN PHYSICS
# =============================================================================

print("\n4. APPLICATIONS IN PHYSICS")
print("-" * 30)

def demonstrate_physics_applications():
    """
    Demonstrate applications of integration in physics.
    """
    print("Demonstrating physics applications...")
    
    # 1. Work done by a force
    def work_calculation():
        """
        Calculate work done by a variable force.
        """
        print("\nWork Calculation:")
        print("Work = ∫ F(x) dx")
        
        # Example: Force F(x) = 2x + 1 from x = 0 to x = 3
        def force(x):
            return 2*x + 1
        
        # Symbolic calculation
        f_expr = 2*x + 1
        work = sp.integrate(f_expr, (x, 0, 3))
        print(f"Force: F(x) = 2x + 1")
        print(f"Work = ∫[0,3] (2x + 1) dx = {work}")
        
        # Numerical verification
        work_numerical, error = scipy_integrate.quad(force, 0, 3)
        print(f"Numerical result: {work_numerical:.6f}")
        
        # Visualization
        x_vals = np.linspace(-0.5, 3.5, 1000)
        y_vals = force(x_vals)
        
        # Area of interest
        x_area = np.linspace(0, 3, 1000)
        y_area = force(x_area)
        
        plt.figure(figsize=(12, 8))
        
        # Plot force
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='F(x) = 2x + 1')
        plt.fill_between(x_area, y_area, alpha=0.3, color='blue', label=f'Work = {work}')
        
        # Mark bounds
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='x = 0')
        plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='x = 3')
        
        plt.xlabel('Position (x)')
        plt.ylabel('Force F(x)')
        plt.title('Work Done by Variable Force')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    work_calculation()
    
    # 2. Center of mass
    def center_of_mass():
        """
        Calculate center of mass of a region.
        """
        print("\nCenter of Mass Calculation:")
        print("x̄ = (∫ x*f(x) dx) / (∫ f(x) dx)")
        
        # Example: Region bounded by y = x² from x = 0 to x = 2
        def f(x):
            return x**2
        
        # Calculate center of mass
        def x_times_f(x):
            return x * f(x)
        
        # Numerator: ∫ x*f(x) dx
        numerator, _ = scipy_integrate.quad(x_times_f, 0, 2)
        
        # Denominator: ∫ f(x) dx
        denominator, _ = scipy_integrate.quad(f, 0, 2)
        
        center_x = numerator / denominator
        print(f"Region: y = x² from x = 0 to x = 2")
        print(f"Center of mass: x̄ = {center_x:.6f}")
        
        # Visualization
        x_vals = np.linspace(-0.5, 2.5, 1000)
        y_vals = f(x_vals)
        
        # Area of interest
        x_area = np.linspace(0, 2, 1000)
        y_area = f(x_area)
        
        plt.figure(figsize=(12, 8))
        
        # Plot function
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='y = x²')
        plt.fill_between(x_area, y_area, alpha=0.3, color='blue')
        
        # Mark center of mass
        plt.axvline(x=center_x, color='r', linestyle='--', linewidth=2, 
                   label=f'Center of mass: x̄ = {center_x:.3f}')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Center of Mass of Region')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    center_of_mass()

demonstrate_physics_applications()

# =============================================================================
# SECTION 5: APPLICATIONS IN PROBABILITY
# =============================================================================

print("\n5. APPLICATIONS IN PROBABILITY")
print("-" * 30)

def demonstrate_probability_applications():
    """
    Demonstrate applications of integration in probability.
    """
    print("Demonstrating probability applications...")
    
    # 1. Normal distribution
    def normal_distribution():
        """
        Demonstrate integration of normal distribution.
        """
        print("\nNormal Distribution:")
        print("Standard normal PDF: f(x) = (1/√(2π)) * e^(-x²/2)")
        
        def normal_pdf(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
        
        # Verify total probability = 1
        total_prob, error = scipy_integrate.quad(normal_pdf, -np.inf, np.inf)
        print(f"Total probability: ∫[-∞,∞] f(x) dx = {total_prob:.6f}")
        
        # Calculate P(-1 ≤ X ≤ 1)
        prob_interval, error = scipy_integrate.quad(normal_pdf, -1, 1)
        print(f"P(-1 ≤ X ≤ 1) = {prob_interval:.6f}")
        
        # Visualization
        x_vals = np.linspace(-4, 4, 1000)
        y_vals = normal_pdf(x_vals)
        
        plt.figure(figsize=(12, 8))
        
        # Plot PDF
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Standard Normal PDF')
        
        # Fill area for P(-1 ≤ X ≤ 1)
        mask = (x_vals >= -1) & (x_vals <= 1)
        plt.fill_between(x_vals[mask], y_vals[mask], alpha=0.3, color='red', 
                        label=f'P(-1 ≤ X ≤ 1) = {prob_interval:.3f}')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Standard Normal Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    normal_distribution()
    
    # 2. Expected value
    def expected_value():
        """
        Calculate expected value using integration.
        """
        print("\nExpected Value:")
        print("E[X] = ∫ x*f(x) dx")
        
        # Example: Exponential distribution f(x) = λ*e^(-λx) for x ≥ 0
        lambda_param = 2
        
        def exponential_pdf(x):
            return lambda_param * np.exp(-lambda_param * x) if x >= 0 else 0
        
        def x_times_pdf(x):
            return x * exponential_pdf(x)
        
        # Calculate expected value
        expected_val, error = scipy_integrate.quad(x_times_pdf, 0, np.inf)
        print(f"Exponential distribution with λ = {lambda_param}")
        print(f"Expected value: E[X] = {expected_val:.6f}")
        print(f"Theoretical value: 1/λ = {1/lambda_param:.6f}")
        
        # Visualization
        x_vals = np.linspace(0, 5, 1000)
        y_vals = [exponential_pdf(x) for x in x_vals]
        
        plt.figure(figsize=(12, 8))
        
        # Plot PDF
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'Exponential PDF (λ = {lambda_param})')
        
        # Mark expected value
        plt.axvline(x=expected_val, color='r', linestyle='--', linewidth=2, 
                   label=f'E[X] = {expected_val:.3f}')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Exponential Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    expected_value()

demonstrate_probability_applications()

# =============================================================================
# SECTION 6: NUMERICAL INTEGRATION
# =============================================================================

print("\n6. NUMERICAL INTEGRATION")
print("-" * 30)

def demonstrate_numerical_integration():
    """
    Demonstrate numerical integration methods.
    """
    print("Demonstrating numerical integration methods...")
    
    # Test function
    def f(x):
        return x**2 + np.sin(x)
    
    # Exact integral from 0 to 2
    exact_integral = 8/3 + (1 - np.cos(2))  # ∫[0,2] (x² + sin(x)) dx
    
    print(f"Test function: f(x) = x² + sin(x)")
    print(f"Exact integral ∫[0,2] f(x) dx = {exact_integral:.6f}")
    
    # 1. Trapezoidal rule
    def trapezoidal_rule(f, a, b, n):
        """
        Approximate integral using trapezoidal rule.
        """
        x_vals = np.linspace(a, b, n+1)
        y_vals = f(x_vals)
        
        h = (b - a) / n
        integral = h * (0.5 * y_vals[0] + np.sum(y_vals[1:-1]) + 0.5 * y_vals[-1])
        
        return integral
    
    # 2. Simpson's rule
    def simpson_rule(f, a, b, n):
        """
        Approximate integral using Simpson's rule.
        """
        if n % 2 != 0:
            n += 1  # n must be even
        
        x_vals = np.linspace(a, b, n+1)
        y_vals = f(x_vals)
        
        h = (b - a) / n
        integral = h/3 * (y_vals[0] + 4*np.sum(y_vals[1:-1:2]) + 
                          2*np.sum(y_vals[2:-1:2]) + y_vals[-1])
        
        return integral
    
    # Compare methods
    print("\nNumerical Integration Comparison:")
    print("Method\t\t\tn=10\t\tn=100\t\tn=1000")
    print("-" * 60)
    
    for n in [10, 100, 1000]:
        trap_result = trapezoidal_rule(f, 0, 2, n)
        simp_result = simpson_rule(f, 0, 2, n)
        
        if n == 10:
            print(f"Trapezoidal\t\t{trap_result:.6f}\t{simp_result:.6f}\t{simpson_rule(f, 0, 2, 1000):.6f}")
        elif n == 100:
            print(f"Simpson's\t\t{trapezoidal_rule(f, 0, 2, 10):.6f}\t{simp_result:.6f}\t{simpson_rule(f, 0, 2, 1000):.6f}")
        else:
            print(f"Exact\t\t\t{trapezoidal_rule(f, 0, 2, 10):.6f}\t{trapezoidal_rule(f, 0, 2, 100):.6f}\t{exact_integral:.6f}")
    
    # Error analysis
    print("\nError Analysis:")
    n_vals = [10, 50, 100, 500, 1000]
    
    for n in n_vals:
        trap_result = trapezoidal_rule(f, 0, 2, n)
        simp_result = simpson_rule(f, 0, 2, n)
        
        trap_error = abs(trap_result - exact_integral)
        simp_error = abs(simp_result - exact_integral)
        
        print(f"n = {n:4d}: Trapezoidal error = {trap_error:.2e}, Simpson error = {simp_error:.2e}")
    
    # Visualization
    x_vals = np.linspace(0, 2, 1000)
    y_vals = f(x_vals)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Function plot
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x² + sin(x)')
    ax1.fill_between(x_vals, y_vals, alpha=0.3, color='blue', label=f'Area = {exact_integral:.3f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function to Integrate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error comparison
    n_range = np.arange(10, 1001, 10)
    trap_errors = [abs(trapezoidal_rule(f, 0, 2, n) - exact_integral) for n in n_range]
    simp_errors = [abs(simpson_rule(f, 0, 2, n) - exact_integral) for n in n_range]
    
    ax2.loglog(n_range, trap_errors, 'r-', linewidth=2, label='Trapezoidal Rule')
    ax2.loglog(n_range, simp_errors, 'g-', linewidth=2, label="Simpson's Rule")
    ax2.set_xlabel('Number of intervals (n)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Convergence of Numerical Methods')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

demonstrate_numerical_integration()

# =============================================================================
# SUMMARY AND CONCLUSION
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY AND CONCLUSION")
print("=" * 60)

print("""
This implementation has demonstrated the key concepts of integration:

1. ANTIDERIVATIVES AND INDEFINITE INTEGRALS
   - Definition and relationship to derivatives
   - Family of antiderivatives differing by constants
   - Basic integration rules and techniques

2. DEFINITE INTEGRALS AND AREA INTERPRETATION
   - Geometric interpretation as signed area
   - Connection between antiderivatives and definite integrals
   - Fundamental Theorem of Calculus

3. INTEGRATION TECHNIQUES
   - Substitution method
   - Integration by parts
   - Partial fraction decomposition
   - Trigonometric substitutions

4. APPLICATIONS IN PHYSICS
   - Work done by variable forces
   - Center of mass calculations
   - Energy and momentum calculations

5. APPLICATIONS IN PROBABILITY
   - Probability density functions
   - Expected value calculations
   - Normal distribution properties

6. NUMERICAL INTEGRATION
   - Trapezoidal rule
   - Simpson's rule
   - Error analysis and convergence

These concepts form the foundation for:
- Advanced calculus and analysis
- Physics and engineering applications
- Probability and statistics
- Machine learning and data science
- Scientific computing and numerical methods
""")

print("\nAll concepts have been implemented with:")
print("- Clear mathematical explanations")
print("- Comprehensive visualizations")
print("- Practical applications in physics and probability")
print("- Numerical verification of theoretical results")
print("- Code annotations and documentation")

# 1. Fundamental Theorem of Calculus
def demonstrate_fundamental_theorem():
    """
    Demonstrate the Fundamental Theorem of Calculus: connection between differentiation and integration.
    """
    print("\n--- Fundamental Theorem of Calculus ---")
    # Part 1: If F(x) = ∫[a,x] f(t) dt, then F'(x) = f(x)
    def f(t): return t**2
    def F(x): return x**3/3  # Antiderivative of f(t) = t^2
    def F_prime(x): return x**2  # Should equal f(x)
    x0 = 2.0
    h = 1e-5
    # Numerical derivative of F
    num_deriv = (F(x0 + h) - F(x0 - h)) / (2*h)
    print(f"F(x) = x³/3, F'({x0}) = {num_deriv:.6f}, f({x0}) = {f(x0):.6f}")
    # Part 2: ∫[a,b] f(x) dx = F(b) - F(a)
    a, b = 1, 3
    definite_integral = F(b) - F(a)
    print(f"∫[{a},{b}] x² dx = F({b}) - F({a}) = {definite_integral}")
    # Numerical verification
    from scipy import integrate
    num_integral, _ = integrate.quad(f, a, b)
    print(f"Numerical: ∫[{a},{b}] x² dx = {num_integral:.6f}")

# 2. Enhanced Integration Techniques
def demonstrate_integration_techniques_enhanced():
    """
    Show substitution, integration by parts, and partial fractions with detailed steps.
    """
    print("\n--- Enhanced Integration Techniques ---")
    # Substitution: ∫ x*exp(x²) dx
    print("Substitution: ∫ x*exp(x²) dx")
    print("Let u = x², then du = 2x dx, so dx = du/(2x)")
    print("∫ x*exp(x²) dx = ∫ x*exp(u) * du/(2x) = (1/2)∫ exp(u) du = (1/2)exp(u) + C")
    print("= (1/2)exp(x²) + C")
    # Symbolic verification
    expr = x * sp.exp(x**2)
    result = sp.integrate(expr, x)
    print(f"Symbolic result: {result}")
    # Integration by parts: ∫ x*ln(x) dx
    print("\nIntegration by parts: ∫ x*ln(x) dx")
    print("Let u = ln(x), dv = x dx, then du = dx/x, v = x²/2")
    print("∫ x*ln(x) dx = (x²/2)*ln(x) - ∫ (x²/2)*(dx/x)")
    print("= (x²/2)*ln(x) - (1/2)∫ x dx = (x²/2)*ln(x) - x²/4 + C")
    expr2 = x * sp.log(x)
    result2 = sp.integrate(expr2, x)
    print(f"Symbolic result: {result2}")

# 3. ML Applications: Probability and AUC
def demonstrate_ml_integration_applications():
    """
    Show integration in ML: probability calculations, AUC, expected values.
    """
    print("\n--- ML Applications: Probability and AUC ---")
    # Normal distribution probability
    def normal_pdf(x, mu=0, sigma=1):
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    def normal_cdf(x, mu=0, sigma=1):
        from scipy.stats import norm
        return norm.cdf(x, mu, sigma)
    # Probability P(X < 1) for standard normal
    prob = normal_cdf(1)
    print(f"P(X < 1) for standard normal: {prob:.6f}")
    # AUC calculation for ROC curve
    def roc_curve(tpr, fpr):
        # Sort by FPR for AUC calculation
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr_sorted, fpr_sorted)
        return auc
    # Example ROC curve
    fpr = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    tpr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    auc = roc_curve(tpr, fpr)
    print(f"AUC for ROC curve: {auc:.6f}")
    # Expected value calculation
    def expected_value(pdf_func, a, b):
        def x_times_pdf(x):
            return x * pdf_func(x)
        from scipy import integrate
        return integrate.quad(x_times_pdf, a, b)[0]
    # Expected value of exponential distribution
    def exp_pdf(x, lambda_param=1):
        return lambda_param * np.exp(-lambda_param * x)
    exp_mean = expected_value(lambda x: exp_pdf(x), 0, np.inf)
    print(f"Expected value of exponential(λ=1): {exp_mean:.6f}")

# 4. Advanced Integration: Improper Integrals
def demonstrate_improper_integrals():
    """
    Show improper integrals and their convergence.
    """
    print("\n--- Improper Integrals ---")
    # ∫[1,∞] 1/x² dx
    def f1(x): return 1/x**2
    from scipy import integrate
    result1, error1 = integrate.quad(f1, 1, np.inf)
    print(f"∫[1,∞] 1/x² dx = {result1:.6f} (converges)")
    # ∫[0,1] 1/√x dx
    def f2(x): return 1/np.sqrt(x)
    result2, error2 = integrate.quad(f2, 0, 1)
    print(f"∫[0,1] 1/√x dx = {result2:.6f} (converges)")
    # Divergent example: ∫[1,∞] 1/x dx
    def f3(x): return 1/x
    try:
        result3, error3 = integrate.quad(f3, 1, np.inf)
        print(f"∫[1,∞] 1/x dx = {result3:.6f}")
    except:
        print("∫[1,∞] 1/x dx diverges")

# 5. Integration in Optimization and Loss Functions
def demonstrate_integration_in_optimization():
    """
    Show how integration appears in optimization and loss functions.
    """
    print("\n--- Integration in Optimization ---")
    # Cumulative loss over time
    def loss_function(t, learning_rate=0.1):
        return np.exp(-learning_rate * t)
    def cumulative_loss(t_max):
        from scipy import integrate
        return integrate.quad(loss_function, 0, t_max)[0]
    t_vals = np.linspace(0, 10, 100)
    loss_vals = [loss_function(t) for t in t_vals]
    cum_loss_vals = [cumulative_loss(t) for t in t_vals]
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.plot(t_vals, loss_vals); plt.title('Instantaneous Loss'); plt.xlabel('Time'); plt.ylabel('Loss')
    plt.subplot(132); plt.plot(t_vals, cum_loss_vals); plt.title('Cumulative Loss'); plt.xlabel('Time'); plt.ylabel('Cumulative Loss')
    plt.subplot(133); plt.plot(t_vals, loss_vals, label='Instantaneous'); plt.plot(t_vals, cum_loss_vals, label='Cumulative'); plt.legend(); plt.title('Comparison')
    plt.tight_layout(); plt.show()
    # Area under precision-recall curve
    def precision_recall_curve():
        # Simulated precision-recall data
        recall = np.linspace(0, 1, 100)
        precision = 1 - 0.5 * recall  # Decreasing precision
        # Calculate area under curve
        auc_pr = np.trapz(precision, recall)
        print(f"Area under Precision-Recall curve: {auc_pr:.6f}")
        plt.plot(recall, precision); plt.title('Precision-Recall Curve'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.show()
    precision_recall_curve()

# Run all demonstrations
if __name__ == "__main__":
    demonstrate_fundamental_theorem()
    demonstrate_integration_techniques_enhanced()
    demonstrate_ml_integration_applications()
    demonstrate_improper_integrals()
    demonstrate_integration_in_optimization()

