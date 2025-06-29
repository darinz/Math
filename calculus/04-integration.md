# Integration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Integration is the reverse process of differentiation and is essential for calculating areas, volumes, and cumulative effects. In machine learning, integration is used for probability calculations, expected values, and continuous optimization.

## 4.1 Antiderivatives and Indefinite Integrals

The indefinite integral finds the antiderivative of a function.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate

# Symbolic integration using SymPy
x = sp.Symbol('x')

# Basic antiderivatives
def demonstrate_antiderivatives():
    # ∫ x² dx = x³/3 + C
    integral1 = sp.integrate(x**2, x)
    print(f"∫ x² dx = {integral1}")
    
    # ∫ sin(x) dx = -cos(x) + C
    integral2 = sp.integrate(sp.sin(x), x)
    print(f"∫ sin(x) dx = {integral2}")
    
    # ∫ e^x dx = e^x + C
    integral3 = sp.integrate(sp.exp(x), x)
    print(f"∫ e^x dx = {integral3}")
    
    # ∫ 1/x dx = ln|x| + C
    integral4 = sp.integrate(1/x, x)
    print(f"∫ 1/x dx = {integral4}")

demonstrate_antiderivatives()

# Visualize antiderivatives
def f(x):
    return x**2

def F(x):
    return x**3 / 3

x_vals = np.linspace(-2, 2, 100)
y_vals = f(x_vals)
Y_vals = F(x_vals)

plt.figure(figsize=(12, 5))

# Original function
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True)

# Antiderivative
plt.subplot(1, 2, 2)
plt.plot(x_vals, Y_vals, 'r-', linewidth=2, label='F(x) = x³/3')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Antiderivative')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 4.2 Definite Integrals

Definite integrals calculate the area under a curve between two points.

```python
# Definite integration
def definite_integration_examples():
    # ∫₀¹ x² dx = [x³/3]₀¹ = 1/3
    integral1 = sp.integrate(x**2, (x, 0, 1))
    print(f"∫₀¹ x² dx = {integral1}")
    
    # ∫₀^π sin(x) dx = [-cos(x)]₀^π = 2
    integral2 = sp.integrate(sp.sin(x), (x, 0, sp.pi))
    print(f"∫₀^π sin(x) dx = {integral2}")
    
    # ∫₀^∞ e^(-x) dx = [-e^(-x)]₀^∞ = 1
    integral3 = sp.integrate(sp.exp(-x), (x, 0, sp.oo))
    print(f"∫₀^∞ e^(-x) dx = {integral3}")

definite_integration_examples()

# Numerical integration using SciPy
def numerical_integration():
    # ∫₀¹ x² dx
    result1, error1 = integrate.quad(lambda x: x**2, 0, 1)
    print(f"Numerical ∫₀¹ x² dx = {result1:.6f} (error: {error1:.2e})")
    
    # ∫₀^π sin(x) dx
    result2, error2 = integrate.quad(lambda x: np.sin(x), 0, np.pi)
    print(f"Numerical ∫₀^π sin(x) dx = {result2:.6f} (error: {error2:.2e})")
    
    # ∫₀^∞ e^(-x) dx
    result3, error3 = integrate.quad(lambda x: np.exp(-x), 0, np.inf)
    print(f"Numerical ∫₀^∞ e^(-x) dx = {result3:.6f} (error: {error3:.2e})")

numerical_integration()

# Visualize definite integrals
def visualize_definite_integral():
    x_vals = np.linspace(0, 1, 1000)
    y_vals = x_vals**2
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue', label='Area under curve')
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Definite Integral: ∫₀¹ x² dx = 1/3')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_definite_integral()
```

## 4.3 Integration Techniques

### Substitution Method

```python
# Integration by substitution
def substitution_examples():
    # ∫ x*e^(x²) dx using u = x²
    # du = 2x dx, so dx = du/(2x)
    # ∫ x*e^(x²) dx = ∫ x*e^u * du/(2x) = (1/2)∫ e^u du = (1/2)e^u + C = (1/2)e^(x²) + C
    
    # Using SymPy
    integral = sp.integrate(x * sp.exp(x**2), x)
    print(f"∫ x*e^(x²) dx = {integral}")
    
    # ∫ sin(2x) dx using u = 2x
    integral2 = sp.integrate(sp.sin(2*x), x)
    print(f"∫ sin(2x) dx = {integral2}")

substitution_examples()
```

### Integration by Parts

```python
# Integration by parts: ∫ u dv = uv - ∫ v du
def integration_by_parts_examples():
    # ∫ x*e^x dx
    # Let u = x, dv = e^x dx
    # Then du = dx, v = e^x
    # ∫ x*e^x dx = x*e^x - ∫ e^x dx = x*e^x - e^x + C
    
    integral = sp.integrate(x * sp.exp(x), x)
    print(f"∫ x*e^x dx = {integral}")
    
    # ∫ x*ln(x) dx
    integral2 = sp.integrate(x * sp.log(x), x)
    print(f"∫ x*ln(x) dx = {integral2}")

integration_by_parts_examples()
```

## 4.4 Applications in Probability and Statistics

### Probability Density Functions

```python
# Normal distribution integration
def normal_distribution_integration():
    # Standard normal distribution: f(x) = (1/√(2π)) * e^(-x²/2)
    def normal_pdf(x):
        return (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    
    # ∫₋∞^∞ f(x) dx = 1 (total probability)
    total_prob, error = integrate.quad(normal_pdf, -np.inf, np.inf)
    print(f"Total probability: {total_prob:.6f}")
    
    # P(-1 ≤ X ≤ 1) = ∫₋₁¹ f(x) dx
    prob_1sigma, error = integrate.quad(normal_pdf, -1, 1)
    print(f"P(-1 ≤ X ≤ 1): {prob_1sigma:.6f}")
    
    # P(-2 ≤ X ≤ 2) = ∫₋₂² f(x) dx
    prob_2sigma, error = integrate.quad(normal_pdf, -2, 2)
    print(f"P(-2 ≤ X ≤ 2): {prob_2sigma:.6f}")
    
    # Visualize
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = normal_pdf(x_vals)
    
    plt.figure(figsize=(12, 5))
    
    # Full distribution
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Standard Normal Distribution')
    plt.grid(True)
    
    # ±1σ region
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    mask = (x_vals >= -1) & (x_vals <= 1)
    plt.fill_between(x_vals[mask], y_vals[mask], alpha=0.5, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('P(-1 ≤ X ≤ 1)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

normal_distribution_integration()
```

### Expected Values

```python
# Expected value calculations
def expected_value_examples():
    # E[X] = ∫ x*f(x) dx for continuous random variables
    
    # Exponential distribution: f(x) = λ*e^(-λx) for x ≥ 0
    def exponential_pdf(x, lambda_param=1):
        return lambda_param * np.exp(-lambda_param * x)
    
    # E[X] = ∫₀^∞ x*λ*e^(-λx) dx = 1/λ
    def expected_value_exponential(lambda_param=1):
        integrand = lambda x: x * exponential_pdf(x, lambda_param)
        result, error = integrate.quad(integrand, 0, np.inf)
        return result
    
    expected_val = expected_value_exponential(1)
    print(f"E[X] for exponential(λ=1): {expected_val:.6f}")
    print(f"Theoretical value: {1/1:.6f}")
    
    # Variance: Var(X) = E[X²] - (E[X])²
    def variance_exponential(lambda_param=1):
        integrand = lambda x: x**2 * exponential_pdf(x, lambda_param)
        e_x_squared, _ = integrate.quad(integrand, 0, np.inf)
        e_x = expected_value_exponential(lambda_param)
        return e_x_squared - e_x**2
    
    variance = variance_exponential(1)
    print(f"Var(X) for exponential(λ=1): {variance:.6f}")
    print(f"Theoretical value: {1/1**2:.6f}")

expected_value_examples()
```

## 4.5 Numerical Integration Methods

### Trapezoidal Rule

```python
# Trapezoidal rule implementation
def trapezoidal_rule(f, a, b, n=1000):
    """Approximate ∫ₐᵇ f(x) dx using trapezoidal rule"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

# Compare different methods
def compare_integration_methods():
    f = lambda x: x**2
    
    # Exact value
    exact = 1/3
    
    # Trapezoidal rule
    trapezoidal = trapezoidal_rule(f, 0, 1, 100)
    
    # SciPy quad
    scipy_result, _ = integrate.quad(f, 0, 1)
    
    print(f"Exact value: {exact:.6f}")
    print(f"Trapezoidal rule: {trapezoidal:.6f}")
    print(f"SciPy quad: {scipy_result:.6f}")
    
    # Error analysis
    trapezoidal_error = abs(trapezoidal - exact)
    scipy_error = abs(scipy_result - exact)
    
    print(f"Trapezoidal error: {trapezoidal_error:.2e}")
    print(f"SciPy error: {scipy_error:.2e}")

compare_integration_methods()
```

### Monte Carlo Integration

```python
# Monte Carlo integration
def monte_carlo_integration(f, a, b, n=10000):
    """Approximate ∫ₐᵇ f(x) dx using Monte Carlo method"""
    x_random = np.random.uniform(a, b, n)
    y_random = f(x_random)
    return (b - a) * np.mean(y_random)

# Example: ∫₀¹ x² dx
def monte_carlo_example():
    f = lambda x: x**2
    exact = 1/3
    
    # Run multiple times to see variation
    results = []
    for i in range(10):
        result = monte_carlo_integration(f, 0, 1, 10000)
        results.append(result)
    
    print(f"Exact value: {exact:.6f}")
    print(f"Monte Carlo results: {[f'{r:.6f}' for r in results]}")
    print(f"Mean: {np.mean(results):.6f}")
    print(f"Std: {np.std(results):.6f}")

monte_carlo_example()
```

## 4.6 Applications in Machine Learning

### Loss Function Integration

```python
# Integration in loss functions
def loss_function_integration():
    # Expected loss over a distribution
    def expected_loss(loss_func, data_distribution, n_samples=10000):
        """Calculate expected loss over a data distribution"""
        samples = data_distribution.rvs(n_samples)
        losses = [loss_func(x) for x in samples]
        return np.mean(losses)
    
    # Example: Expected MSE loss
    from scipy.stats import norm
    
    def mse_loss(x, target=0):
        return (x - target)**2
    
    # Expected MSE when data follows N(0, 1)
    expected_mse = expected_loss(mse_loss, norm(0, 1))
    print(f"Expected MSE for N(0,1): {expected_mse:.6f}")
    
    # Theoretical: E[(X-0)²] = Var(X) = 1
    print(f"Theoretical value: {1:.6f}")

loss_function_integration()
```

### Area Under ROC Curve (AUC)

```python
# AUC calculation using integration
def calculate_auc():
    # Simulate ROC curve data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate scores for two classes
    scores_class_0 = np.random.normal(0, 1, n_samples//2)
    scores_class_1 = np.random.normal(1, 1, n_samples//2)
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    y_scores = np.concatenate([scores_class_0, scores_class_1])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    print(f"AUC: {auc_score:.6f}")
    
    # Visualize ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

calculate_auc()
```

## Summary

- **Antiderivatives** reverse the process of differentiation
- **Definite integrals** calculate areas and cumulative effects
- **Integration techniques** include substitution and integration by parts
- **Numerical methods** provide approximations when symbolic integration is difficult
- **Applications** include probability calculations, expected values, and performance metrics
- **Machine learning** uses integration for AUC, expected losses, and probability distributions

## Next Steps

Integration provides the foundation for understanding areas, volumes, and cumulative effects. In the next section, we'll explore applications of integration in optimization and advanced calculus concepts. 