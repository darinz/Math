# Numerical Methods in Calculus

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Numerical methods provide computational techniques for solving calculus problems when analytical solutions are difficult or impossible to obtain. These methods are essential for practical applications in engineering, physics, machine learning, and data science where exact solutions may not be available or computationally feasible.

**Key Concepts:**
- **Approximation:** Numerical methods provide approximate solutions with controlled error
- **Discretization:** Continuous problems are converted to discrete computational problems
- **Convergence:** Methods improve accuracy as computational effort increases
- **Stability:** Small errors in input don't lead to large errors in output

**Relevance to AI/ML:**
- Numerical integration appears in probability calculations and model evaluation
- Numerical differentiation is used in gradient computation and sensitivity analysis
- Understanding numerical methods helps choose appropriate algorithms and assess accuracy

---

## 10.1 Numerical Integration

### Mathematical Foundations

Numerical integration approximates definite integrals when analytical solutions are unavailable. The goal is to compute:
\[
\int_a^b f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
\]
where \( w_i \) are weights and \( x_i \) are evaluation points.

**Key Properties:**
- **Rectangle Rule:** Uses function values at left endpoints of subintervals
- **Trapezoidal Rule:** Uses linear interpolation between points
- **Simpson's Rule:** Uses quadratic interpolation for higher accuracy
- **Error Analysis:** Error typically decreases as \( O(h^p) \) where \( h \) is step size and \( p \) is order

**Relevance to AI/ML:**
- Computing expectations and probabilities in probabilistic models
- Evaluating model performance metrics (AUC, etc.)
- Numerical optimization and sampling methods

### Python Implementation: Rectangle Rule

The rectangle rule approximates the integral by summing rectangles under the curve. For a partition \( a = x_0 < x_1 < \cdots < x_n = b \):
\[
\int_a^b f(x) \, dx \approx h \sum_{i=0}^{n-1} f(x_i)
\]
where \( h = (b-a)/n \) is the step size.

**Explanation:**
- The rectangle rule approximates the integral by summing the areas of rectangles
- Each rectangle has height equal to the function value at the left endpoint
- The approximation improves as the number of subintervals increases
- The visualization shows how rectangles approximate the area under the curve

---

### Python Implementation: Trapezoidal Rule

The trapezoidal rule uses linear interpolation between points. For a partition \( a = x_0 < x_1 < \cdots < x_n = b \):
\[
\int_a^b f(x) \, dx \approx \frac{h}{2} \left(f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right)
\]
where \( h = (b-a)/n \) is the step size.

**Explanation:**
- The trapezoidal rule uses linear interpolation between adjacent points
- Each trapezoid has area \( \frac{h}{2}(f(x_i) + f(x_{i+1})) \)
- The method is more accurate than the rectangle rule (O(h²) vs O(h))
- The visualization shows how trapezoids approximate the area under the curve

---

### Python Implementation: Simpson's Rule

Simpson's rule uses quadratic interpolation for even higher accuracy. For an even number of subintervals:
\[
\int_a^b f(x) \, dx \approx \frac{h}{3} \left(f(x_0) + 4\sum_{i=1,3,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,\ldots}^{n-2} f(x_i) + f(x_n)\right)
\]

**Explanation:**
- Simpson's rule uses quadratic interpolation for higher accuracy
- The method requires an even number of subintervals
- Error decreases as O(h⁴), making it much more accurate than other methods
- The comparison shows the relative accuracy of different methods

---

## 10.2 Numerical Differentiation

### Mathematical Foundations

Numerical differentiation approximates derivatives when analytical differentiation is difficult. The goal is to compute:
\[
f'(x) \approx \frac{f(x+h) - f(x)}{h}
\]
for some small step size \( h \).

**Key Methods:**
- **Forward Difference:** \( f'(x) \approx \frac{f(x+h) - f(x)}{h} \)
- **Backward Difference:** \( f'(x) \approx \frac{f(x) - f(x-h)}{h} \)
- **Central Difference:** \( f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \)

**Error Analysis:**
- Forward/Backward: Error \( O(h) \)
- Central: Error \( O(h²) \) - more accurate
- Trade-off between accuracy and step size

**Relevance to AI/ML:**
- Computing gradients for optimization algorithms
- Sensitivity analysis of model parameters
- Finite difference methods for complex functions

### Python Implementation: Finite Difference Methods

**Explanation:**
- Forward difference uses the function value at the current point and one step ahead
- Backward difference uses the function value at the current point and one step back
- Central difference uses points on both sides, providing better accuracy
- The error analysis shows how accuracy depends on step size and method choice

---

## 10.3 Root Finding Methods

### Bisection Method

### Newton's Method

## 10.4 Applications in Machine Learning

### Gradient Descent with Numerical Gradients

## Summary

- **Numerical Integration**: Rectangle, trapezoidal, and Simpson's rules for approximating definite integrals
- **Numerical Differentiation**: Forward, backward, and central difference methods for approximating derivatives
- **Root Finding**: Bisection and Newton's methods for finding zeros of functions
- **Applications**: Gradient descent with numerical gradients when analytical derivatives are unavailable
- **Error Analysis**: Understanding convergence and accuracy of numerical methods

## Next Steps

Numerical methods provide essential tools for solving calculus problems computationally. These techniques are fundamental to scientific computing, machine learning algorithms, and engineering applications where analytical solutions are not feasible. 