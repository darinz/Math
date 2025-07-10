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

**Explanation:**
- The code computes partial derivatives for several functions, illustrating how each variable affects the output.
- 3D plots show the original function and the effect of changing each variable independently.
- These concepts are foundational for gradient-based optimization and feature sensitivity in AI/ML.

## 6.3 Gradient and Directional Derivatives

### Gradient Vector

### Directional Derivatives

## 6.4 Optimization in Multiple Dimensions

### Critical Points and Classification

## 6.5 Lagrange Multipliers

### Constrained Optimization

## 6.6 Applications in Machine Learning

### Gradient Descent in Multiple Dimensions

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