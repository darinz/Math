# Numerical Methods in Calculus

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Numerical methods provide computational techniques for solving calculus problems when analytical solutions are difficult or impossible to obtain. These methods are essential for practical applications in engineering, physics, machine learning, and data science where exact solutions may not be available or computationally feasible.

### Why Numerical Methods Matter

Numerical methods bridge the gap between theoretical calculus and practical computation:

1. **Analytical Intractability**: Many integrals and derivatives cannot be expressed in closed form
2. **Computational Efficiency**: Numerical methods can be faster than symbolic computation
3. **Real-World Data**: Experimental data often requires numerical approximation
4. **High-Dimensional Problems**: Analytical solutions become impractical in many dimensions
5. **Error Control**: Numerical methods provide controlled approximation with known error bounds
6. **Adaptive Methods**: Algorithms can adjust precision based on problem difficulty

### Mathematical Foundation

Numerical methods approximate continuous mathematical operations using discrete computational procedures. The fundamental challenge is to balance:

- **Accuracy**: How close the approximation is to the true value
- **Efficiency**: Computational cost and time requirements
- **Stability**: Sensitivity to input errors and roundoff
- **Convergence**: Rate at which accuracy improves with computational effort

### Key Concepts

- **Approximation**: Numerical methods provide approximate solutions with controlled error
- **Discretization**: Continuous problems are converted to discrete computational problems
- **Convergence**: Methods improve accuracy as computational effort increases
- **Stability**: Small errors in input don't lead to large errors in output
- **Error Analysis**: Understanding and bounding approximation errors
- **Adaptive Methods**: Algorithms that adjust precision based on local behavior

### Relevance to AI/ML

Numerical methods are fundamental to modern machine learning:

- **Gradient Computation**: Numerical differentiation when analytical gradients are unavailable
- **Integration**: Computing expectations, probabilities, and model evaluation metrics
- **Optimization**: Numerical algorithms for training neural networks
- **Sampling**: Monte Carlo methods for probabilistic inference
- **Interpolation**: Kernel methods and function approximation
- **Root Finding**: Solving optimization problems and finding equilibria

## 10.1 Numerical Integration

### Mathematical Foundations

Numerical integration approximates definite integrals when analytical solutions are unavailable. The goal is to compute:

```math
\int_a^b f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
```

where $`w_i`$ are weights and $`x_i`$ are evaluation points.

#### Error Analysis

The error in numerical integration typically follows:
```math
\text{Error} = \int_a^b f(x) \, dx - \sum_{i=1}^n w_i f(x_i) = O(h^p)
```

where $`h`$ is the step size and $`p`$ is the order of the method.

#### Key Properties

1. **Rectangle Rule**: Uses function values at endpoints of subintervals
2. **Trapezoidal Rule**: Uses linear interpolation between points
3. **Simpson's Rule**: Uses quadratic interpolation for higher accuracy
4. **Adaptive Methods**: Adjust step size based on local error estimates
5. **Gaussian Quadrature**: Optimal point placement for maximum accuracy

#### Relevance to AI/ML

- Computing expectations and probabilities in probabilistic models
- Evaluating model performance metrics (AUC, etc.)
- Numerical optimization and sampling methods
- Kernel density estimation and smoothing

### Rectangle Rule

#### Mathematical Formulation

For a partition $`a = x_0 < x_1 < \cdots < x_n = b`$:

```math
\int_a^b f(x) \, dx \approx h \sum_{i=0}^{n-1} f(x_i)
```

where $`h = (b-a)/n`$ is the step size.

#### Error Analysis

The error for the rectangle rule is:
```math
\text{Error} = \frac{(b-a)h}{2} f'(\xi)
```

where $`\xi \in [a, b]`$ is some point in the interval.

#### Properties

- **Order**: $`O(h)`$ - first-order accurate
- **Simplicity**: Easy to implement and understand
- **Bias**: Tends to underestimate or overestimate depending on function behavior
- **Efficiency**: Requires only function evaluations, no derivatives

### Python Implementation: Rectangle Rule

The rectangle rule approximates the integral by summing rectangles under the curve. For a partition $`a = x_0 < x_1 < \cdots < x_n = b`$:

```math
\int_a^b f(x) \, dx \approx h \sum_{i=0}^{n-1} f(x_i)
```

where $`h = (b-a)/n`$ is the step size.

**Explanation:**
- The rectangle rule approximates the integral by summing the areas of rectangles
- Each rectangle has height equal to the function value at the left endpoint
- The approximation improves as the number of subintervals increases
- The visualization shows how rectangles approximate the area under the curve

### Trapezoidal Rule

#### Mathematical Formulation

The trapezoidal rule uses linear interpolation between points:

```math
\int_a^b f(x) \, dx \approx \frac{h}{2} \left(f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right)
```

where $`h = (b-a)/n`$ is the step size.

#### Error Analysis

The error for the trapezoidal rule is:
```math
\text{Error} = -\frac{(b-a)h^2}{12} f''(\xi)
```

where $`\xi \in [a, b]`$ is some point in the interval.

#### Properties

- **Order**: $`O(h^2)`$ - second-order accurate
- **Accuracy**: More accurate than rectangle rule
- **Symmetry**: Treats endpoints equally
- **Interpolation**: Uses linear interpolation between points

### Python Implementation: Trapezoidal Rule

The trapezoidal rule uses linear interpolation between points. For a partition $`a = x_0 < x_1 < \cdots < x_n = b`$:

```math
\int_a^b f(x) \, dx \approx \frac{h}{2} \left(f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right)
```

where $`h = (b-a)/n`$ is the step size.

**Explanation:**
- The trapezoidal rule uses linear interpolation between adjacent points
- Each trapezoid has area $`\frac{h}{2}(f(x_i) + f(x_{i+1}))`$
- The method is more accurate than the rectangle rule (O(h²) vs O(h))
- The visualization shows how trapezoids approximate the area under the curve

### Simpson's Rule

#### Mathematical Formulation

Simpson's rule uses quadratic interpolation for even higher accuracy:

```math
\int_a^b f(x) \, dx \approx \frac{h}{3} \left(f(x_0) + 4\sum_{i=1,3,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,\ldots}^{n-2} f(x_i) + f(x_n)\right)
```

#### Error Analysis

The error for Simpson's rule is:
```math
\text{Error} = -\frac{(b-a)h^4}{180} f^{(4)}(\xi)
```

where $`\xi \in [a, b]`$ is some point in the interval.

#### Properties

- **Order**: $`O(h^4)`$ - fourth-order accurate
- **Requirement**: Needs even number of subintervals
- **Accuracy**: Much more accurate than trapezoidal rule
- **Interpolation**: Uses quadratic interpolation

### Python Implementation: Simpson's Rule

Simpson's rule uses quadratic interpolation for even higher accuracy. For an even number of subintervals:

```math
\int_a^b f(x) \, dx \approx \frac{h}{3} \left(f(x_0) + 4\sum_{i=1,3,\ldots}^{n-1} f(x_i) + 2\sum_{i=2,4,\ldots}^{n-2} f(x_i) + f(x_n)\right)
```

**Explanation:**
- Simpson's rule uses quadratic interpolation for higher accuracy
- The method requires an even number of subintervals
- Error decreases as O(h⁴), making it much more accurate than other methods
- The comparison shows the relative accuracy of different methods

### Gaussian Quadrature

#### Mathematical Formulation

Gaussian quadrature uses optimal point placement:

```math
\int_a^b f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
```

where $`x_i`$ are the roots of orthogonal polynomials and $`w_i`$ are corresponding weights.

#### Properties

- **Optimal Accuracy**: Maximum accuracy for given number of points
- **Non-Uniform Points**: Evaluation points are not equally spaced
- **High Order**: Can achieve very high accuracy with few points
- **Specialized**: Different rules for different weight functions

#### Example: Gauss-Legendre Quadrature

For the interval $`[-1, 1]`$:
```math
\int_{-1}^1 f(x) \, dx \approx \sum_{i=1}^n w_i f(x_i)
```

where $`x_i`$ are roots of Legendre polynomials.

### Adaptive Integration

#### Mathematical Formulation

Adaptive methods estimate local error and adjust step size:

```math
\text{Error Estimate} = |I_1 - I_2|
```

where $`I_1`$ and $`I_2`$ are estimates using different methods or step sizes.

#### Algorithm

1. **Estimate**: Compute integral with current step size
2. **Refine**: Compute integral with smaller step size
3. **Compare**: Estimate error from difference
4. **Adjust**: Reduce step size if error is too large
5. **Repeat**: Continue until desired accuracy is achieved

#### Benefits

- **Efficiency**: Concentrates effort where needed
- **Accuracy**: Achieves desired precision with minimal work
- **Robustness**: Handles functions with varying smoothness

## 10.2 Numerical Differentiation

### Mathematical Foundations

Numerical differentiation approximates derivatives when analytical differentiation is difficult. The goal is to compute:

```math
f'(x) \approx \frac{f(x+h) - f(x)}{h}
```

for some small step size $`h`$.

#### Taylor Series Analysis

Using Taylor series expansion:
```math
f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \cdots
```

Rearranging gives:
```math
f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) - \cdots
```

#### Error Analysis

The truncation error is:
```math
\text{Error} = -\frac{h}{2}f''(\xi)
```

where $`\xi \in [x, x+h]`$.

### Forward Difference

#### Mathematical Formulation

```math
f'(x) \approx \frac{f(x+h) - f(x)}{h}
```

#### Error Analysis

- **Order**: $`O(h)`$ - first-order accurate
- **Bias**: Systematic error due to truncation
- **Stability**: Sensitive to roundoff errors for small $`h`$

#### Optimal Step Size

The optimal step size balances truncation and roundoff errors:
```math
h_{\text{opt}} = \sqrt{\frac{2\epsilon}{f''(x)}}
```

where $`\epsilon`$ is machine precision.

### Backward Difference

#### Mathematical Formulation

```math
f'(x) \approx \frac{f(x) - f(x-h)}{h}
```

#### Error Analysis

- **Order**: $`O(h)`$ - first-order accurate
- **Bias**: Opposite systematic error from forward difference
- **Stability**: Similar roundoff sensitivity

### Central Difference

#### Mathematical Formulation

```math
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
```

#### Error Analysis

Using Taylor series for both $`f(x+h)`$ and $`f(x-h)`$:
```math
f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \cdots
```

```math
f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + \cdots
```

Subtracting and solving for $`f'(x)`$:
```math
f'(x) = \frac{f(x+h) - f(x-h)}{2h} - \frac{h^2}{6}f'''(x) - \cdots
```

#### Properties

- **Order**: $`O(h^2)`$ - second-order accurate
- **Accuracy**: Much more accurate than forward/backward differences
- **Symmetry**: Cancels leading error terms
- **Optimal Step Size**: $`h_{\text{opt}} = \sqrt[3]{\frac{3\epsilon}{f'''(x)}}`$

### Higher-Order Methods

#### Second-Order Central Difference

For second derivatives:
```math
f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}
```

#### Fourth-Order Central Difference

For first derivatives:
```math
f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}
```

### Python Implementation: Finite Difference Methods

**Explanation:**
- Forward difference uses the function value at the current point and one step ahead
- Backward difference uses the function value at the current point and one step back
- Central difference uses points on both sides, providing better accuracy
- The error analysis shows how accuracy depends on step size and method choice

## 10.3 Root Finding Methods

### Mathematical Foundations

Root finding methods solve equations of the form:
```math
f(x) = 0
```

where $`f`$ is a continuous function.

#### Key Properties

1. **Existence**: Intermediate Value Theorem guarantees roots for continuous functions
2. **Uniqueness**: Not guaranteed without additional conditions
3. **Convergence**: Methods typically converge to a root under appropriate conditions
4. **Order**: Rate at which error decreases with iterations

### Bisection Method

#### Mathematical Formulation

The bisection method finds a root in the interval $`[a, b]`$ where $`f(a)f(b) < 0`$:

```math
c = \frac{a + b}{2}
```

If $`f(c) = 0`$, we're done. Otherwise:
- If $`f(a)f(c) < 0`$, root is in $`[a, c]`$
- If $`f(c)f(b) < 0``, root is in $`[c, b]`$

#### Error Analysis

After $`n`$ iterations, the error is bounded by:
```math
|x_n - x^*| \leq \frac{b-a}{2^n}
```

where $`x^*`$ is the true root.

#### Properties

- **Guaranteed Convergence**: Always converges if initial conditions are met
- **Linear Convergence**: Error decreases by factor of 2 each iteration
- **Robustness**: Very stable and reliable
- **Efficiency**: Simple but may be slow for high precision

### Newton's Method

#### Mathematical Formulation

Newton's method uses linear approximation:
```math
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
```

#### Derivation

Using Taylor series around $`x_n`$:
```math
f(x) \approx f(x_n) + f'(x_n)(x - x_n)
```

Setting $`f(x) = 0`$ and solving for $`x`$:
```math
x = x_n - \frac{f(x_n)}{f'(x_n)}
```

#### Error Analysis

For a simple root $`x^*`$:
```math
|x_{n+1} - x^*| \leq \frac{M}{2m} |x_n - x^*|^2
```

where $`M = \max |f''(x)|`$ and $`m = \min |f'(x)|`$ in a neighborhood of $`x^*`$.

#### Properties

- **Quadratic Convergence**: Very fast when it converges
- **Local Convergence**: Only guaranteed near the root
- **Requires Derivative**: Needs $`f'(x)`$ at each iteration
- **Sensitivity**: Can fail if $`f'(x_n) \approx 0`$

### Secant Method

#### Mathematical Formulation

The secant method approximates the derivative:
```math
x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}
```

#### Properties

- **Superlinear Convergence**: Faster than linear but slower than quadratic
- **No Derivatives**: Only requires function evaluations
- **Two Starting Points**: Needs two initial guesses
- **Robustness**: More robust than Newton's method

### Fixed Point Iteration

#### Mathematical Formulation

Rewrite $`f(x) = 0`$ as $`x = g(x)`$:
```math
x_{n+1} = g(x_n)
```

#### Convergence

If $`|g'(x)| < 1`$ in a neighborhood of the fixed point, then the method converges.

#### Error Analysis

```math
|x_{n+1} - x^*| \leq |g'(x^*)| |x_n - x^*|
```

### Comparison of Methods

| Method | Convergence | Derivatives | Robustness | Efficiency |
|--------|-------------|-------------|------------|------------|
| Bisection | Linear | No | High | Low |
| Newton | Quadratic | Yes | Low | High |
| Secant | Superlinear | No | Medium | Medium |
| Fixed Point | Linear | No | Medium | High |

## 10.4 Applications in Machine Learning

### Gradient Descent with Numerical Gradients

#### Motivation

When analytical gradients are unavailable or expensive to compute, numerical differentiation provides an alternative.

#### Implementation

Using central difference:
```math
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}
```

where $`e_i`$ is the unit vector in direction $`i`$.

#### Challenges

1. **Computational Cost**: Requires $`2d`$ function evaluations for $`d`$ parameters
2. **Accuracy**: Sensitive to step size choice
3. **Noise**: Numerical errors can affect convergence
4. **Scaling**: Cost grows linearly with parameter dimension

### Numerical Integration in ML

#### Expectation Computation

For probabilistic models:
```math
\mathbb{E}[f(x)] = \int f(x) p(x) \, dx
```

#### Monte Carlo Integration

```math
\mathbb{E}[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i)
```

where $`x_i \sim p(x)`$.

#### Properties

- **Convergence**: $`O(1/\sqrt{N})`$ regardless of dimension
- **Robustness**: Works for high-dimensional problems
- **Variance**: Error depends on variance of $`f(x)`$

### Numerical Optimization

#### Line Search

Finding optimal step size:
```math
\alpha^* = \arg\min_{\alpha} f(x + \alpha d)
```

#### Trust Region Methods

Solving subproblems:
```math
\min_d \frac{1}{2} d^T H d + g^T d
```

subject to $`\|d\| \leq \Delta`$.

### Sensitivity Analysis

#### Parameter Sensitivity

Computing how outputs change with inputs:
```math
\frac{\partial y}{\partial x_i} \approx \frac{y(x + h e_i) - y(x)}{h}
```

#### Gradient-Based Feature Importance

```math
\text{Importance}_i = \left|\frac{\partial f}{\partial x_i}\right|
```

## 10.5 Advanced Numerical Methods

### Adaptive Methods

#### Adaptive Integration

```math
\text{Error Estimate} = |I_1 - I_2|
```

where $`I_1`$ and $`I_2`$ are estimates using different methods.

#### Adaptive Differentiation

```math
h_{\text{opt}} = \sqrt{\frac{2\epsilon}{|f''(x)|}}
```

### Multidimensional Methods

#### Numerical Integration

For $`f: \mathbb{R}^n \to \mathbb{R}`$:
```math
\int_D f(x) \, dx \approx \sum_{i=1}^N w_i f(x_i)
```

#### Numerical Differentiation

For gradients:
```math
\nabla f(x) \approx \left(\frac{f(x + h e_1) - f(x - h e_1)}{2h}, \ldots, \frac{f(x + h e_n) - f(x - h e_n)}{2h}\right)
```

### Error Analysis and Control

#### Error Propagation

For composite functions:
```math
\text{Error}(f \circ g) \approx |f'(g(x))| \cdot \text{Error}(g) + \text{Error}(f)
```

#### Condition Number

```math
\kappa = \frac{\text{Relative Error in Output}}{\text{Relative Error in Input}}
```

## Summary

Numerical methods provide essential computational tools for calculus:

1. **Numerical Integration**: Rectangle, trapezoidal, Simpson's, and Gaussian quadrature methods
2. **Numerical Differentiation**: Forward, backward, central, and higher-order difference methods
3. **Root Finding**: Bisection, Newton's, secant, and fixed point iteration methods
4. **Applications**: Gradient computation, expectation calculation, and sensitivity analysis in ML
5. **Error Analysis**: Understanding convergence rates and accuracy bounds
6. **Adaptive Methods**: Algorithms that adjust precision based on local behavior

### Key Takeaways

- **Numerical Integration**: Approximates definite integrals using discrete sums
- **Numerical Differentiation**: Approximates derivatives using finite differences
- **Root Finding**: Solves equations using iterative methods
- **Error Analysis**: Essential for understanding accuracy and convergence
- **Adaptive Methods**: Improve efficiency by adjusting computational effort
- **Applications**: Fundamental to scientific computing and machine learning

### Next Steps

With a solid understanding of numerical methods, you're ready to explore:
- **Advanced Optimization**: Interior point methods, sequential quadratic programming
- **Partial Differential Equations**: Finite difference, finite element methods
- **Stochastic Methods**: Monte Carlo, Markov chain Monte Carlo
- **High-Performance Computing**: Parallel algorithms and GPU acceleration 