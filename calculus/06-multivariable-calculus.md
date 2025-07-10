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

### Why Multivariable Calculus Matters in AI/ML

Multivariable calculus is the mathematical foundation for understanding and implementing machine learning algorithms:

1. **High-Dimensional Optimization**: Neural networks have millions of parameters, requiring optimization in high-dimensional spaces
2. **Gradient-Based Learning**: Understanding gradients, Hessians, and directional derivatives is crucial for training algorithms
3. **Feature Interactions**: Multivariable functions model complex interactions between features
4. **Loss Landscapes**: Understanding the geometry of loss functions helps with optimization and convergence
5. **Sensitivity Analysis**: Partial derivatives quantify how changes in inputs affect outputs
6. **Constrained Optimization**: Lagrange multipliers handle constraints in optimization problems
7. **Vector Fields**: Understanding vector fields is essential for gradient flows and optimization paths
8. **Dimensionality Reduction**: Understanding high-dimensional geometry aids in feature selection and dimensionality reduction

### Mathematical Foundation

Multivariable calculus extends single-variable concepts to functions $`f: \mathbb{R}^n \to \mathbb{R}`$:

- **Functions of Multiple Variables**: $`f(x_1, x_2, \ldots, x_n)`$
- **Partial Derivatives**: $`\frac{\partial f}{\partial x_i}`$ measures change in the $`i`$-th direction
- **Gradient**: $`\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)`$
- **Hessian Matrix**: $`H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}`$

### Intuitive Understanding

Think of multivariable functions as:
- **Surfaces**: Functions of two variables create surfaces in 3D space
- **Landscapes**: High-dimensional functions create complex "landscapes" in many dimensions
- **Contours**: Level sets help visualize the shape of these landscapes
- **Gradients**: Vectors pointing in the direction of steepest ascent

## 6.1 Functions of Multiple Variables

### Mathematical Foundations and Visualization

A function of two variables, $`f(x, y)`$, assigns a real number to each point $`(x, y)`$ in its domain. The graph of such a function is a surface in three-dimensional space.

#### Domain and Range

- **Domain**: The set of all possible input points $`(x, y)`$ where the function is defined
- **Range**: The set of all possible output values $`f(x, y)`$
- **Level Sets**: Curves where $`f(x, y) = c`$ for constant $`c`$

#### Key Concepts

**Level Curves (Contours)**:
Curves where $`f(x, y) = c`$ for constant $`c`$. These help visualize the function's behavior in the plane.

**Surfaces**:
The set of points $`(x, y, f(x, y))`$ forms a surface, which can be visualized in 3D.

**Continuity**:
A function $`f(x, y)`$ is continuous at $`(a, b)`$ if:
```math
\lim_{(x,y) \to (a,b)} f(x, y) = f(a, b)
```

#### Common Multivariable Functions

**Linear Functions**:
```math
f(x, y) = ax + by + c
```

**Quadratic Functions**:
```math
f(x, y) = ax^2 + by^2 + cxy + dx + ey + f
```

**Exponential Functions**:
```math
f(x, y) = e^{ax + by}
```

**Trigonometric Functions**:
```math
f(x, y) = \sin(x) \cos(y)
```

### Applications in Machine Learning

**Loss Functions**:
Neural network loss functions are typically multivariable functions of the form:
```math
L(\theta_1, \theta_2, \ldots, \theta_n) = \frac{1}{m} \sum_{i=1}^m \ell(y_i, \hat{y}_i)
```

where $`\theta_i`$ are the model parameters.

**Feature Interactions**:
Multivariable functions model complex interactions between features:
```math
f(x_1, x_2) = x_1^2 + x_2^2 + x_1 x_2
```

**Activation Functions**:
Many activation functions can be extended to multiple variables:
```math
\text{ReLU}(x_1, x_2) = \max(0, x_1 + x_2)
```

### Python Implementation: Multivariable Functions

The following code demonstrates how to define and visualize several multivariable functions, with commentary on their geometric and practical significance.

**Explanation:**
- The code defines and visualizes three types of surfaces: convex (paraboloid), oscillatory, and saddle.
- 3D plots show the geometry of each function, while contour plots reveal level curves and critical points.
- These visualizations are directly relevant to understanding optimization landscapes and feature interactions in AI/ML.

## 6.2 Partial Derivatives

### Mathematical Foundations and Computation

A partial derivative measures how a multivariable function changes as one variable varies, holding the others constant.

#### Definition

For a function $`f(x, y)`$, the partial derivatives are:

```math
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
```

```math
\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y + h) - f(x, y)}{h}
```

#### Geometric Interpretation

- **$`\frac{\partial f}{\partial x}`$**: Slope of the tangent line in the x-direction
- **$`\frac{\partial f}{\partial y}`$**: Slope of the tangent line in the y-direction
- **Cross-sections**: Partial derivatives represent slopes of cross-sections of the surface

#### Higher-Order Partial Derivatives

**Second-Order Partial Derivatives**:
```math
\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)
```

```math
\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right)
```

**Mixed Partial Derivatives**:
```math
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)
```

```math
\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)
```

**Clairaut's Theorem**: If $`f`$ has continuous second-order partial derivatives, then:
```math
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}
```

#### Examples

**Example 1**: $`f(x, y) = x^2 + y^2`$
```math
\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 2y
```

**Example 2**: $`f(x, y) = e^{xy}`$
```math
\frac{\partial f}{\partial x} = ye^{xy}, \quad \frac{\partial f}{\partial y} = xe^{xy}
```

**Example 3**: $`f(x, y) = \sin(xy)`$
```math
\frac{\partial f}{\partial x} = y\cos(xy), \quad \frac{\partial f}{\partial y} = x\cos(xy)
```

### Applications in Machine Learning

**Gradient Computation**:
Partial derivatives are the components of the gradient vector:
```math
\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)
```

**Sensitivity Analysis**:
Partial derivatives quantify how sensitive a model's output is to each input feature:
```math
\text{Sensitivity}_i = \left|\frac{\partial f}{\partial x_i}\right|
```

**Feature Importance**:
The magnitude of partial derivatives can indicate feature importance:
```math
\text{Importance}_i = \frac{1}{n} \sum_{j=1}^n \left|\frac{\partial f}{\partial x_i}\right|_{x=x_j}
```

### Python Implementation: Partial Derivatives

**Explanation:**
- The code computes partial derivatives for several functions, illustrating how each variable affects the output.
- 3D plots show the original function and the effect of changing each variable independently.
- These concepts are foundational for gradient-based optimization and feature sensitivity in AI/ML.

## 6.3 Gradient and Directional Derivatives

### Gradient Vector

The gradient of a function $`f(x, y)`$ is the vector of partial derivatives:

```math
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)
```

#### Properties of the Gradient

1. **Direction of Steepest Ascent**: The gradient points in the direction of maximum increase
2. **Magnitude**: $`\|\nabla f\|`$ gives the rate of change in the direction of steepest ascent
3. **Orthogonality**: The gradient is perpendicular to level curves
4. **Linear Approximation**: For small changes $`(\Delta x, \Delta y)`$:
```math
f(x + \Delta x, y + \Delta y) \approx f(x, y) + \frac{\partial f}{\partial x}\Delta x + \frac{\partial f}{\partial y}\Delta y
```

#### Geometric Interpretation

- **Direction**: Points in the direction of steepest ascent
- **Magnitude**: Indicates how steep the ascent is
- **Level Curves**: Gradient is perpendicular to level curves
- **Tangent Plane**: Gradient defines the normal vector to the tangent plane

#### Examples

**Example 1**: $`f(x, y) = x^2 + y^2`$
```math
\nabla f = (2x, 2y)
```

**Example 2**: $`f(x, y) = e^{x^2 + y^2}`$
```math
\nabla f = (2xe^{x^2 + y^2}, 2ye^{x^2 + y^2})
```

### Directional Derivatives

The directional derivative of $`f`$ in the direction of unit vector $`\mathbf{u} = (u_1, u_2)`$ is:

```math
D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}u_1 + \frac{\partial f}{\partial y}u_2
```

#### Properties

1. **Maximum Rate of Change**: The directional derivative is maximized when $`\mathbf{u}`$ points in the direction of $`\nabla f`$
2. **Minimum Rate of Change**: The directional derivative is minimized when $`\mathbf{u}`$ points in the opposite direction of $`\nabla f`$
3. **Zero Rate of Change**: The directional derivative is zero when $`\mathbf{u}`$ is perpendicular to $`\nabla f`$

#### Applications in Machine Learning

**Gradient Descent**:
The update rule in gradient descent uses the negative gradient:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
```

**Stochastic Gradient Descent**:
Uses noisy estimates of the gradient:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \hat{\nabla} f(\mathbf{x}_k)
```

**Adaptive Methods**:
Methods like Adam use gradient information to adapt learning rates:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\alpha}{\sqrt{v_k + \epsilon}} \nabla f(\mathbf{x}_k)
```

## 6.4 Optimization in Multiple Dimensions

### Critical Points and Classification

A critical point of $`f(x, y)`$ is a point where $`\nabla f = \mathbf{0}`$ or where the gradient is undefined.

#### Finding Critical Points

1. **Set partial derivatives to zero**:
```math
\frac{\partial f}{\partial x} = 0, \quad \frac{\partial f}{\partial y} = 0
```
2. **Solve the system of equations**
3. **Check for points where derivatives are undefined**

#### Second Derivative Test

For a critical point $`(a, b)`$, let:
```math
D = \frac{\partial^2 f}{\partial x^2}(a, b) \cdot \frac{\partial^2 f}{\partial y^2}(a, b) - \left(\frac{\partial^2 f}{\partial x \partial y}(a, b)\right)^2
```

Then:
- If $`D > 0`$ and $`\frac{\partial^2 f}{\partial x^2}(a, b) > 0`$: Local minimum
- If $`D > 0`$ and $`\frac{\partial^2 f}{\partial x^2}(a, b) < 0`$: Local maximum
- If $`D < 0`$: Saddle point
- If $`D = 0`$: Test is inconclusive

#### Examples

**Example 1**: $`f(x, y) = x^2 + y^2`$
- Critical point: $`(0, 0)`$
- $`D = 4 > 0`$ and $`\frac{\partial^2 f}{\partial x^2} = 2 > 0`$
- Result: Local minimum

**Example 2**: $`f(x, y) = x^2 - y^2`$
- Critical point: $`(0, 0)`$
- $`D = -4 < 0`$
- Result: Saddle point

### Applications in Machine Learning

**Loss Function Optimization**:
Finding the minimum of loss functions:
```math
\min_{\theta} L(\theta) = \min_{\theta} \frac{1}{m} \sum_{i=1}^m \ell(y_i, f_\theta(x_i))
```

**Regularization**:
Adding regularization terms to prevent overfitting:
```math
\min_{\theta} L(\theta) + \lambda \|\theta\|^2
```

**Constrained Optimization**:
Optimizing subject to constraints:
```math
\min_{\theta} L(\theta) \quad \text{subject to} \quad g(\theta) = 0
```

## 6.5 Lagrange Multipliers

### Constrained Optimization

Lagrange multipliers provide a method for finding extrema of a function subject to constraints.

#### Method

To find extrema of $`f(x, y)`$ subject to constraint $`g(x, y) = 0`$:

1. **Form the Lagrangian**:
```math
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda g(x, y)
```

2. **Set partial derivatives to zero**:
```math
\frac{\partial \mathcal{L}}{\partial x} = 0, \quad \frac{\partial \mathcal{L}}{\partial y} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0
```

3. **Solve the system of equations**

#### Geometric Interpretation

- The gradient of $`f`$ is parallel to the gradient of $`g`$ at the optimal point
- The constraint defines a curve (or surface) in the domain
- The optimal point occurs where the level curves of $`f`$ are tangent to the constraint curve

#### Examples

**Example 1**: Find the maximum of $`f(x, y) = xy`$ subject to $`x + y = 1`$

The Lagrangian is:
```math
\mathcal{L}(x, y, \lambda) = xy - \lambda(x + y - 1)
```

Setting partial derivatives to zero:
```math
\frac{\partial \mathcal{L}}{\partial x} = y - \lambda = 0
```

```math
\frac{\partial \mathcal{L}}{\partial y} = x - \lambda = 0
```

```math
\frac{\partial \mathcal{L}}{\partial \lambda} = -(x + y - 1) = 0
```

Solving: $`x = y = \frac{1}{2}`$, so the maximum is $`f(\frac{1}{2}, \frac{1}{2}) = \frac{1}{4}`$

### Applications in Machine Learning

**Regularization**:
Lagrange multipliers can be used to enforce constraints in optimization:
```math
\min_{\theta} L(\theta) \quad \text{subject to} \quad \|\theta\|^2 \leq C
```

**Support Vector Machines**:
The dual formulation of SVMs uses Lagrange multipliers:
```math
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)
```

**Neural Network Training**:
Constrained optimization in neural networks:
```math
\min_{\theta} L(\theta) \quad \text{subject to} \quad \|\theta\|_1 \leq \lambda
```

## 6.6 Applications in Machine Learning

### Gradient Descent in Multiple Dimensions

**Batch Gradient Descent**:
```math
\theta_{k+1} = \theta_k - \alpha \nabla L(\theta_k)
```

**Stochastic Gradient Descent**:
```math
\theta_{k+1} = \theta_k - \alpha \nabla L_i(\theta_k)
```
where $`L_i`$ is the loss for a single training example.

**Mini-Batch Gradient Descent**:
```math
\theta_{k+1} = \theta_k - \alpha \nabla L_B(\theta_k)
```
where $`L_B`$ is the loss for a batch of training examples.

### Advanced Optimization Methods

**Newton's Method**:
```math
\theta_{k+1} = \theta_k - H^{-1}(\theta_k) \nabla L(\theta_k)
```
where $`H`$ is the Hessian matrix.

**Quasi-Newton Methods**:
Methods like BFGS approximate the Hessian:
```math
\theta_{k+1} = \theta_k - B_k^{-1} \nabla L(\theta_k)
```

**Adam Optimizer**:
Combines momentum and adaptive learning rates:
```math
m_k = \beta_1 m_{k-1} + (1 - \beta_1) \nabla L(\theta_k)
```

```math
v_k = \beta_2 v_{k-1} + (1 - \beta_2) (\nabla L(\theta_k))^2
```

```math
\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{v_k} + \epsilon} m_k
```

### Loss Landscape Analysis

**Saddle Points**:
Understanding saddle points in high-dimensional loss landscapes:
```math
\text{Saddle Point} \iff \nabla L = 0 \text{ and } \lambda_{\min}(H) < 0 < \lambda_{\max}(H)
```

**Plateaus**:
Regions where gradients are very small:
```math
\|\nabla L\| < \epsilon
```

**Sharp Minima**:
Minima with large second derivatives:
```math
\lambda_{\min}(H) \gg 0
```

### Feature Interactions

**Hessian Analysis**:
The Hessian matrix reveals feature interactions:
```math
H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
```

**Feature Importance**:
Using gradient magnitudes to assess feature importance:
```math
\text{Importance}_i = \frac{1}{n} \sum_{j=1}^n \left|\frac{\partial L}{\partial x_i}\right|_{x=x_j}
```

**Interaction Effects**:
Mixed partial derivatives reveal interaction effects:
```math
\text{Interaction}_{ij} = \frac{\partial^2 L}{\partial x_i \partial x_j}
```

## 6.7 Advanced Topics

### Vector Fields and Flows

**Vector Fields**:
A vector field assigns a vector to each point in space:
```math
\mathbf{F}(x, y) = (P(x, y), Q(x, y))
```

**Gradient Fields**:
Vector fields that are gradients of scalar functions:
```math
\mathbf{F} = \nabla f
```

**Conservative Fields**:
Vector fields that are gradients of potential functions.

### Divergence and Curl

**Divergence**:
Measures how much a vector field spreads out:
```math
\text{div } \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y}
```

**Curl**:
Measures the rotation of a vector field:
```math
\text{curl } \mathbf{F} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}
```

### Applications in Machine Learning

**Gradient Flows**:
Understanding the flow of gradients during optimization:
```math
\frac{d\theta}{dt} = -\nabla L(\theta)
```

**Neural Tangent Kernel**:
Analyzing the dynamics of neural network training:
```math
K(x, y) = \mathbb{E}_{\theta} [\nabla f_\theta(x) \cdot \nabla f_\theta(y)]
```

**Information Geometry**:
Using differential geometry to understand parameter spaces:
```math
g_{ij} = \mathbb{E}_{x,y} \left[\frac{\partial \log p(y|x,\theta)}{\partial \theta_i} \frac{\partial \log p(y|x,\theta)}{\partial \theta_j}\right]
```

## Summary

Multivariable calculus provides the mathematical foundation for:

1. **High-Dimensional Optimization**: Understanding optimization in spaces with many variables
2. **Gradient-Based Learning**: Computing gradients and using them for optimization
3. **Feature Interactions**: Modeling complex relationships between variables
4. **Loss Landscape Analysis**: Understanding the geometry of loss functions
5. **Constrained Optimization**: Using Lagrange multipliers for constrained problems
6. **Vector Fields**: Understanding flows and dynamics in optimization

### Key Takeaways

- **Partial derivatives** measure change in specific directions
- **Gradients** point in the direction of steepest ascent
- **Critical points** occur where gradients are zero or undefined
- **Second derivative test** classifies critical points as minima, maxima, or saddle points
- **Lagrange multipliers** handle constrained optimization problems
- **Gradient descent** uses gradients for optimization in high dimensions
- **Loss landscapes** in neural networks are high-dimensional surfaces

### Next Steps

With a solid understanding of multivariable calculus, you're ready to explore:
- **Vector Calculus**: Understanding vector fields, divergence, and curl
- **Differential Geometry**: Understanding manifolds and curvature
- **Functional Analysis**: Extending to infinite-dimensional spaces
- **Optimization Theory**: Advanced optimization algorithms and convergence analysis 