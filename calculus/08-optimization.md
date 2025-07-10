# Optimization Techniques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Optimization is the process of finding the best solution to a problem, typically involving finding the minimum or maximum of a function. This is fundamental to machine learning, where we optimize loss functions to train models. Calculus provides the mathematical foundation for most optimization algorithms used in machine learning and data science.

### Why Optimization Matters in AI/ML

Optimization is the core of machine learning and artificial intelligence:

1. **Model Training**: Finding optimal parameters that minimize loss functions
2. **Hyperparameter Tuning**: Optimizing learning rates, regularization parameters, etc.
3. **Feature Selection**: Finding optimal subsets of features
4. **Neural Network Architecture**: Optimizing network structure and connections
5. **Reinforcement Learning**: Finding optimal policies and value functions
6. **Bayesian Optimization**: Efficiently exploring parameter spaces
7. **Multi-objective Optimization**: Balancing multiple competing objectives
8. **Constrained Optimization**: Handling physical and mathematical constraints

### Mathematical Foundation

Optimization problems can be formulated as:

**Unconstrained Optimization**: 
```math
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{or} \quad \max_{\mathbf{x}} f(\mathbf{x})
```

**Constrained Optimization**: 
```math
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) = 0, \quad h_j(\mathbf{x}) \leq 0
```

where $`f(\mathbf{x})`$ is the objective function, and $`g_i(\mathbf{x})`$, $`h_j(\mathbf{x})`$ are constraint functions.

### Types of Optimization Problems

1. **Linear Programming**: Linear objective and constraints
2. **Nonlinear Programming**: Nonlinear objective and/or constraints
3. **Convex Optimization**: Convex objective and constraints
4. **Non-convex Optimization**: Non-convex objective or constraints
5. **Discrete Optimization**: Integer or binary variables
6. **Stochastic Optimization**: Objective functions with random components
7. **Multi-objective Optimization**: Multiple competing objectives

### Optimization Landscape

Understanding the optimization landscape is crucial:
- **Local Minima/Maxima**: Points where the function is lower/higher than nearby points
- **Global Minima/Maxima**: Points where the function attains its lowest/highest value
- **Saddle Points**: Points where the gradient is zero but not an extremum
- **Plateaus**: Regions where the gradient is very small
- **Valleys and Ridges**: Long, narrow regions of low/high function values

## 8.1 First and Second Derivative Tests

### Critical Points and Extrema

The foundation of optimization lies in understanding where functions attain their extreme values. Critical points are where the first derivative is zero or undefined.

#### Mathematical Theory

**First Derivative Test**: If $`f'(c) = 0`$ and $`f'(x)`$ changes sign at $`x = c`$, then $`f`$ has a local extremum at $`c`$.

**Second Derivative Test**: If $`f'(c) = 0`$, then:
- If $`f''(c) > 0`$: $`f`$ has a local minimum at $`c`$
- If $`f''(c) < 0`$: $`f`$ has a local maximum at $`c`$
- If $`f''(c) = 0`$: The test is inconclusive

#### Multivariable Case

For functions of multiple variables $`f(\mathbf{x})`$:

**Critical Points**: Points where $`\nabla f = \mathbf{0}`$

**Second Derivative Test**: For a critical point $`\mathbf{c}`$, let $`H`$ be the Hessian matrix:
```math
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{c})
```

Then:
- If all eigenvalues of $`H`$ are positive: Local minimum
- If all eigenvalues of $`H`$ are negative: Local maximum
- If eigenvalues have mixed signs: Saddle point

#### Examples

**Example 1**: $`f(x) = x^2`$
- Critical point: $`x = 0`$
- $`f''(0) = 2 > 0`$
- Result: Local minimum

**Example 2**: $`f(x, y) = x^2 + y^2`$
- Critical point: $`(0, 0)`$
- Hessian: $`H = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}`$
- Eigenvalues: $`\lambda_1 = \lambda_2 = 2 > 0`$
- Result: Local minimum

**Example 3**: $`f(x, y) = x^2 - y^2`$
- Critical point: $`(0, 0)`$
- Hessian: $`H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}`$
- Eigenvalues: $`\lambda_1 = 2 > 0`$, $`\lambda_2 = -2 < 0`$
- Result: Saddle point

### Why These Tests Matter

1. **Gradient Descent**: Relies on first derivatives to find descent directions
2. **Newton's Method**: Uses second derivatives for faster convergence
3. **Convergence Analysis**: Understanding local vs global optima
4. **Model Interpretability**: Understanding where models are most sensitive
5. **Optimization Algorithm Selection**: Choosing appropriate methods based on landscape

## 8.2 Gradient-Based Methods

### Gradient Descent

Gradient descent is the most fundamental optimization algorithm:

```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
```

where $`\alpha`$ is the learning rate.

#### Properties

1. **Convergence**: Converges to local minima for sufficiently small learning rates
2. **Linear Convergence**: Convergence rate depends on the condition number of the Hessian
3. **Sensitivity**: Performance depends heavily on the choice of learning rate

#### Variants

**Stochastic Gradient Descent (SGD)**:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f_i(\mathbf{x}_k)
```

where $`f_i`$ is the loss for a single training example.

**Mini-Batch Gradient Descent**:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f_B(\mathbf{x}_k)
```

where $`f_B`$ is the loss for a batch of training examples.

**Momentum**:
```math
\mathbf{v}_{k+1} = \beta \mathbf{v}_k - \alpha \nabla f(\mathbf{x}_k)
```

```math
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}
```

### Newton's Method

Newton's method uses second derivatives for faster convergence:

```math
\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)
```

where $`H(\mathbf{x}_k)`$ is the Hessian matrix.

#### Properties

1. **Quadratic Convergence**: Much faster than gradient descent near the optimum
2. **Computational Cost**: Requires computing and inverting the Hessian
3. **Robustness**: Less sensitive to learning rate choice

#### Quasi-Newton Methods

Methods like BFGS approximate the Hessian:

```math
\mathbf{x}_{k+1} = \mathbf{x}_k - B_k^{-1} \nabla f(\mathbf{x}_k)
```

where $`B_k`$ is an approximation of the Hessian.

### Adaptive Methods

#### Adam Optimizer

Adam combines momentum and adaptive learning rates:

```math
\mathbf{m}_{k+1} = \beta_1 \mathbf{m}_k + (1 - \beta_1) \nabla f(\mathbf{x}_k)
```

```math
\mathbf{v}_{k+1} = \beta_2 \mathbf{v}_k + (1 - \beta_2) (\nabla f(\mathbf{x}_k))^2
```

```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\alpha}{\sqrt{\mathbf{v}_{k+1}} + \epsilon} \mathbf{m}_{k+1}
```

#### RMSprop

RMSprop adapts learning rates based on gradient magnitudes:

```math
\mathbf{v}_{k+1} = \beta \mathbf{v}_k + (1 - \beta) (\nabla f(\mathbf{x}_k))^2
```

```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\alpha}{\sqrt{\mathbf{v}_{k+1}} + \epsilon} \nabla f(\mathbf{x}_k)
```

## 8.3 Constrained Optimization

### Lagrange Multipliers

Lagrange multipliers provide a method for constrained optimization.

#### Equality Constraints

For the problem:
```math
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) = 0
```

Form the Lagrangian:
```math
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) - \sum_i \lambda_i g_i(\mathbf{x})
```

The optimal solution satisfies:
```math
\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{0}, \quad \nabla_{\boldsymbol{\lambda}} \mathcal{L} = \mathbf{0}
```

#### Inequality Constraints

For the problem:
```math
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad h_j(\mathbf{x}) \leq 0
```

The Karush-Kuhn-Tucker (KKT) conditions are:
```math
\nabla f(\mathbf{x}) - \sum_j \mu_j \nabla h_j(\mathbf{x}) = \mathbf{0}
```

```math
\mu_j h_j(\mathbf{x}) = 0, \quad \mu_j \geq 0
```

#### Examples

**Example 1**: Maximize $`f(x, y) = xy`$ subject to $`x + y = 10`$

The Lagrangian is:
```math
\mathcal{L}(x, y, \lambda) = xy - \lambda(x + y - 10)
```

Setting derivatives to zero:
```math
\frac{\partial \mathcal{L}}{\partial x} = y - \lambda = 0
```

```math
\frac{\partial \mathcal{L}}{\partial y} = x - \lambda = 0
```

```math
\frac{\partial \mathcal{L}}{\partial \lambda} = -(x + y - 10) = 0
```

Solving: $`x = y = 5`$, so the maximum is $`f(5, 5) = 25`$

### Penalty Methods

Penalty methods convert constrained problems to unconstrained ones:

```math
\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\mu}{2} \sum_i g_i(\mathbf{x})^2 + \frac{\mu}{2} \sum_j [h_j(\mathbf{x})]_+^2
```

where $`[x]_+ = \max(0, x)`$ and $`\mu`$ is a penalty parameter.

### Barrier Methods

Barrier methods use logarithmic barriers for inequality constraints:

```math
\min_{\mathbf{x}} f(\mathbf{x}) - \frac{1}{t} \sum_j \log(-h_j(\mathbf{x}))
```

where $`t > 0`$ is a barrier parameter.

## 8.4 Global vs Local Optimization

### Global Optimization Methods

#### Grid Search

Systematically evaluate function at grid points:
```math
\mathbf{x}_{i,j} = \mathbf{a} + (i, j) \Delta \mathbf{x}
```

#### Random Search

Randomly sample points from the domain:
```math
\mathbf{x}_k \sim \text{Uniform}(\mathcal{D})
```

#### Genetic Algorithms

Evolutionary algorithms that maintain a population of solutions:
```math
\mathbf{x}_{k+1} = \text{Mutation}(\text{Crossover}(\mathbf{x}_k, \mathbf{y}_k))
```

#### Simulated Annealing

Probabilistic method that accepts worse solutions with decreasing probability:
```math
P(\text{accept}) = \exp\left(-\frac{\Delta f}{T_k}\right)
```

where $`T_k`$ is the temperature at iteration $`k`$.

### Bayesian Optimization

Bayesian optimization uses probabilistic models to guide search:

```math
\mathbf{x}_{k+1} = \arg\max_{\mathbf{x}} \text{Acquisition}(\mathbf{x})
```

Common acquisition functions include:
- **Expected Improvement**: $`\text{EI}(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f^*)]`$
- **Upper Confidence Bound**: $`\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \sigma(\mathbf{x})`$
- **Probability of Improvement**: $`\text{PI}(\mathbf{x}) = P(f(\mathbf{x}) > f^*)`$

## 8.5 Convex Optimization

### Convex Functions and Properties

A function $`f`$ is convex if:
```math
f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})
```

for all $`\mathbf{x}, \mathbf{y}`$ in the domain and $`\lambda \in [0, 1]`$.

#### Properties

1. **Local Minima are Global**: Any local minimum is a global minimum
2. **First-Order Condition**: $`f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})`$
3. **Second-Order Condition**: $`\nabla^2 f(\mathbf{x}) \succeq 0`$ (positive semidefinite)

#### Examples

**Convex Functions**:
- $`f(x) = x^2`$
- $`f(x) = e^x`$
- $`f(\mathbf{x}) = \|\mathbf{x}\|_2^2`$
- $`f(\mathbf{x}) = \log(1 + e^{\mathbf{a}^T\mathbf{x}})`$

**Non-Convex Functions**:
- $`f(x) = x^3`$
- $`f(x) = \sin(x)`$
- $`f(\mathbf{x}) = \|\mathbf{x}\|_1`$

### Convex Optimization Problems

Standard form:
```math
\min_{\mathbf{x}} f_0(\mathbf{x}) \quad \text{subject to} \quad f_i(\mathbf{x}) \leq 0, \quad A\mathbf{x} = \mathbf{b}
```

where $`f_0, f_1, \ldots, f_m`$ are convex functions.

#### Duality

The dual problem is:
```math
\max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \quad \text{subject to} \quad \boldsymbol{\lambda} \geq \mathbf{0}
```

where:
```math
g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \left(f_0(\mathbf{x}) + \sum_i \lambda_i f_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})\right)
```

#### Interior Point Methods

Interior point methods solve convex problems by:
1. Adding barrier functions for inequality constraints
2. Using Newton's method to solve the barrier problem
3. Reducing the barrier parameter to approach the original problem

## 8.6 Multi-objective Optimization

### Pareto Optimality

A solution $`\mathbf{x}^*`$ is Pareto optimal if there exists no $`\mathbf{x}`$ such that:
```math
f_i(\mathbf{x}) \leq f_i(\mathbf{x}^*) \quad \forall i
```

and:
```math
f_j(\mathbf{x}) < f_j(\mathbf{x}^*) \quad \text{for some } j
```

#### Methods

**Weighted Sum Method**:
```math
\min_{\mathbf{x}} \sum_i w_i f_i(\mathbf{x})
```

**ε-Constraint Method**:
```math
\min_{\mathbf{x}} f_1(\mathbf{x}) \quad \text{subject to} \quad f_i(\mathbf{x}) \leq \epsilon_i
```

**Goal Programming**:
```math
\min_{\mathbf{x}} \sum_i w_i |f_i(\mathbf{x}) - g_i|
```

where $`g_i`$ are goal values.

### Evolutionary Multi-objective Optimization

**NSGA-II (Non-dominated Sorting Genetic Algorithm II)**:
1. Maintain a population of solutions
2. Sort solutions by non-domination rank
3. Use crowding distance for diversity
4. Apply genetic operators (crossover, mutation)

## 8.7 Optimization in Machine Learning

### Hyperparameter Optimization

#### Grid Search

Systematic search over hyperparameter grid:
```python
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
for lr in learning_rates:
    for bs in batch_sizes:
        train_model(lr, bs)
```

#### Random Search

Random sampling of hyperparameters:
```python
for _ in range(n_trials):
    lr = np.random.uniform(1e-4, 1e-1)
    bs = np.random.choice([16, 32, 64, 128])
    train_model(lr, bs)
```

#### Bayesian Optimization

Using probabilistic models to guide search:
```python
def objective(hyperparams):
    return train_and_evaluate(hyperparams)

optimizer = BayesianOptimization(
    f=objective,
    pbounds={'lr': (1e-4, 1e-1), 'bs': (16, 128)}
)
optimizer.maximize(init_points=5, n_iter=20)
```

### Neural Network Optimization

#### Backpropagation

Computing gradients through the network:
```math
\frac{\partial L}{\partial W_{ij}^{(l)}} = \frac{\partial L}{\partial a_i^{(l+1)}} \frac{\partial a_i^{(l+1)}}{\partial W_{ij}^{(l)}}
```

#### Batch Normalization

Normalizing activations to improve training:
```math
\text{BN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

#### Dropout

Regularization technique that randomly sets activations to zero:
```math
\text{Dropout}(x) = \text{mask} \odot x
```

where $`\text{mask}`$ is a binary mask with elements drawn from Bernoulli distribution.

### Advanced Optimization Techniques

#### Natural Gradient

Using the Fisher information matrix:
```math
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha F^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)
```

where $`F(\mathbf{x})`$ is the Fisher information matrix.

#### Mirror Descent

Using Bregman divergences:
```math
\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} \langle \nabla f(\mathbf{x}_k), \mathbf{x} - \mathbf{x}_k \rangle + \frac{1}{\alpha} D_\phi(\mathbf{x}, \mathbf{x}_k)
```

where $`D_\phi`$ is the Bregman divergence.

#### Proximal Methods

Handling non-smooth objectives:
```math
\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} f(\mathbf{x}) + \frac{1}{2\alpha} \|\mathbf{x} - \mathbf{x}_k\|^2
```

## 8.8 Convergence Analysis

### Convergence Rates

#### Linear Convergence

```math
\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq \rho \|\mathbf{x}_k - \mathbf{x}^*\|
```

where $`\rho \in (0, 1)`$.

#### Quadratic Convergence

```math
\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C \|\mathbf{x}_k - \mathbf{x}^*\|^2
```

#### Sublinear Convergence

```math
\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq \frac{C}{\sqrt{k}}
```

### Conditions for Convergence

#### Lipschitz Continuity

```math
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|
```

#### Strong Convexity

```math
f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|^2
```

where $`\mu > 0`$.

## Summary

Optimization provides powerful tools for:

1. **Gradient-Based Methods**: Gradient descent, Newton's method, and their variants
2. **Constrained Optimization**: Lagrange multipliers, penalty methods, and barrier methods
3. **Global Optimization**: Grid search, random search, genetic algorithms, and Bayesian optimization
4. **Convex Optimization**: Guaranteed global optimality for convex problems
5. **Multi-objective Optimization**: Pareto optimality and evolutionary methods
6. **Machine Learning Applications**: Hyperparameter tuning and neural network training

### Key Takeaways

- **First and second derivative tests** help identify local extrema
- **Gradient descent** is the foundation of most ML optimization algorithms
- **Newton's method** provides faster convergence but higher computational cost
- **Constrained optimization** uses Lagrange multipliers and KKT conditions
- **Convex optimization** guarantees global optimality for convex problems
- **Bayesian optimization** efficiently explores parameter spaces
- **Multi-objective optimization** balances competing objectives
- **Convergence analysis** helps understand algorithm performance

### Next Steps

With a solid understanding of optimization techniques, you're ready to explore:
- **Numerical Methods**: When analytical solutions are not available
- **Machine Learning Applications**: Advanced optimization in neural networks
- **Convex Analysis**: Deepening understanding of convex optimization
- **Stochastic Optimization**: Handling uncertainty in optimization problems 