# Optimization Techniques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

## Introduction

Optimization is the process of finding the best solution to a problem, typically involving finding the minimum or maximum of a function. This is fundamental to machine learning, where we optimize loss functions to train models. Calculus provides the mathematical foundation for most optimization algorithms used in machine learning and data science.

## 8.1 First and Second Derivative Tests

### Critical Points and Extrema

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize, minimize_scalar

# Find critical points using calculus
def find_critical_points():
    x = sp.Symbol('x')
    
    # Example function: f(x) = x³ - 3x² + 2
    f = x**3 - 3*x**2 + 2
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)
    
    print(f"f(x) = {f}")
    print(f"f'(x) = {f_prime}")
    print(f"f''(x) = {f_double_prime}")
    
    # Find critical points: f'(x) = 0
    critical_points = sp.solve(f_prime, x)
    print(f"Critical points: {critical_points}")
    
    # Classify critical points using second derivative test
    for point in critical_points:
        second_deriv = f_double_prime.subs(x, point)
        if second_deriv > 0:
            print(f"x = {point}: Local minimum (f''({point}) = {second_deriv})")
        elif second_deriv < 0:
            print(f"x = {point}: Local maximum (f''({point}) = {second_deriv})")
        else:
            print(f"x = {point}: Saddle point or inflection point")
    
    return f, f_prime, f_double_prime, critical_points

f, f_prime, f_double_prime, critical_points = find_critical_points()

# Visualize the function and its derivatives
x_vals = np.linspace(-1, 3, 1000)
y_vals = [f.subs(sp.Symbol('x'), x) for x in x_vals]
dy_vals = [f_prime.subs(sp.Symbol('x'), x) for x in x_vals]
ddy_vals = [f_double_prime.subs(sp.Symbol('x'), x) for x in x_vals]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 3x² + 2')
for point in critical_points:
    y_point = f.subs(sp.Symbol('x'), point)
    plt.scatter(point, y_point, c='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and Critical Points')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_vals, dy_vals, 'r-', linewidth=2, label="f'(x)")
plt.axhline(y=0, color='k', linestyle='--')
for point in critical_points:
    plt.scatter(point, 0, c='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('First Derivative')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_vals, ddy_vals, 'g-', linewidth=2, label="f''(x)")
plt.axhline(y=0, color='k', linestyle='--')
for point in critical_points:
    second_deriv = f_double_prime.subs(sp.Symbol('x'), point)
    plt.scatter(point, second_deriv, c='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.title('Second Derivative')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 8.2 Constrained Optimization

### Lagrange Multipliers

```python
# Lagrange multipliers for constrained optimization
def lagrange_multipliers_example():
    """
    Example: Maximize f(x,y) = xy subject to x + y = 10
    """
    x, y, lambda_var = sp.symbols('x y lambda')
    
    # Objective function: f(x,y) = xy
    f = x * y
    
    # Constraint: g(x,y) = x + y - 10 = 0
    g = x + y - 10
    
    # Lagrange function: L = f - λg
    L = f - lambda_var * g
    
    # Partial derivatives
    dL_dx = sp.diff(L, x)
    dL_dy = sp.diff(L, y)
    dL_dlambda = sp.diff(L, lambda_var)
    
    print("Lagrange equations:")
    print(f"∂L/∂x = {dL_dx} = 0")
    print(f"∂L/∂y = {dL_dy} = 0")
    print(f"∂L/∂λ = {dL_dlambda} = 0")
    
    # Solve the system of equations
    solution = sp.solve([dL_dx, dL_dy, dL_dlambda], [x, y, lambda_var])
    print(f"\nSolution: {solution}")
    
    # Verify the solution
    if solution:
        x_opt, y_opt, lambda_opt = solution[0]
        print(f"Optimal x = {x_opt}")
        print(f"Optimal y = {y_opt}")
        print(f"Optimal value = {f.subs([(x, x_opt), (y, y_opt)])}")
        print(f"Constraint satisfied: {g.subs([(x, x_opt), (y, y_opt)])}")

lagrange_multipliers_example()

# Visualize constrained optimization
def visualize_constrained_optimization():
    # Create grid
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Objective function: f(x,y) = xy
    Z = X * Y
    
    # Constraint: x + y = 10
    constraint_x = np.linspace(0, 10, 100)
    constraint_y = 10 - constraint_x
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot of objective function
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Constraint line
    plt.plot(constraint_x, constraint_y, 'r-', linewidth=3, label='Constraint: x + y = 10')
    
    # Optimal point
    plt.scatter(5, 5, c='red', s=200, zorder=5, label='Optimal point (5, 5)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Constrained Optimization: Maximize xy subject to x + y = 10')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_constrained_optimization()
```

## 8.3 Global vs Local Optimization

### Global Optimization Methods

```python
# Global optimization using different methods
def global_optimization_comparison():
    # Test function with multiple local minima
    def test_function(x):
        return np.sin(x) * np.exp(-x/10) + 0.1 * x**2
    
    def test_function_gradient(x):
        return np.cos(x) * np.exp(-x/10) - np.sin(x) * np.exp(-x/10) / 10 + 0.2 * x
    
    # Define search range
    x_range = (0, 20)
    
    # Method 1: Local optimization from multiple starting points
    def multi_start_optimization():
        n_starts = 10
        start_points = np.random.uniform(x_range[0], x_range[1], n_starts)
        results = []
        
        for start_point in start_points:
            result = minimize_scalar(test_function, bounds=x_range, 
                                   method='bounded', x0=start_point)
            results.append((result.x, result.fun))
        
        # Find the best result
        best_result = min(results, key=lambda x: x[1])
        return best_result
    
    # Method 2: Global optimization using differential evolution
    from scipy.optimize import differential_evolution
    
    def differential_evolution_optimization():
        result = differential_evolution(test_function, bounds=[x_range])
        return result.x, result.fun
    
    # Method 3: Grid search
    def grid_search_optimization():
        x_grid = np.linspace(x_range[0], x_range[1], 1000)
        y_grid = test_function(x_grid)
        min_idx = np.argmin(y_grid)
        return x_grid[min_idx], y_grid[min_idx]
    
    # Run all methods
    print("Global Optimization Comparison:")
    print("-" * 50)
    
    # Multi-start
    x_opt1, f_opt1 = multi_start_optimization()
    print(f"Multi-start optimization: x = {x_opt1:.6f}, f(x) = {f_opt1:.6f}")
    
    # Differential evolution
    x_opt2, f_opt2 = differential_evolution_optimization()
    print(f"Differential evolution: x = {x_opt2:.6f}, f(x) = {f_opt2:.6f}")
    
    # Grid search
    x_opt3, f_opt3 = grid_search_optimization()
    print(f"Grid search: x = {x_opt3:.6f}, f(x) = {f_opt3:.6f}")
    
    # Visualize
    x_vals = np.linspace(x_range[0], x_range[1], 1000)
    y_vals = test_function(x_vals)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = sin(x)*exp(-x/10) + 0.1x²')
    
    # Mark optimal points
    plt.scatter(x_opt1, f_opt1, c='red', s=100, label=f'Multi-start: ({x_opt1:.3f}, {f_opt1:.3f})')
    plt.scatter(x_opt2, f_opt2, c='green', s=100, label=f'Differential evolution: ({x_opt2:.3f}, {f_opt2:.3f})')
    plt.scatter(x_opt3, f_opt3, c='orange', s=100, label=f'Grid search: ({x_opt3:.3f}, {f_opt3:.3f})')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Global Optimization Methods Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

global_optimization_comparison()
```

## 8.4 Convex Optimization

### Convex Functions and Properties

```python
# Convex optimization examples
def convex_optimization_examples():
    # Example 1: Quadratic function (convex)
    def quadratic_function(x):
        return x**2 + 2*x + 1
    
    def quadratic_gradient(x):
        return 2*x + 2
    
    # Example 2: Non-convex function
    def non_convex_function(x):
        return x**3 - 3*x**2 + 2
    
    def non_convex_gradient(x):
        return 3*x**2 - 6*x
    
    # Test convexity using second derivative
    def test_convexity():
        x = sp.Symbol('x')
        
        # Quadratic function
        f1 = x**2 + 2*x + 1
        f1_double_prime = sp.diff(sp.diff(f1, x), x)
        print(f"f(x) = x² + 2x + 1")
        print(f"f''(x) = {f1_double_prime}")
        print(f"Convex: {f1_double_prime >= 0}")
        
        # Non-convex function
        f2 = x**3 - 3*x**2 + 2
        f2_double_prime = sp.diff(sp.diff(f2, x), x)
        print(f"\nf(x) = x³ - 3x² + 2")
        print(f"f''(x) = {f2_double_prime}")
        print(f"Convex: {f2_double_prime >= 0}")
    
    test_convexity()
    
    # Visualize convex vs non-convex functions
    x_vals = np.linspace(-2, 4, 1000)
    y1_vals = quadratic_function(x_vals)
    y2_vals = non_convex_function(x_vals)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y1_vals, 'b-', linewidth=2, label='f(x) = x² + 2x + 1 (Convex)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Convex Function')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y2_vals, 'r-', linewidth=2, label='f(x) = x³ - 3x² + 2 (Non-convex)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Non-convex Function')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

convex_optimization_examples()
```

## 8.5 Multi-objective Optimization

### Pareto Optimality

```python
# Multi-objective optimization example
def multi_objective_optimization():
    """
    Example: Minimize both f1(x) = x² and f2(x) = (x-2)²
    """
    def objective_functions(x):
        f1 = x**2
        f2 = (x - 2)**2
        return f1, f2
    
    # Generate Pareto front
    x_vals = np.linspace(0, 2, 100)
    f1_vals, f2_vals = objective_functions(x_vals)
    
    # Find Pareto optimal solutions
    pareto_indices = []
    for i in range(len(x_vals)):
        is_pareto = True
        for j in range(len(x_vals)):
            if i != j:
                # Check if point j dominates point i
                if f1_vals[j] <= f1_vals[i] and f2_vals[j] <= f2_vals[i] and \
                   (f1_vals[j] < f1_vals[i] or f2_vals[j] < f2_vals[i]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    plt.figure(figsize=(10, 8))
    
    # Plot all solutions
    plt.scatter(f1_vals, f2_vals, c='blue', alpha=0.6, label='All solutions')
    
    # Plot Pareto optimal solutions
    pareto_f1 = f1_vals[pareto_indices]
    pareto_f2 = f2_vals[pareto_indices]
    plt.scatter(pareto_f1, pareto_f2, c='red', s=100, label='Pareto optimal solutions')
    
    # Connect Pareto optimal solutions
    plt.plot(pareto_f1, pareto_f2, 'r-', linewidth=2, label='Pareto front')
    
    plt.xlabel('f₁(x) = x²')
    plt.ylabel('f₂(x) = (x-2)²')
    plt.title('Multi-objective Optimization: Pareto Front')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Pareto optimal solutions:")
    for i in pareto_indices:
        print(f"x = {x_vals[i]:.3f}, f₁ = {f1_vals[i]:.3f}, f₂ = {f2_vals[i]:.3f}")

multi_objective_optimization()
```

## 8.6 Optimization in Machine Learning

### Hyperparameter Optimization

```python
# Hyperparameter optimization example
def hyperparameter_optimization():
    """
    Example: Optimize learning rate and regularization for a simple model
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define objective function for optimization
    def objective_function(params):
        learning_rate, alpha = params
        
        # Simple model with Ridge regression
        model = Ridge(alpha=alpha)
        
        # Cross-validation score (negative because we want to maximize)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)  # Return positive MSE
    
    # Grid search
    learning_rates = np.logspace(-3, 0, 10)
    alphas = np.logspace(-3, 2, 10)
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for lr in learning_rates:
        for alpha in alphas:
            score = objective_function([lr, alpha])
            results.append((lr, alpha, score))
            
            if score < best_score:
                best_score = score
                best_params = (lr, alpha)
    
    print(f"Best parameters: learning_rate = {best_params[0]:.6f}, alpha = {best_params[1]:.6f}")
    print(f"Best score (MSE): {best_score:.6f}")
    
    # Visualize results
    results = np.array(results)
    lr_vals = results[:, 0]
    alpha_vals = results[:, 1]
    scores = results[:, 2]
    
    plt.figure(figsize=(12, 5))
    
    # 3D scatter plot
    ax1 = plt.subplot(121, projection='3d')
    scatter = ax1.scatter(lr_vals, alpha_vals, scores, c=scores, cmap='viridis')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Alpha')
    ax1.set_zlabel('MSE')
    ax1.set_title('Hyperparameter Optimization Results')
    plt.colorbar(scatter)
    
    # 2D contour plot
    ax2 = plt.subplot(122)
    # Create grid for contour plot
    lr_grid = np.logspace(-3, 0, 20)
    alpha_grid = np.logspace(-3, 2, 20)
    LR, ALPHA = np.meshgrid(lr_grid, alpha_grid)
    
    # Interpolate scores for contour plot
    from scipy.interpolate import griddata
    points = np.column_stack((lr_vals, alpha_vals))
    grid_scores = griddata(points, scores, (LR, ALPHA), method='linear')
    
    contour = ax2.contour(LR, ALPHA, grid_scores, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter(best_params[0], best_params[1], c='red', s=200, marker='*', 
                label=f'Best: ({best_params[0]:.3f}, {best_params[1]:.3f})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Alpha')
    ax2.set_title('Contour Plot of MSE')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

hyperparameter_optimization()
```

## Summary

- **First and second derivative tests** help identify local extrema
- **Constrained optimization** uses Lagrange multipliers for equality constraints
- **Global optimization** methods find the best solution across the entire domain
- **Convex optimization** guarantees global optimality for convex problems
- **Multi-objective optimization** finds Pareto optimal solutions
- **Hyperparameter optimization** is crucial for machine learning model tuning

## Next Steps

Understanding optimization techniques enables you to design efficient algorithms, tune machine learning models, and solve complex real-world problems. The next section covers numerical methods for when analytical solutions are not available. 