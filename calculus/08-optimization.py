# Python code extracted from 08-optimization.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize, minimize_scalar

# Comprehensive critical point analysis
def find_critical_points_comprehensive():
    x = sp.Symbol('x')
    
    print("=== CRITICAL POINT ANALYSIS ===\n")
    
    # Example 1: Polynomial function
    print("1. POLYNOMIAL FUNCTION")
    f1 = x**3 - 3*x**2 + 2
    f1_prime = sp.diff(f1, x)
    f1_double_prime = sp.diff(f1_prime, x)
    
    print(f"   f(x) = {f1}")
    print(f"   f'(x) = {f1_prime}")
    print(f"   f''(x) = {f1_double_prime}")
    
    # Find critical points
    critical_points1 = sp.solve(f1_prime, x)
    print(f"   Critical points: {critical_points1}")
    
    # Classify critical points
    for point in critical_points1:
        second_deriv = f1_double_prime.subs(x, point)
        func_value = f1.subs(x, point)
        
        if second_deriv > 0:
            print(f"   x = {point}: Local minimum (f''({point}) = {second_deriv}, f({point}) = {func_value})")
        elif second_deriv < 0:
            print(f"   x = {point}: Local maximum (f''({point}) = {second_deriv}, f({point}) = {func_value})")
        else:
            print(f"   x = {point}: Saddle point or inflection point (f''({point}) = {second_deriv})")
    
    # Example 2: Trigonometric function
    print("\n2. TRIGONOMETRIC FUNCTION")
    f2 = sp.sin(x) + 0.5*x**2
    f2_prime = sp.diff(f2, x)
    f2_double_prime = sp.diff(f2_prime, x)
    
    print(f"   f(x) = {f2}")
    print(f"   f'(x) = {f2_prime}")
    print(f"   f''(x) = {f2_double_prime}")
    
    # Find critical points numerically for complex functions
    critical_points2 = sp.solve(f2_prime, x)
    print(f"   Critical points: {critical_points2}")
    
    # Example 3: Exponential function
    print("\n3. EXPONENTIAL FUNCTION")
    f3 = sp.exp(-x**2/2) * sp.sin(x)
    f3_prime = sp.diff(f3, x)
    f3_double_prime = sp.diff(f3_prime, x)
    
    print(f"   f(x) = {f3}")
    print(f"   f'(x) = {f3_prime}")
    print(f"   f''(x) = {f3_double_prime}")
    
    return f1, f1_prime, f1_double_prime, critical_points1

f1, f1_prime, f1_double_prime, critical_points1 = find_critical_points_comprehensive()

# Advanced visualization of critical points
def visualize_critical_points_advanced():
    x_vals = np.linspace(-1, 3, 1000)
    
    # Convert symbolic expressions to numerical functions
    f_numeric = sp.lambdify(sp.Symbol('x'), f1)
    f_prime_numeric = sp.lambdify(sp.Symbol('x'), f1_prime)
    f_double_prime_numeric = sp.lambdify(sp.Symbol('x'), f1_double_prime)
    
    y_vals = f_numeric(x_vals)
    dy_vals = f_prime_numeric(x_vals)
    ddy_vals = f_double_prime_numeric(x_vals)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Function and critical points
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 3x² + 2')
    for point in critical_points1:
        y_point = f_numeric(point)
        ax1.scatter(point, y_point, c='red', s=100, zorder=5, 
                   label=f'Critical point: ({point:.2f}, {y_point:.2f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function and Critical Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # First derivative
    ax2.plot(x_vals, dy_vals, 'r-', linewidth=2, label="f'(x)")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in critical_points1:
        ax2.scatter(point, 0, c='red', s=100, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title('First Derivative (Critical Points at Zeros)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Second derivative
    ax3.plot(x_vals, ddy_vals, 'g-', linewidth=2, label="f''(x)")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for point in critical_points1:
        second_deriv = f_double_prime_numeric(point)
        ax3.scatter(point, second_deriv, c='red', s=100, zorder=5,
                   label=f'f''({point:.2f}) = {second_deriv:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel("f''(x)")
    ax3.set_title('Second Derivative (Classifies Critical Points)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Optimization landscape
    ax4.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    ax4.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    
    # Mark different regions
    for i, point in enumerate(critical_points1):
        y_point = f_numeric(point)
        second_deriv = f_double_prime_numeric(point)
        
        if second_deriv > 0:
            ax4.scatter(point, y_point, c='green', s=200, zorder=5, 
                       marker='^', label=f'Local min: ({point:.2f}, {y_point:.2f})')
        elif second_deriv < 0:
            ax4.scatter(point, y_point, c='red', s=200, zorder=5, 
                       marker='v', label=f'Local max: ({point:.2f}, {y_point:.2f})')
        else:
            ax4.scatter(point, y_point, c='orange', s=200, zorder=5, 
                       marker='s', label=f'Saddle point: ({point:.2f}, {y_point:.2f})')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Optimization Landscape')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_critical_points_advanced()

# Numerical optimization comparison
def numerical_optimization_comparison():
    print("\n=== NUMERICAL OPTIMIZATION COMPARISON ===\n")
    
    def objective_function(x):
        return x**3 - 3*x**2 + 2
    
    def gradient_function(x):
        return 3*x**2 - 6*x
    
    def hessian_function(x):
        return 6*x - 6
    
    # Test different optimization methods
    methods = [
        ('BFGS', 'BFGS (quasi-Newton)'),
        ('CG', 'Conjugate Gradient'),
        ('L-BFGS-B', 'L-BFGS-B (bounded)'),
        ('TNC', 'Truncated Newton'),
        ('SLSQP', 'Sequential Least Squares')
    ]
    
    # Test from different starting points
    starting_points = [-0.5, 0.5, 1.5, 2.5]
    
    print("Optimization Results from Different Starting Points:")
    print("Method\t\tStart\t\tOptimal x\tOptimal f(x)\tIterations")
    print("-" * 80)
    
    for method_name, method_desc in methods:
        for start_point in starting_points:
            try:
                result = minimize(objective_function, start_point, 
                                method=method_name, jac=gradient_function)
                
                print(f"{method_name:12s}\t{start_point:6.1f}\t{result.x[0]:10.6f}\t{result.fun:12.6f}\t{result.nit:10d}")
            except:
                print(f"{method_name:12s}\t{start_point:6.1f}\t{'Failed':>10s}\t{'N/A':>12s}\t{'N/A':>10s}")
        print()

numerical_optimization_comparison()

# Advanced critical point analysis with multiple functions
def advanced_critical_point_analysis():
    print("\n=== ADVANCED CRITICAL POINT ANALYSIS ===\n")
    
    x = sp.Symbol('x')
    
    # Test functions with different characteristics
    test_functions = [
        (x**4 - 4*x**2, "Quartic function with multiple extrema"),
        (sp.sin(x) + 0.1*x**2, "Sine with quadratic trend"),
        (sp.exp(-x**2/2) * sp.cos(x), "Gaussian modulated cosine"),
        (x**3 - x, "Cubic with inflection point"),
        (sp.log(1 + x**2), "Logarithmic function")
    ]
    
    for i, (func, description) in enumerate(test_functions, 1):
        print(f"{i}. {description}")
        print(f"   f(x) = {func}")
        
        # Compute derivatives
        f_prime = sp.diff(func, x)
        f_double_prime = sp.diff(f_prime, x)
        
        print(f"   f'(x) = {f_prime}")
        print(f"   f''(x) = {f_double_prime}")
        
        # Find critical points
        try:
            critical_points = sp.solve(f_prime, x)
            print(f"   Critical points: {critical_points}")
            
            # Classify critical points
            for point in critical_points:
                if point.is_real:  # Only consider real critical points
                    second_deriv = f_double_prime.subs(x, point)
                    func_value = func.subs(x, point)
                    
                    if second_deriv > 0:
                        print(f"     x = {point}: Local minimum (f''({point}) = {second_deriv:.4f})")
                    elif second_deriv < 0:
                        print(f"     x = {point}: Local maximum (f''({point}) = {second_deriv:.4f})")
                    else:
                        print(f"     x = {point}: Saddle point or inflection point")
        except:
            print(f"   Critical points: Could not solve analytically")
        
        print()

advanced_critical_point_analysis()

# Convergence analysis of optimization methods
def convergence_analysis():
    print("\n=== CONVERGENCE ANALYSIS ===\n")
    
    def objective_function(x):
        return x**3 - 3*x**2 + 2
    
    def gradient_function(x):
        return 3*x**2 - 6*x
    
    # Track optimization progress
    def track_optimization(start_point, method='BFGS'):
        history = []
        
        def callback(xk):
            history.append(xk[0])
        
        result = minimize(objective_function, start_point, 
                         method=method, jac=gradient_function,
                         callback=callback, options={'maxiter': 100})
        
        return result, history
    
    # Test different starting points
    starting_points = [0.5, 1.5, 2.5]
    methods = ['BFGS', 'CG', 'L-BFGS-B']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, method in enumerate(methods):
        ax = axes[i]
        
        for start_point in starting_points:
            result, history = track_optimization(start_point, method)
            
            # Plot convergence
            iterations = range(len(history))
            ax.plot(iterations, history, 'o-', linewidth=2, 
                   label=f'Start: {start_point}')
            
            # Mark final point
            ax.scatter(len(history)-1, history[-1], c='red', s=100, zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('x value')
        ax.set_title(f'{method} Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze convergence rates
    print("Convergence Analysis:")
    for method in methods:
        print(f"\n{method} method:")
        for start_point in starting_points:
            result, history = track_optimization(start_point, method)
            print(f"  Start {start_point}: {len(history)} iterations, final x = {result.x[0]:.6f}")

convergence_analysis()

### Applications in Machine Learning

# First and second derivative tests are fundamental to:

# 1. **Gradient Descent**: Uses first derivatives to find descent directions
# 2. **Newton's Method**: Uses both first and second derivatives for faster convergence
# 3. **Loss Function Analysis**: Understanding where loss functions have minima
# 4. **Model Convergence**: Analyzing whether optimization algorithms will converge
# 5. **Hyperparameter Optimization**: Finding optimal learning rates and other parameters

## 8.2 Constrained Optimization

### Lagrange Multipliers

# Code Block 2
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

# Code Block 3
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

# Code Block 4
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

# Code Block 5
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

# --- Advanced Optimization Topics: Penalty/Barrier, Bayesian, Natural Gradient, Mirror Descent, Proximal ---

# Penalty and Barrier Methods for Constrained Optimization

def penalty_method_example():
    """Solve a simple constrained optimization problem using a penalty method."""
    # Minimize f(x, y) = (x-1)^2 + (y-2)^2 subject to x + y = 3
    def objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    def constraint(x):
        return x[0] + x[1] - 3
    def penalty_objective(x, mu=100):
        return objective(x) + mu * constraint(x)**2
    # Unconstrained minimization of penalized objective
    result = minimize(penalty_objective, [0, 0])
    print(f"Penalty method solution: x = {result.x}, f(x) = {objective(result.x):.4f}, constraint = {constraint(result.x):.4f}")

penalty_method_example()


def barrier_method_example():
    """Solve a simple constrained optimization problem using a barrier method."""
    # Minimize f(x) = (x-1)^2 subject to x > 0
    def objective(x):
        return (x[0] - 1)**2
    def barrier_objective(x, t=1):
        if x[0] <= 0:
            return np.inf
        return objective(x) - (1/t) * np.log(x[0])
    result = minimize(barrier_objective, [0.5], bounds=[(1e-6, None)])
    print(f"Barrier method solution: x = {result.x[0]:.4f}, f(x) = {objective(result.x):.4f}")

barrier_method_example()

# Bayesian Optimization (stub/example)
try:
    from skopt import gp_minimize
    def bayesian_optimization_example():
        """Bayesian optimization using Gaussian processes (requires scikit-optimize)."""
        def objective(x):
            return (x[0] - 2)**2 + np.sin(5 * x[0])
        res = gp_minimize(objective, [(0.0, 4.0)], n_calls=15, random_state=0)
        print(f"Bayesian optimization best x: {res.x[0]:.4f}, f(x) = {res.fun:.4f}")
    bayesian_optimization_example()
except ImportError:
    print("scikit-optimize not installed; skipping Bayesian optimization example.")

# Natural Gradient (symbolic demonstration)
def natural_gradient_example():
    """Symbolic demonstration of natural gradient for a normal distribution."""
    mu, sigma, x = sp.symbols('mu sigma x')
    logp = -0.5 * sp.log(2 * sp.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)
    dlogp_dmu = sp.diff(logp, mu)
    fisher_info = sp.simplify(sp.integrate(dlogp_dmu**2 * (1/(sp.sqrt(2*sp.pi)*sigma)) * sp.exp(-(x-mu)**2/(2*sigma**2)), (x, -sp.oo, sp.oo)))
    grad = sp.Symbol('grad')
    nat_grad = grad / fisher_info
    print(f"Fisher information: {fisher_info}, natural gradient: grad / {fisher_info}")

natural_gradient_example()

# Mirror Descent (illustrative code)
def mirror_descent_example():
    """Illustrative mirror descent for minimizing f(x) = x^2 with KL divergence as mirror map."""
    # Mirror map: phi(x) = x*log(x) - x (for x > 0)
    # Bregman divergence: D_phi(x, y) = x*log(x/y) - x + y
    x = 2.0
    for i in range(5):
        grad = 2 * x
        # Mirror step (exponential update)
        x_new = x * np.exp(-0.1 * grad)
        print(f"Iter {i}: x = {x:.4f}, grad = {grad:.4f}, x_new = {x_new:.4f}")
        x = x_new

mirror_descent_example()

# Proximal Methods (illustrative code)
def proximal_method_example():
    """Illustrative proximal gradient step for f(x) = |x| + 0.5*(x-2)^2."""
    def soft_thresholding(x, lam):
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    x = 3.0
    alpha = 0.2
    lam = 0.5
    for i in range(5):
        grad = x - 2
        x_temp = x - alpha * grad
        x_new = soft_thresholding(x_temp, alpha * lam)
        print(f"Iter {i}: x = {x:.4f}, grad = {grad:.4f}, x_new = {x_new:.4f}")
        x = x_new

proximal_method_example()

