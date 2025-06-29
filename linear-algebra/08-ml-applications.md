# Applications in Machine Learning

[![Chapter](https://img.shields.io/badge/Chapter-8-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-ML_Applications-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Linear algebra is the mathematical foundation of machine learning, providing the theoretical framework and computational tools that enable modern AI systems. This chapter explores how fundamental linear algebra concepts are applied in various machine learning algorithms, from simple linear regression to complex neural networks and deep learning systems.

**Mathematical Foundation:**
Machine learning algorithms can be viewed as optimization problems in high-dimensional vector spaces:
- **Feature Space**: Data points are vectors in ℝⁿ
- **Parameter Space**: Model parameters are vectors in ℝᵖ
- **Loss Functions**: Scalar functions mapping parameter vectors to real numbers
- **Gradients**: Vector derivatives indicating direction of steepest descent

**Key Linear Algebra Concepts in ML:**
1. **Vector Operations**: Dot products, norms, and projections for similarity and distance
2. **Matrix Operations**: Transformations, decompositions, and eigenvalue problems
3. **Optimization**: Gradient descent, Hessian matrices, and convex optimization
4. **Dimensionality Reduction**: Principal component analysis and matrix factorizations
5. **Kernel Methods**: Inner products and feature space transformations

**Geometric Interpretation:**
Machine learning can be understood geometrically:
- **Linear Models**: Find hyperplanes that best separate or fit data
- **Nonlinear Models**: Transform data into higher-dimensional spaces where linear separation is possible
- **Clustering**: Find centroids that minimize distances to data points
- **Dimensionality Reduction**: Project data onto lower-dimensional subspaces while preserving structure

## 1. Linear Regression

Linear regression is the foundation of supervised learning, modeling the relationship between features and target as a linear combination with additive noise.

### Mathematical Foundation

**Model Definition:**
For input features x ∈ ℝⁿ and target y ∈ ℝ, the linear regression model is:
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε = xᵀβ + ε

where:
- β = [β₀, β₁, ..., βₙ]ᵀ is the parameter vector (including bias term)
- x = [1, x₁, x₂, ..., xₙ]ᵀ is the feature vector (with bias term)
- ε ~ N(0, σ²) is the noise term

**Matrix Formulation:**
For a dataset with m samples, we have:
Y = Xβ + ε

where:
- Y ∈ ℝᵐ is the target vector
- X ∈ ℝᵐˣ⁽ⁿ⁺¹⁾ is the design matrix (with bias column)
- β ∈ ℝⁿ⁺¹ is the parameter vector
- ε ∈ ℝᵐ is the noise vector

**Objective Function:**
The least squares objective is:
min ||Xβ - Y||₂² = min Σᵢ₌₁ᵐ (xᵢᵀβ - yᵢ)²

**Geometric Interpretation:**
Linear regression finds the projection of Y onto the column space of X, minimizing the orthogonal distance between Y and the subspace spanned by X's columns.

### Normal Equation Solution

**Mathematical Derivation:**
The normal equation is derived by setting the gradient of the objective function to zero:

∇β(||Xβ - Y||₂²) = 2Xᵀ(Xβ - Y) = 0

Solving for β:
XᵀXβ = XᵀY
β = (XᵀX)⁻¹XᵀY

**Properties:**
1. **Uniqueness**: Solution is unique if X has full column rank
2. **Optimality**: Global minimum of the convex objective function
3. **Computational Cost**: O(n³) for matrix inversion, O(n²) for matrix multiplication
4. **Numerical Stability**: Can be ill-conditioned for nearly singular XᵀX

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def linear_regression_comprehensive(X, y, method='normal_equation', regularization=None, lambda_reg=1.0):
    """
    Comprehensive linear regression implementation with multiple solution methods
    
    Mathematical approaches:
    1. Normal equation: β = (X^T X)^(-1) X^T y
    2. QR decomposition: More numerically stable
    3. SVD decomposition: Handles rank-deficient cases
    4. Gradient descent: Iterative optimization
    
    Parameters:
    X: numpy array - feature matrix (samples × features)
    y: numpy array - target vector
    method: str - solution method
    regularization: str - 'ridge' or 'lasso' regularization
    lambda_reg: float - regularization strength
    
    Returns:
    dict - comprehensive results including parameters, predictions, and analysis
    """
    n_samples, n_features = X.shape
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # Check for multicollinearity
    condition_number = np.linalg.cond(X_with_bias)
    rank = np.linalg.matrix_rank(X_with_bias)
    well_conditioned = condition_number < 1000
    
    results = {
        'method': method,
        'regularization': regularization,
        'lambda_reg': lambda_reg,
        'condition_number': condition_number,
        'rank': rank,
        'well_conditioned': well_conditioned,
        'n_samples': n_samples,
        'n_features': n_features
    }
    
    if method == 'normal_equation':
        if regularization == 'ridge':
            # Ridge regression: β = (X^T X + λI)^(-1) X^T y
            I = np.eye(X_with_bias.shape[1])
            I[0, 0] = 0  # Don't regularize bias term
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias + lambda_reg * I) @ X_with_bias.T @ y
        else:
            # Standard normal equation
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            
    elif method == 'qr':
        # QR decomposition: more numerically stable
        Q, R = np.linalg.qr(X_with_bias)
        beta = np.linalg.solve(R, Q.T @ y)
        
    elif method == 'svd':
        # SVD decomposition: handles rank-deficient cases
        U, S, Vt = np.linalg.svd(X_with_bias, full_matrices=False)
        # Use pseudo-inverse for numerical stability
        S_inv = np.diag(1.0 / np.where(S > 1e-10, S, 0))
        beta = Vt.T @ S_inv @ U.T @ y
        
    elif method == 'gradient_descent':
        # Gradient descent implementation
        beta = gradient_descent_linear_regression(X_with_bias, y, learning_rate=0.01, epochs=1000)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute predictions and metrics
    y_pred = X_with_bias @ beta
    residuals = y - y_pred
    
    # Statistical analysis
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Residual analysis
    residual_std = np.std(residuals)
    residual_mean = np.mean(residuals)
    
    # Parameter analysis
    parameter_std = np.std(beta[1:])  # Exclude bias term
    parameter_norm = np.linalg.norm(beta[1:])
    
    # Confidence intervals (simplified)
    if method == 'normal_equation' and regularization is None:
        # Compute standard errors
        residual_variance = np.sum(residuals**2) / (n_samples - n_features - 1)
        XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias)
        standard_errors = np.sqrt(np.diag(XtX_inv) * residual_variance)
        
        # 95% confidence intervals
        confidence_intervals = np.column_stack([
            beta - 1.96 * standard_errors,
            beta + 1.96 * standard_errors
        ])
    else:
        confidence_intervals = None
    
    results.update({
        'beta': beta,
        'y_pred': y_pred,
        'residuals': residuals,
        'mse': mse,
        'r2': r2,
        'rmse': rmse,
        'residual_std': residual_std,
        'residual_mean': residual_mean,
        'parameter_std': parameter_std,
        'parameter_norm': parameter_norm,
        'confidence_intervals': confidence_intervals
    })
    
    return results

def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    """
    Gradient descent for linear regression
    
    Mathematical approach:
    β^(t+1) = β^(t) - α ∇β J(β)
    where J(β) = (1/2m) ||Xβ - y||²
    and ∇β J(β) = (1/m) X^T (Xβ - y)
    
    Parameters:
    X: numpy array - design matrix (including bias)
    y: numpy array - target vector
    learning_rate: float - learning rate
    epochs: int - maximum number of iterations
    tolerance: float - convergence tolerance
    
    Returns:
    numpy array - optimized parameter vector
    """
    n_samples = X.shape[0]
    beta = np.zeros(X.shape[1])
    costs = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = X @ beta
        
        # Compute cost
        cost = np.mean((predictions - y) ** 2) / 2
        costs.append(cost)
        
        # Compute gradient
        gradients = (1/n_samples) * X.T @ (predictions - y)
        
        # Update parameters
        beta_new = beta - learning_rate * gradients
        
        # Check convergence
        if np.linalg.norm(beta_new - beta) < tolerance:
            break
            
        beta = beta_new
    
    return beta

def analyze_linear_regression_results(results, X, y):
    """
    Comprehensive analysis of linear regression results
    
    Parameters:
    results: dict - results from linear_regression_comprehensive
    X: numpy array - original feature matrix
    y: numpy array - target vector
    
    Returns:
    dict - comprehensive analysis
    """
    analysis = {}
    
    # Model performance
    analysis['performance'] = {
        'mse': results['mse'],
        'r2': results['r2'],
        'rmse': results['rmse'],
        'explained_variance': results['r2']
    }
    
    # Residual analysis
    residuals = results['residuals']
    analysis['residuals'] = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**3),
        'kurtosis': np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**4) - 3,
        'normality_test': np.allclose(np.mean(residuals), 0, atol=1e-10)
    }
    
    # Parameter analysis
    beta = results['beta']
    analysis['parameters'] = {
        'bias': beta[0],
        'feature_weights': beta[1:],
        'weight_magnitudes': np.abs(beta[1:]),
        'largest_weight': np.max(np.abs(beta[1:])),
        'smallest_weight': np.min(np.abs(beta[1:])),
        'weight_std': np.std(beta[1:])
    }
    
    # Multicollinearity analysis
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        high_corr_pairs = []
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        analysis['multicollinearity'] = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'vif_scores': compute_vif_scores(X)
        }
    
    # Numerical stability
    analysis['numerical_stability'] = {
        'condition_number': results['condition_number'],
        'well_conditioned': results['well_conditioned'],
        'rank': results['rank'],
        'method_stability': results['method'] in ['qr', 'svd']
    }
    
    return analysis

def compute_vif_scores(X):
    """
    Compute Variance Inflation Factor (VIF) scores for multicollinearity detection
    
    VIF for feature i is computed as:
    VIF_i = 1 / (1 - R²_i)
    where R²_i is the coefficient of determination when feature i is regressed on all other features
    """
    n_features = X.shape[1]
    vif_scores = np.zeros(n_features)
    
    for i in range(n_features):
        # Regress feature i on all other features
        X_others = np.delete(X, i, axis=1)
        y_feature = X[:, i]
        
        # Fit regression
        beta = np.linalg.lstsq(X_others, y_feature, rcond=None)[0]
        y_pred = X_others @ beta
        
        # Compute R²
        ss_res = np.sum((y_feature - y_pred) ** 2)
        ss_tot = np.sum((y_feature - np.mean(y_feature)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Compute VIF
        vif_scores[i] = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
    
    return vif_scores

def compare_linear_regression_methods(X, y, methods=['normal_equation', 'qr', 'svd', 'gradient_descent']):
    """
    Compare different linear regression solution methods
    
    Parameters:
    X: numpy array - feature matrix
    y: numpy array - target vector
    methods: list - methods to compare
    
    Returns:
    dict - comparison results
    """
    comparison = {}
    
    for method in methods:
        try:
            results = linear_regression_comprehensive(X, y, method=method)
            analysis = analyze_linear_regression_results(results, X, y)
            
            comparison[method] = {
                'success': True,
                'results': results,
                'analysis': analysis
            }
            
        except Exception as e:
            comparison[method] = {
                'success': False,
                'error': str(e)
            }
    
    return comparison

# Example: Comprehensive linear regression analysis
print("=== Comprehensive Linear Regression Analysis ===")

# Generate synthetic data with known structure
np.random.seed(42)
n_samples, n_features = 200, 5

# Create features with some correlation
X = np.random.randn(n_samples, n_features)
X[:, 2] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)  # Correlated features
X[:, 4] = 0.5 * X[:, 1] + 0.5 * np.random.randn(n_samples)  # Another correlation

# True parameters
true_beta = np.array([2.5, -1.0, 0.8, 1.2, -0.5])
true_bias = 3.0

# Generate target with noise
y = X @ true_beta + true_bias + np.random.normal(0, 0.5, n_samples)

print(f"Data shape: {X.shape}")
print(f"True parameters: bias={true_bias}, weights={true_beta}")

# Compare different methods
methods = ['normal_equation', 'qr', 'svd', 'gradient_descent']
comparison = compare_linear_regression_methods(X, y, methods)

print(f"\nMethod Comparison:")
for method, result in comparison.items():
    if result['success']:
        results = result['results']
        analysis = result['analysis']
        
        print(f"\n{method.upper()}:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  R²: {results['r2']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        print(f"  Condition number: {results['condition_number']:.2e}")
        print(f"  Well-conditioned: {results['well_conditioned']}")
        print(f"  Residual mean: {analysis['residuals']['mean']:.6f}")
        print(f"  Residual std: {analysis['residuals']['std']:.6f}")
        
        # Compare with true parameters
        estimated_bias = results['beta'][0]
        estimated_weights = results['beta'][1:]
        bias_error = abs(estimated_bias - true_bias)
        weight_error = np.linalg.norm(estimated_weights - true_beta)
        
        print(f"  Bias error: {bias_error:.6f}")
        print(f"  Weight error: {weight_error:.6f}")
        
    else:
        print(f"\n{method.upper()}: Failed - {result['error']}")

# Test with regularization
print(f"\n=== Regularization Analysis ===")

# Test ridge regression
ridge_results = linear_regression_comprehensive(X, y, method='normal_equation', regularization='ridge', lambda_reg=0.1)
ridge_analysis = analyze_linear_regression_results(ridge_results, X, y)

print(f"Ridge Regression (λ=0.1):")
print(f"  MSE: {ridge_results['mse']:.6f}")
print(f"  R²: {ridge_results['r2']:.6f}")
print(f"  Parameter norm: {ridge_results['parameter_norm']:.6f}")

# Compare parameter magnitudes
standard_weights = comparison['normal_equation']['results']['beta'][1:]
ridge_weights = ridge_results['beta'][1:]

print(f"  Standard weights: {np.linalg.norm(standard_weights):.6f}")
print(f"  Ridge weights: {np.linalg.norm(ridge_weights):.6f}")
print(f"  Shrinkage: {np.linalg.norm(standard_weights) - np.linalg.norm(ridge_weights):.6f}")

# Multicollinearity analysis
print(f"\n=== Multicollinearity Analysis ===")

analysis = comparison['normal_equation']['analysis']
if 'multicollinearity' in analysis:
    vif_scores = analysis['multicollinearity']['vif_scores']
    high_corr_pairs = analysis['multicollinearity']['high_correlation_pairs']
    
    print(f"VIF scores:")
    for i, vif in enumerate(vif_scores):
        print(f"  Feature {i+1}: {vif:.2f}")
    
    print(f"High correlation pairs:")
    for pair in high_corr_pairs:
        print(f"  Features {pair[0]+1} and {pair[1]+1}: {pair[2]:.3f}")

# Visualize results
print(f"\n=== Visualization ===")

# Plot predictions vs actual
plt.figure(figsize=(15, 5))

# Method 1: Predictions vs Actual
plt.subplot(1, 3, 1)
for method in ['normal_equation', 'qr', 'svd']:
    if comparison[method]['success']:
        y_pred = comparison[method]['results']['y_pred']
        plt.scatter(y, y_pred, alpha=0.6, label=method, s=20)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual')
plt.legend()

# Method 2: Residuals
plt.subplot(1, 3, 2)
residuals = comparison['normal_equation']['results']['residuals']
plt.scatter(comparison['normal_equation']['results']['y_pred'], residuals, alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Method 3: Parameter comparison
plt.subplot(1, 3, 3)
x_pos = np.arange(len(true_beta))
plt.bar(x_pos - 0.2, true_beta, width=0.4, label='True', alpha=0.7)
plt.bar(x_pos + 0.2, comparison['normal_equation']['results']['beta'][1:], width=0.4, label='Estimated', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Weight')
plt.title('Parameter Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# Test with ill-conditioned data
print(f"\n=== Ill-conditioned Data Test ===")

# Create nearly singular design matrix
X_ill = X.copy()
X_ill[:, 3] = X_ill[:, 0] + 1e-8 * np.random.randn(n_samples)  # Nearly dependent

ill_comparison = compare_linear_regression_methods(X_ill, y, methods)

print(f"Ill-conditioned data results:")
for method, result in ill_comparison.items():
    if result['success']:
        results = result['results']
        print(f"  {method}: MSE={results['mse']:.6f}, Condition number={results['condition_number']:.2e}")
    else:
        print(f"  {method}: Failed")
```

### Ridge Regression (L2 Regularization)

**Mathematical Foundation:**
Ridge regression adds L2 regularization to prevent overfitting and handle multicollinearity:

**Objective Function:**
min ||Xβ - Y||₂² + λ||β||₂²

where λ > 0 is the regularization parameter.

**Solution:**
β = (XᵀX + λI)⁻¹XᵀY

**Geometric Interpretation:**
Ridge regression shrinks parameter estimates toward zero, trading bias for variance reduction.

**Key Properties:**
1. **Bias-Variance Trade-off**: Increases bias, decreases variance
2. **Multicollinearity**: Handles correlated features effectively
3. **Numerical Stability**: Improves conditioning of XᵀX
4. **Shrinkage**: All parameters are shrunk by the same factor

```python
def ridge_regression_comprehensive(X, y, lambda_reg=1.0, method='normal_equation'):
    """
    Comprehensive ridge regression implementation
    
    Mathematical approach:
    β = (X^T X + λI)^(-1) X^T y
    
    Parameters:
    X: numpy array - feature matrix
    y: numpy array - target vector
    lambda_reg: float - regularization parameter
    method: str - solution method
    
    Returns:
    dict - comprehensive results
    """
    n_samples, n_features = X.shape
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    if method == 'normal_equation':
        # Standard ridge solution
        I = np.eye(X_with_bias.shape[1])
        I[0, 0] = 0  # Don't regularize bias term
        
        beta = np.linalg.inv(X_with_bias.T @ X_with_bias + lambda_reg * I) @ X_with_bias.T @ y
        
    elif method == 'svd':
        # SVD-based solution for numerical stability
        U, S, Vt = np.linalg.svd(X_with_bias, full_matrices=False)
        
        # Ridge regularization in SVD space
        S_ridge = S / (S**2 + lambda_reg)
        beta = Vt.T @ np.diag(S_ridge) @ U.T @ y
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute predictions and metrics
    y_pred = X_with_bias @ beta
    residuals = y - y_pred
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Effective degrees of freedom
    if method == 'normal_equation':
        XtX = X_with_bias.T @ X_with_bias
        I_reg = np.eye(X_with_bias.shape[1])
        I_reg[0, 0] = 0
        effective_df = np.trace(X_with_bias @ np.linalg.inv(XtX + lambda_reg * I_reg) @ X_with_bias.T)
    else:
        effective_df = np.sum(S**2 / (S**2 + lambda_reg))
    
    return {
        'beta': beta,
        'y_pred': y_pred,
        'residuals': residuals,
        'mse': mse,
        'r2': r2,
        'lambda_reg': lambda_reg,
        'effective_df': effective_df,
        'parameter_norm': np.linalg.norm(beta[1:])
    }

def ridge_regression_path(X, y, lambda_range=np.logspace(-3, 3, 50)):
    """
    Compute ridge regression for a range of regularization parameters
    
    Parameters:
    X: numpy array - feature matrix
    y: numpy array - target vector
    lambda_range: array - range of lambda values
    
    Returns:
    dict - results for each lambda value
    """
    path_results = []
    
    for lambda_val in lambda_range:
        results = ridge_regression_comprehensive(X, y, lambda_val)
        path_results.append({
            'lambda': lambda_val,
            'beta': results['beta'],
            'mse': results['mse'],
            'r2': results['r2'],
            'parameter_norm': results['parameter_norm'],
            'effective_df': results['effective_df']
        })
    
    return path_results

# Example: Ridge regression analysis
print("\n=== Ridge Regression Analysis ===")

# Test different lambda values
lambda_values = np.logspace(-3, 3, 20)
ridge_path = ridge_regression_path(X, y, lambda_values)

# Extract results for plotting
lambdas = [r['lambda'] for r in ridge_path]
mses = [r['mse'] for r in ridge_path]
r2s = [r['r2'] for r in ridge_path]
param_norms = [r['parameter_norm'] for r in ridge_path]
effective_dfs = [r['effective_df'] for r in ridge_path]

# Plot ridge path
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.semilogx(lambdas, mses, 'b-', linewidth=2)
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Path: MSE vs λ')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.semilogx(lambdas, r2s, 'r-', linewidth=2)
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('R² Score')
plt.title('Ridge Path: R² vs λ')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.semilogx(lambdas, param_norms, 'g-', linewidth=2)
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Parameter Norm')
plt.title('Ridge Path: Parameter Norm vs λ')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare with standard regression
print(f"Standard regression parameter norm: {np.linalg.norm(comparison['normal_equation']['results']['beta'][1:]):.6f}")
print(f"Ridge regression (λ=1.0) parameter norm: {ridge_regression_comprehensive(X, y, 1.0)['parameter_norm']:.6f}")
```

## 2. Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in data by computing eigenvectors of the covariance matrix.

### Manual PCA Implementation
```python
def manual_pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    sorted_indices = np.argsort(eigenvals)[::-1]
    eigenvals_sorted = eigenvals[sorted_indices]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    
    # Project data
    X_pca = X_centered @ eigenvecs_sorted[:, :n_components]
    
    return X_pca, eigenvecs_sorted[:, :n_components], eigenvals_sorted

# Generate data with known structure
np.random.seed(42)
n_samples = 1000
# Create correlated features
X = np.random.randn(n_samples, 3)
X[:, 2] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n_samples)

# Apply PCA
X_pca, components, eigenvals = manual_pca(X, n_components=2)

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data (Features 1 vs 2)')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 2], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 3')
plt.title('Original Data (Features 1 vs 3)')

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Data')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", eigenvals[:2] / np.sum(eigenvals))
```

### PCA for Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
for i in range(3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f'Class {i}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs Components')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 3. Recommender Systems

### Matrix Factorization
```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    R: rating matrix
    P: user matrix
    Q: item matrix
    K: number of latent factors
    """
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:], Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T

# Example: Simple rating matrix
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

print("Original ratings:")
print(R)
print("\nPredicted ratings:")
print(nR)
```

## 4. Neural Networks

### Forward Pass
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        
        # Backpropagation
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train neural network
nn = SimpleNeuralNetwork(2, 4, 1)

for epoch in range(10000):
    # Forward pass
    output = nn.forward(X)
    
    # Backward pass
    nn.backward(X, y, learning_rate=0.1)
    
    if epoch % 1000 == 0:
        loss = np.mean((output - y) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
predictions = nn.forward(X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
```

## 5. Support Vector Machines (SVM)

### Linear SVM with Gradient Descent
```python
def linear_svm_gradient_descent(X, y, learning_rate=0.01, epochs=1000, lambda_reg=0.1):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        for i in range(n_samples):
            # Hinge loss gradient
            if y[i] * (X[i] @ w + b) < 1:
                dw = -y[i] * X[i] + lambda_reg * w
                db = -y[i]
            else:
                dw = lambda_reg * w
                db = 0
            
            w -= learning_rate * dw
            b -= learning_rate * db
    
    return w, b

# Generate linearly separable data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Train SVM
w, b = linear_svm_gradient_descent(X, y)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1')

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = np.sign(xx * w[0] + yy * w[1] + b)
plt.contour(xx, yy, Z, levels=[0], colors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM Decision Boundary')
plt.legend()

plt.subplot(1, 2, 2)
# Show margin
margin_points = np.array([[-2, 2], [2, -2]])
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.6)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', alpha=0.6)
plt.plot(margin_points[:, 0], margin_points[:, 1], 'k--', alpha=0.5, label='Margin')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Margin')
plt.legend()

plt.tight_layout()
plt.show()
```

## 6. Clustering with Linear Algebra

### K-Means Clustering
```python
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Generate clustered data
np.random.seed(42)
n_samples = 300
centers = np.array([[0, 0], [4, 4], [8, 0]])
X = np.vstack([
    np.random.randn(n_samples//3, 2) + centers[0],
    np.random.randn(n_samples//3, 2) + centers[1],
    np.random.randn(n_samples//3, 2) + centers[2]
])

# Apply K-means
labels, centroids = kmeans(X, k=3)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Original Data')

plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], alpha=0.6, label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering')
plt.legend()

plt.tight_layout()
plt.show()
```

## 7. Optimization in Machine Learning

### Gradient Descent for Linear Regression
```python
def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(X_with_bias.shape[1])
    
    costs = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = X_with_bias @ theta
        
        # Compute cost
        cost = np.mean((predictions - y) ** 2)
        costs.append(cost)
        
        # Compute gradients
        gradients = (2/n_samples) * X_with_bias.T @ (predictions - y)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta, costs

# Test gradient descent
theta_gd, costs = gradient_descent_linear_regression(X, y)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], y, alpha=0.6, label='Data')
X_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
y_line = np.column_stack([np.ones(100), X_line]) @ theta_gd
plt.plot(X_line, y_line, 'r-', label='Gradient Descent')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.legend()

plt.tight_layout()
plt.show()
```

## Exercises

1. **Linear Regression**: Implement linear regression using both normal equations and gradient descent.
2. **PCA Implementation**: Implement PCA from scratch and compare with sklearn.
3. **Neural Network**: Build a neural network to solve the XOR problem.
4. **SVM**: Implement linear SVM using gradient descent.
5. **Matrix Factorization**: Implement matrix factorization for a simple recommender system.

## Solutions

```python
# Exercise 1: Linear Regression
def linear_regression_normal_equations(X, y):
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples = X.shape[0]
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(X_with_bias.shape[1])
    
    for _ in range(epochs):
        predictions = X_with_bias @ theta
        gradients = (2/n_samples) * X_with_bias.T @ (predictions - y)
        theta -= learning_rate * gradients
    
    return theta

# Exercise 2: PCA Implementation
def pca_manual(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvals)[::-1]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    return X_centered @ eigenvecs_sorted[:, :n_components]

# Exercise 3: Neural Network for XOR
def train_xor_network():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    nn = SimpleNeuralNetwork(2, 4, 1)
    for _ in range(10000):
        output = nn.forward(X)
        nn.backward(X, y, learning_rate=0.1)
    
    return nn

# Exercise 4: Linear SVM
def linear_svm(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        for i in range(n_samples):
            if y[i] * (X[i] @ w + b) < 1:
                w -= learning_rate * (-y[i] * X[i] + 0.1 * w)
                b -= learning_rate * (-y[i])
            else:
                w -= learning_rate * (0.1 * w)
    
    return w, b

# Exercise 5: Matrix Factorization
def simple_matrix_factorization(R, k=2, epochs=1000, learning_rate=0.01):
    n_users, n_items = R.shape
    P = np.random.randn(n_users, k)
    Q = np.random.randn(n_items, k)
    
    for _ in range(epochs):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:
                    eij = R[i, j] - P[i, :] @ Q[j, :]
                    P[i, :] += learning_rate * eij * Q[j, :]
                    Q[j, :] += learning_rate * eij * P[i, :]
    
    return P, Q
```

## Key Takeaways

- Linear algebra is fundamental to all machine learning algorithms
- Matrix operations enable efficient computation of gradients and predictions
- Eigenvalue decomposition (PCA) is crucial for dimensionality reduction
- Matrix factorization enables collaborative filtering in recommender systems
- Neural networks rely heavily on matrix multiplication for forward and backward passes
- Optimization algorithms use linear algebra for parameter updates

## Next Chapter

In the next chapter, we'll explore numerical linear algebra, focusing on numerical stability, conditioning, and efficient algorithms for large-scale problems. 