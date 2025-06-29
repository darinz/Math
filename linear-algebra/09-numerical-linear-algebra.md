# Numerical Linear Algebra

[![Chapter](https://img.shields.io/badge/Chapter-9-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Numerical_Linear_Algebra-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Numerical linear algebra is the study of algorithms for performing linear algebra computations on computers with finite precision arithmetic. This field bridges the gap between theoretical linear algebra and practical computational methods, addressing issues of numerical stability, computational efficiency, and algorithmic complexity.

**Mathematical Foundation:**
Numerical linear algebra deals with the practical implementation of mathematical concepts:
- **Finite Precision**: Computers represent real numbers with limited precision (typically 64 bits for double precision)
- **Rounding Errors**: Accumulation of small errors in arithmetic operations
- **Conditioning**: Sensitivity of problems to small perturbations in input data
- **Stability**: Ability of algorithms to produce accurate results despite rounding errors

**Key Challenges:**
1. **Numerical Stability**: Ensuring algorithms don't amplify rounding errors
2. **Computational Complexity**: Balancing accuracy with computational cost
3. **Memory Efficiency**: Handling large-scale problems within memory constraints
4. **Parallelization**: Exploiting modern hardware for performance

**Geometric Interpretation:**
Numerical issues can be understood geometrically:
- **Ill-conditioned problems**: Small changes in input cause large changes in output
- **Stable algorithms**: Preserve geometric relationships despite numerical errors
- **Convergence**: Iterative methods approach solutions through geometric optimization

**Applications in Modern Computing:**
- **Machine Learning**: Large-scale matrix operations in neural networks
- **Scientific Computing**: Solving partial differential equations
- **Data Science**: Principal component analysis and dimensionality reduction
- **Computer Graphics**: 3D transformations and rendering
- **Signal Processing**: Filtering and spectral analysis

## 1. Numerical Stability and Conditioning

### Mathematical Foundation

**Condition Number:**
The condition number measures how sensitive a problem is to perturbations in the input data. For a matrix A, the condition number is:

κ(A) = ||A|| · ||A⁻¹||

where ||·|| is a matrix norm (typically the 2-norm).

**Error Analysis:**
For the linear system Ax = b, if we perturb b by δb, the solution changes by δx:

(A + δA)(x + δx) = b + δb

The relative error bound is:
||δx||/||x|| ≤ κ(A) · (||δA||/||A|| + ||δb||/||b||)

**Backward Error Analysis:**
Instead of asking "how accurate is the computed solution?", we ask "for what perturbed problem is our computed solution exact?"

**Stability Definitions:**
1. **Forward Stability**: Computed solution is close to exact solution
2. **Backward Stability**: Computed solution is exact for slightly perturbed problem
3. **Mixed Stability**: Combination of forward and backward stability

### Condition Number Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, eigsh
import seaborn as sns

def comprehensive_condition_analysis(A, name="Matrix"):
    """
    Comprehensive condition number analysis
    
    Mathematical approach:
    κ(A) = ||A||₂ · ||A⁻¹||₂ = σ_max(A) / σ_min(A)
    
    Parameters:
    A: numpy array - matrix to analyze
    name: str - name for display
    
    Returns:
    dict - comprehensive analysis results
    """
    analysis = {}
    
    # Basic condition number
    try:
        cond_2 = np.linalg.cond(A, p=2)
        cond_1 = np.linalg.cond(A, p=1)
        cond_inf = np.linalg.cond(A, p=np.inf)
        analysis['condition_numbers'] = {
            '2_norm': cond_2,
            '1_norm': cond_1,
            'inf_norm': cond_inf
        }
    except np.linalg.LinAlgError:
        analysis['condition_numbers'] = {'2_norm': np.inf, '1_norm': np.inf, 'inf_norm': np.inf}
    
    # SVD-based analysis
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        analysis['svd'] = {
            'singular_values': S,
            'max_singular': S[0],
            'min_singular': S[-1],
            'condition_from_svd': S[0] / S[-1] if S[-1] > 0 else np.inf,
            'rank': np.sum(S > 1e-10 * S[0])
        }
    except np.linalg.LinAlgError:
        analysis['svd'] = None
    
    # Eigenvalue analysis (for square matrices)
    if A.shape[0] == A.shape[1]:
        try:
            eigenvals = np.linalg.eigvals(A)
            analysis['eigenvalues'] = {
                'eigenvalues': eigenvals,
                'max_magnitude': np.max(np.abs(eigenvals)),
                'min_magnitude': np.min(np.abs(eigenvals)),
                'condition_from_eigenvalues': np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals)) if np.min(np.abs(eigenvals)) > 0 else np.inf
            }
        except np.linalg.LinAlgError:
            analysis['eigenvalues'] = None
    
    # Numerical rank
    if analysis['svd']:
        analysis['numerical_rank'] = analysis['svd']['rank']
    
    # Stability assessment
    cond_2 = analysis['condition_numbers']['2_norm']
    if cond_2 < 100:
        stability = "Excellent"
    elif cond_2 < 1000:
        stability = "Good"
    elif cond_2 < 1e6:
        stability = "Fair"
    elif cond_2 < 1e12:
        stability = "Poor"
    else:
        stability = "Very Poor"
    
    analysis['stability_assessment'] = stability
    
    # Print results
    print(f"\n=== {name} Condition Analysis ===")
    print(f"Condition number (2-norm): {cond_2:.2e}")
    print(f"Stability: {stability}")
    
    if analysis['svd']:
        print(f"Singular values range: {S[0]:.2e} to {S[-1]:.2e}")
        print(f"Numerical rank: {analysis['numerical_rank']}")
    
    if analysis['eigenvalues']:
        eigenvals = analysis['eigenvalues']['eigenvalues']
        print(f"Eigenvalues range: {np.min(np.abs(eigenvals)):.2e} to {np.max(np.abs(eigenvals)):.2e}")
    
    return analysis

def perturbation_analysis(A, b, perturbation_sizes=np.logspace(-15, -1, 15)):
    """
    Analyze sensitivity to perturbations
    
    Mathematical approach:
    For Ax = b, perturb b by δb and solve A(x + δx) = b + δb
    Measure ||δx||/||x|| vs ||δb||/||b||
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side
    perturbation_sizes: array - sizes of perturbations to test
    
    Returns:
    dict - perturbation analysis results
    """
    # Exact solution
    x_exact = np.linalg.solve(A, b)
    
    # Test different perturbation sizes
    relative_errors = []
    amplification_factors = []
    
    for eps in perturbation_sizes:
        # Perturb b
        b_perturbed = b + eps * np.random.randn(*b.shape)
        
        # Solve perturbed system
        try:
            x_perturbed = np.linalg.solve(A, b_perturbed)
            
            # Compute relative errors
            rel_error_b = np.linalg.norm(b_perturbed - b) / np.linalg.norm(b)
            rel_error_x = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)
            
            relative_errors.append(rel_error_x)
            amplification_factors.append(rel_error_x / rel_error_b if rel_error_b > 0 else np.inf)
            
        except np.linalg.LinAlgError:
            relative_errors.append(np.inf)
            amplification_factors.append(np.inf)
    
    # Theoretical bound
    cond_A = np.linalg.cond(A)
    theoretical_bound = cond_A * perturbation_sizes
    
    analysis = {
        'perturbation_sizes': perturbation_sizes,
        'relative_errors': relative_errors,
        'amplification_factors': amplification_factors,
        'theoretical_bound': theoretical_bound,
        'condition_number': cond_A,
        'max_amplification': np.max(amplification_factors) if not np.any(np.isinf(amplification_factors)) else np.inf
    }
    
    return analysis

def backward_error_analysis(A, b, x_computed):
    """
    Perform backward error analysis
    
    Mathematical approach:
    Find smallest ||δA|| and ||δb|| such that (A + δA)x_computed = b + δb
    
    Parameters:
    A: numpy array - original coefficient matrix
    b: numpy array - original right-hand side
    x_computed: numpy array - computed solution
    
    Returns:
    dict - backward error analysis
    """
    # Compute residual
    residual = b - A @ x_computed
    
    # Backward error for right-hand side
    backward_error_b = np.linalg.norm(residual) / np.linalg.norm(b)
    
    # Backward error for matrix (simplified)
    # In practice, this requires more sophisticated analysis
    backward_error_A = backward_error_b  # Simplified approximation
    
    # Forward error bound
    cond_A = np.linalg.cond(A)
    forward_error_bound = cond_A * backward_error_b
    
    # Actual forward error
    x_exact = np.linalg.solve(A, b)
    actual_forward_error = np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)
    
    analysis = {
        'backward_error_b': backward_error_b,
        'backward_error_A': backward_error_A,
        'forward_error_bound': forward_error_bound,
        'actual_forward_error': actual_forward_error,
        'residual_norm': np.linalg.norm(residual),
        'condition_number': cond_A
    }
    
    return analysis

def demonstrate_numerical_instability():
    """
    Comprehensive demonstration of numerical instability
    
    Mathematical examples:
    1. Hilbert matrix: Classic ill-conditioned matrix
    2. Vandermonde matrix: Polynomial interpolation matrix
    3. Random matrices: Statistical analysis
    4. Structured matrices: Toeplitz, circulant, etc.
    """
    
    # 1. Hilbert matrix (classic example)
    def hilbert_matrix(n):
        """Create n×n Hilbert matrix H[i,j] = 1/(i+j+1)"""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        return H
    
    # 2. Vandermonde matrix
    def vandermonde_matrix(x, n):
        """Create Vandermonde matrix V[i,j] = x[i]^j"""
        V = np.zeros((len(x), n))
        for i in range(len(x)):
            for j in range(n):
                V[i, j] = x[i]**j
        return V
    
    # 3. Toeplitz matrix
    def toeplitz_matrix(c, r):
        """Create Toeplitz matrix with first column c and first row r"""
        n = len(c)
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i >= j:
                    T[i, j] = c[i - j]
                else:
                    T[i, j] = r[j - i]
        return T
    
    # Test different matrix types
    matrix_types = {
        'Identity': np.eye(5),
        'Random': np.random.randn(5, 5),
        'Hilbert (5×5)': hilbert_matrix(5),
        'Hilbert (10×10)': hilbert_matrix(10),
        'Vandermonde': vandermonde_matrix(np.linspace(0, 1, 5), 5),
        'Toeplitz': toeplitz_matrix([1, 0.5, 0.25, 0.125, 0.0625], [1, 0.5, 0.25, 0.125, 0.0625])
    }
    
    # Analyze each matrix
    analyses = {}
    for name, A in matrix_types.items():
        analyses[name] = comprehensive_condition_analysis(A, name)
    
    # Test perturbation sensitivity
    print(f"\n=== Perturbation Sensitivity Analysis ===")
    
    # Use Hilbert matrix as example
    H = hilbert_matrix(10)
    b = np.ones(10)
    x_exact = np.linalg.solve(H, b)
    
    perturbation_analysis_result = perturbation_analysis(H, b)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Condition numbers comparison
    plt.subplot(2, 3, 1)
    names = list(analyses.keys())
    cond_numbers = [analyses[name]['condition_numbers']['2_norm'] for name in names]
    
    plt.bar(range(len(names)), cond_numbers)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Condition Number')
    plt.title('Condition Numbers Comparison')
    plt.yscale('log')
    
    # Plot 2: Perturbation analysis
    plt.subplot(2, 3, 2)
    eps = perturbation_analysis_result['perturbation_sizes']
    rel_errors = perturbation_analysis_result['relative_errors']
    theoretical = perturbation_analysis_result['theoretical_bound']
    
    plt.loglog(eps, rel_errors, 'bo-', label='Actual Error')
    plt.loglog(eps, theoretical, 'r--', label='Theoretical Bound')
    plt.xlabel('Perturbation Size')
    plt.ylabel('Relative Error in Solution')
    plt.title('Perturbation Sensitivity')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Amplification factors
    plt.subplot(2, 3, 3)
    amplification = perturbation_analysis_result['amplification_factors']
    plt.semilogx(eps, amplification, 'go-')
    plt.axhline(y=perturbation_analysis_result['condition_number'], color='r', linestyle='--', label='Condition Number')
    plt.xlabel('Perturbation Size')
    plt.ylabel('Error Amplification Factor')
    plt.title('Error Amplification')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Singular values for Hilbert matrix
    plt.subplot(2, 3, 4)
    S = analyses['Hilbert (10×10)']['svd']['singular_values']
    plt.semilogy(range(1, len(S)+1), S, 'mo-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Hilbert Matrix Singular Values')
    plt.grid(True)
    
    # Plot 5: Eigenvalue distribution for random matrix
    plt.subplot(2, 3, 5)
    eigenvals = analyses['Random']['eigenvalues']['eigenvalues']
    plt.scatter(eigenvals.real, eigenvals.imag, alpha=0.6)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Random Matrix Eigenvalues')
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 6: Stability assessment
    plt.subplot(2, 3, 6)
    stability_levels = ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
    stability_counts = {level: 0 for level in stability_levels}
    
    for name, analysis in analyses.items():
        stability = analysis['stability_assessment']
        stability_counts[stability] += 1
    
    plt.bar(stability_levels, [stability_counts[level] for level in stability_levels])
    plt.ylabel('Number of Matrices')
    plt.title('Stability Assessment')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Test solving linear systems with different methods
    print(f"\n=== Linear System Solution Comparison ===")
    
    # Test with Hilbert matrix
    H = hilbert_matrix(8)
    b = np.ones(8)
    x_true = np.linalg.solve(H, b)
    
    # Different solution methods
    methods = {
        'Direct LU': lambda A, b: np.linalg.solve(A, b),
        'QR Decomposition': lambda A, b: np.linalg.solve(A.T @ A, A.T @ b),
        'SVD Pseudo-inverse': lambda A, b: np.linalg.pinv(A) @ b,
        'Tikhonov Regularization': lambda A, b: np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ b)
    }
    
    results = {}
    for method_name, method_func in methods.items():
        try:
            x_computed = method_func(H, b)
            error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
            backward_analysis = backward_error_analysis(H, b, x_computed)
            
            results[method_name] = {
                'error': error,
                'backward_error': backward_analysis['backward_error_b'],
                'residual': backward_analysis['residual_norm']
            }
            
            print(f"{method_name}:")
            print(f"  Relative error: {error:.2e}")
            print(f"  Backward error: {backward_analysis['backward_error_b']:.2e}")
            print(f"  Residual norm: {backward_analysis['residual_norm']:.2e}")
            
        except Exception as e:
            print(f"{method_name}: Failed - {e}")
    
    return analyses, perturbation_analysis_result, results

# Run comprehensive numerical stability demonstration
print("=== Comprehensive Numerical Stability Analysis ===")
analyses, perturbation_result, solution_results = demonstrate_numerical_instability()

# Additional analysis: Growth of condition number with matrix size
print(f"\n=== Condition Number Growth Analysis ===")

sizes = [5, 10, 15, 20, 25]
hilbert_conditions = []
vandermonde_conditions = []

for n in sizes:
    # Hilbert matrix
    H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
    hilbert_conditions.append(np.linalg.cond(H))
    
    # Vandermonde matrix
    x = np.linspace(0, 1, n)
    V = np.array([[x[i]**j for j in range(n)] for i in range(n)])
    vandermonde_conditions.append(np.linalg.cond(V))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.semilogy(sizes, hilbert_conditions, 'bo-', label='Hilbert Matrix')
plt.xlabel('Matrix Size')
plt.ylabel('Condition Number')
plt.title('Condition Number Growth: Hilbert Matrix')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(sizes, vandermonde_conditions, 'ro-', label='Vandermonde Matrix')
plt.xlabel('Matrix Size')
plt.ylabel('Condition Number')
plt.title('Condition Number Growth: Vandermonde Matrix')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Hilbert matrix condition numbers: {hilbert_conditions}")
print(f"Vandermonde matrix condition numbers: {vandermonde_conditions}")
```

### Numerical Stability Examples

```python
def advanced_numerical_stability_examples():
    """
    Advanced examples of numerical stability issues and solutions
    
    Mathematical concepts:
    1. Catastrophic cancellation
    2. Loss of significance
    3. Algorithm stability
    4. Preconditioning
    """
    
    # 1. Catastrophic cancellation example
    print("=== Catastrophic Cancellation Example ===")
    
    def quadratic_formula_unstable(a, b, c):
        """Unstable quadratic formula implementation"""
        x1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        x2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
        return x1, x2
    
    def quadratic_formula_stable(a, b, c):
        """Stable quadratic formula implementation"""
        # Use Vieta's formula to avoid cancellation
        if b >= 0:
            x1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            x2 = c / (a * x1)
        else:
            x1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            x2 = c / (a * x1)
        return x1, x2
    
    # Test with nearly equal roots
    a, b, c = 1, -1000.001, 1000
    
    x1_unstable, x2_unstable = quadratic_formula_unstable(a, b, c)
    x1_stable, x2_stable = quadratic_formula_stable(a, b, c)
    
    print(f"Unstable: x1 = {x1_unstable:.10f}, x2 = {x2_unstable:.10f}")
    print(f"Stable: x1 = {x1_stable:.10f}, x2 = {x2_stable:.10f}")
    print(f"Product (should be c/a = {c/a}):")
    print(f"  Unstable: {x1_unstable * x2_unstable:.10f}")
    print(f"  Stable: {x1_stable * x2_stable:.10f}")
    
    # 2. Loss of significance in matrix operations
    print(f"\n=== Loss of Significance in Matrix Operations ===")
    
    def matrix_conditioning_example():
        """Demonstrate matrix conditioning and its effects"""
        
        # Create a poorly conditioned matrix
        n = 5
        A = np.random.randn(n, n)
        # Make it nearly singular
        A[:, -1] = A[:, 0] + 1e-10 * np.random.randn(n)
        
        # Create right-hand side
        b = np.ones(n)
        
        # Solve with different methods
        methods = {
            'Direct': lambda: np.linalg.solve(A, b),
            'QR': lambda: np.linalg.solve(A.T @ A, A.T @ b),
            'SVD': lambda: np.linalg.pinv(A) @ b,
            'Regularized': lambda: np.linalg.solve(A.T @ A + 1e-8 * np.eye(n), A.T @ b)
        }
        
        results = {}
        for name, method in methods.items():
            try:
                x = method()
                residual = np.linalg.norm(A @ x - b)
                results[name] = {'x': x, 'residual': residual}
                print(f"{name}: residual = {residual:.2e}")
            except Exception as e:
                print(f"{name}: Failed - {e}")
        
        return results
    
    matrix_results = matrix_conditioning_example()
    
    # 3. Algorithm stability comparison
    print(f"\n=== Algorithm Stability Comparison ===")
    
    def compare_linear_system_solvers():
        """Compare different linear system solvers for stability"""
        
        # Create test problems
        problems = {
            'Well-conditioned': np.random.randn(10, 10),
            'Ill-conditioned': np.array([[1, 1], [1, 1.0001]]),
            'Singular': np.array([[1, 1], [1, 1]])
        }
        
        solvers = {
            'LU': lambda A, b: np.linalg.solve(A, b),
            'QR': lambda A, b: np.linalg.solve(A.T @ A, A.T @ b),
            'SVD': lambda A, b: np.linalg.pinv(A) @ b,
            'Tikhonov': lambda A, b: np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ b)
        }
        
        for problem_name, A in problems.items():
            print(f"\n{problem_name} problem:")
            print(f"Condition number: {np.linalg.cond(A):.2e}")
            
            b = np.ones(A.shape[0])
            
            for solver_name, solver in solvers.items():
                try:
                    x = solver(A, b)
                    residual = np.linalg.norm(A @ x - b)
                    print(f"  {solver_name}: residual = {residual:.2e}")
                except Exception as e:
                    print(f"  {solver_name}: Failed - {e}")
    
    compare_linear_system_solvers()
    
    # 4. Preconditioning example
    print(f"\n=== Preconditioning Example ===")
    
    def preconditioning_demo():
        """Demonstrate the effect of preconditioning"""
        
        # Create a poorly conditioned system
        n = 100
        A = np.random.randn(n, n)
        A = A.T @ A + 1e-6 * np.eye(n)  # Make it symmetric positive definite but ill-conditioned
        
        b = np.random.randn(n)
        
        # Without preconditioning
        x_direct = np.linalg.solve(A, b)
        
        # With diagonal preconditioning
        D = np.diag(np.sqrt(np.diag(A)))
        D_inv = np.diag(1.0 / np.sqrt(np.diag(A)))
        
        A_precond = D_inv @ A @ D_inv
        b_precond = D_inv @ b
        
        x_precond_raw = np.linalg.solve(A_precond, b_precond)
        x_precond = D_inv @ x_precond_raw
        
        # Compare residuals
        residual_direct = np.linalg.norm(A @ x_direct - b)
        residual_precond = np.linalg.norm(A @ x_precond - b)
        
        print(f"Condition number (original): {np.linalg.cond(A):.2e}")
        print(f"Condition number (preconditioned): {np.linalg.cond(A_precond):.2e}")
        print(f"Residual (direct): {residual_direct:.2e}")
        print(f"Residual (preconditioned): {residual_precond:.2e}")
        
        return A, A_precond, x_direct, x_precond
    
    A_orig, A_precond, x_direct, x_precond = preconditioning_demo()
    
    # Visualize the effect of preconditioning
    plt.figure(figsize=(12, 4))
    
    # Original matrix spectrum
    eigenvals_orig = np.linalg.eigvals(A_orig)
    plt.subplot(1, 3, 1)
    plt.hist(eigenvals_orig, bins=20, alpha=0.7, label='Original')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Original Matrix Spectrum')
    plt.legend()
    
    # Preconditioned matrix spectrum
    eigenvals_precond = np.linalg.eigvals(A_precond)
    plt.subplot(1, 3, 2)
    plt.hist(eigenvals_precond, bins=20, alpha=0.7, label='Preconditioned', color='orange')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Preconditioned Matrix Spectrum')
    plt.legend()
    
    # Condition number comparison
    plt.subplot(1, 3, 3)
    cond_orig = np.linalg.cond(A_orig)
    cond_precond = np.linalg.cond(A_precond)
    plt.bar(['Original', 'Preconditioned'], [cond_orig, cond_precond])
    plt.ylabel('Condition Number')
    plt.title('Condition Number Comparison')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Run advanced numerical stability examples
advanced_numerical_stability_examples()
```

## 2. Iterative Methods for Linear Systems

### Jacobi Method
```python
def jacobi_iteration(A, b, x0=None, max_iter=1000, tol=1e-6):
    """Solve Ax = b using Jacobi iteration"""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    
    # Extract diagonal and off-diagonal parts
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    
    for iteration in range(max_iter):
        x_new = np.linalg.solve(D, b - L_plus_U @ x)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        x = x_new
    
    return x

# Test Jacobi method
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 5, 0])

x_jacobi = jacobi_iteration(A, b)
x_exact = np.linalg.solve(A, b)

print("Jacobi solution:", x_jacobi)
print("Exact solution:", x_exact)
print("Error:", np.linalg.norm(x_jacobi - x_exact))
```

### Gauss-Seidel Method
```python
def gauss_seidel_iteration(A, b, x0=None, max_iter=1000, tol=1e-6):
    """Solve Ax = b using Gauss-Seidel iteration"""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            # Forward substitution
            sum_val = 0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            x[i] = (b[i] - sum_val) / A[i, i]
        
        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return x

# Test Gauss-Seidel
x_gauss_seidel = gauss_seidel_iteration(A, b)
print("Gauss-Seidel solution:", x_gauss_seidel)
print("Error:", np.linalg.norm(x_gauss_seidel - x_exact))
```

### Conjugate Gradient Method
```python
def conjugate_gradient(A, b, x0=None, max_iter=1000, tol=1e-6):
    """Solve Ax = b using Conjugate Gradient method (for symmetric positive definite A)"""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    
    for iteration in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        # Check convergence
        if np.linalg.norm(r_new) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    
    return x

# Test Conjugate Gradient
x_cg = conjugate_gradient(A, b)
print("Conjugate Gradient solution:", x_cg)
print("Error:", np.linalg.norm(x_cg - x_exact))
```

## 3. Sparse Matrix Operations

```python
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, eigsh

def sparse_matrix_example():
    """Demonstrate sparse matrix operations"""
    
    # Create a sparse matrix (tridiagonal)
    n = 1000
    diagonals = [np.ones(n), -0.5 * np.ones(n-1), -0.5 * np.ones(n-1)]
    A_dense = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[1], -1)
    
    # Convert to sparse format
    A_sparse = csr_matrix(A_dense)
    
    print(f"Dense matrix size: {A_dense.nbytes / 1024:.1f} KB")
    print(f"Sparse matrix size: {A_sparse.data.nbytes / 1024:.1f} KB")
    print(f"Compression ratio: {A_dense.nbytes / A_sparse.data.nbytes:.1f}x")
    
    # Solve linear system
    b = np.ones(n)
    
    # Dense solve
    import time
    start_time = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    dense_time = time.time() - start_time
    
    # Sparse solve
    start_time = time.time()
    x_sparse = spsolve(A_sparse, b)
    sparse_time = time.time() - start_time
    
    print(f"\nDense solve time: {dense_time:.4f} seconds")
    print(f"Sparse solve time: {sparse_time:.4f} seconds")
    print(f"Speedup: {dense_time / sparse_time:.1f}x")
    
    # Check accuracy
    error = np.linalg.norm(x_dense - x_sparse)
    print(f"Solution error: {error:.2e}")
    
    return A_sparse, x_sparse

A_sparse, x_sparse = sparse_matrix_example()
```

## 4. Eigenvalue Problems

### Power Iteration
```python
def power_iteration(A, max_iter=1000, tol=1e-6):
    """Find the largest eigenvalue and eigenvector using power iteration"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for iteration in range(max_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        
        # Estimate eigenvalue
        eigenvalue = np.dot(v_new, A @ v_new)
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        v = v_new
    
    return eigenvalue, v

# Test power iteration
A = np.random.randn(5, 5)
A = A + A.T  # Make symmetric
eigenvalue, eigenvector = power_iteration(A)

# Compare with numpy
eigenvals_np, eigenvecs_np = np.linalg.eig(A)
max_eigenval_idx = np.argmax(np.abs(eigenvals_np))

print(f"Power iteration eigenvalue: {eigenvalue:.6f}")
print(f"Numpy largest eigenvalue: {eigenvals_np[max_eigenval_idx]:.6f}")
print(f"Eigenvalue error: {abs(eigenvalue - eigenvals_np[max_eigenval_idx]):.2e}")
```

### Inverse Iteration
```python
def inverse_iteration(A, sigma, max_iter=1000, tol=1e-6):
    """Find eigenvalue closest to sigma using inverse iteration"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    # Shift matrix
    A_shifted = A - sigma * np.eye(n)
    
    for iteration in range(max_iter):
        # Solve linear system
        v_new = np.linalg.solve(A_shifted, v)
        v_new = v_new / np.linalg.norm(v_new)
        
        # Estimate eigenvalue
        eigenvalue = np.dot(v_new, A @ v_new)
        
        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        v = v_new
    
    return eigenvalue, v

# Test inverse iteration
sigma = 2.0  # Target eigenvalue
eigenvalue_inv, eigenvector_inv = inverse_iteration(A, sigma)
print(f"Inverse iteration eigenvalue: {eigenvalue_inv:.6f}")
```

## 5. QR Algorithm for Eigenvalues

```python
def qr_algorithm(A, max_iter=100, tol=1e-6):
    """Find all eigenvalues using QR algorithm"""
    n = A.shape[0]
    A_k = A.copy()
    
    for iteration in range(max_iter):
        # QR decomposition
        Q, R = np.linalg.qr(A_k)
        
        # Update A
        A_new = R @ Q
        
        # Check convergence (off-diagonal elements)
        off_diag_norm = np.linalg.norm(A_new - np.diag(np.diag(A_new)))
        
        if off_diag_norm < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        A_k = A_new
    
    eigenvalues = np.diag(A_k)
    return eigenvalues

# Test QR algorithm
eigenvalues_qr = qr_algorithm(A)
eigenvalues_np = np.linalg.eigvals(A)

print("QR algorithm eigenvalues:", eigenvalues_qr)
print("Numpy eigenvalues:", eigenvalues_np)
print("Maximum error:", np.max(np.abs(eigenvalues_qr - eigenvalues_np)))
```

## 6. Singular Value Decomposition (SVD) for Large Matrices

```python
def truncated_svd(A, k):
    """Compute truncated SVD for large matrices"""
    from scipy.sparse.linalg import svds
    
    if A.shape[0] * A.shape[1] > 1e6:  # Use sparse SVD for large matrices
        U, S, Vt = svds(A, k=k)
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    
    return U, S, Vt

# Example: Image compression using truncated SVD
def compress_image_svd(image, k):
    """Compress image using truncated SVD"""
    U, S, Vt = truncated_svd(image, k)
    compressed = U @ np.diag(S) @ Vt
    
    # Calculate compression ratio
    original_size = image.size
    compressed_size = U.size + S.size + Vt.size
    compression_ratio = original_size / compressed_size
    
    return compressed, compression_ratio

# Test with a simple image
image = np.random.rand(100, 100)
compressed_10, ratio_10 = compress_image_svd(image, 10)
compressed_50, ratio_50 = compress_image_svd(image, 50)

print(f"Compression ratio (k=10): {ratio_10:.1f}x")
print(f"Compression ratio (k=50): {ratio_50:.1f}x")
print(f"Error (k=10): {np.linalg.norm(image - compressed_10):.4f}")
print(f"Error (k=50): {np.linalg.norm(image - compressed_50):.4f}")
```

## 7. Performance Comparison

```python
def performance_comparison():
    """Compare performance of different linear algebra operations"""
    import time
    
    sizes = [100, 500, 1000, 2000]
    times = {'dense_solve': [], 'sparse_solve': [], 'eigenvalues': [], 'svd': []}
    
    for n in sizes:
        print(f"\nTesting size {n}x{n}")
        
        # Generate test matrix
        A = np.random.randn(n, n)
        A = A + A.T  # Make symmetric
        b = np.random.randn(n)
        
        # Dense solve
        start_time = time.time()
        x = np.linalg.solve(A, b)
        times['dense_solve'].append(time.time() - start_time)
        
        # Sparse solve
        A_sparse = csr_matrix(A)
        start_time = time.time()
        x_sparse = spsolve(A_sparse, b)
        times['sparse_solve'].append(time.time() - start_time)
        
        # Eigenvalues
        start_time = time.time()
        eigenvals = np.linalg.eigvals(A)
        times['eigenvalues'].append(time.time() - start_time)
        
        # SVD
        start_time = time.time()
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        times['svd'].append(time.time() - start_time)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for i, (operation, time_list) in enumerate(times.items()):
        plt.subplot(2, 2, i+1)
        plt.loglog(sizes, time_list, 'bo-', label=operation)
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'{operation.replace("_", " ").title()}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return times

times = performance_comparison()
```

## Exercises

1. **Condition Number**: Compute condition numbers for different matrix types and analyze their sensitivity.
2. **Iterative Methods**: Implement and compare Jacobi, Gauss-Seidel, and Conjugate Gradient methods.
3. **Sparse Matrices**: Create a sparse matrix and compare dense vs sparse operations.
4. **Eigenvalue Methods**: Implement power iteration and inverse iteration for finding eigenvalues.
5. **Performance Analysis**: Profile different linear algebra operations for various matrix sizes.

## Solutions

```python
# Exercise 1: Condition Number Analysis
def analyze_condition_numbers():
    matrices = {
        'Identity': np.eye(5),
        'Random': np.random.randn(5, 5),
        'Hilbert': linalg.hilbert(5),
        'Vandermonde': np.vander(np.arange(1, 6))
    }
    
    for name, A in matrices.items():
        cond = condition_number(A)
        print(f"{name} matrix condition number: {cond:.2e}")

# Exercise 2: Iterative Methods Comparison
def compare_iterative_methods(A, b):
    x_exact = np.linalg.solve(A, b)
    
    x_jacobi = jacobi_iteration(A, b)
    x_gauss_seidel = gauss_seidel_iteration(A, b)
    x_cg = conjugate_gradient(A, b)
    
    errors = {
        'Jacobi': np.linalg.norm(x_jacobi - x_exact),
        'Gauss-Seidel': np.linalg.norm(x_gauss_seidel - x_exact),
        'Conjugate Gradient': np.linalg.norm(x_cg - x_exact)
    }
    
    for method, error in errors.items():
        print(f"{method} error: {error:.2e}")

# Exercise 3: Sparse vs Dense
def sparse_vs_dense_comparison(n=1000):
    # Create tridiagonal matrix
    diagonals = [np.ones(n), -0.5 * np.ones(n-1), -0.5 * np.ones(n-1)]
    A_dense = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[1], -1)
    A_sparse = csr_matrix(A_dense)
    b = np.ones(n)
    
    # Compare solve times
    import time
    
    start_time = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    dense_time = time.time() - start_time
    
    start_time = time.time()
    x_sparse = spsolve(A_sparse, b)
    sparse_time = time.time() - start_time
    
    print(f"Dense solve time: {dense_time:.4f}s")
    print(f"Sparse solve time: {sparse_time:.4f}s")
    print(f"Speedup: {dense_time / sparse_time:.1f}x")

# Exercise 4: Eigenvalue Methods
def eigenvalue_methods_comparison(A):
    # Power iteration
    eigenval_power, eigenvec_power = power_iteration(A)
    
    # Inverse iteration (target largest eigenvalue)
    eigenvals_np = np.linalg.eigvals(A)
    target = np.max(np.abs(eigenvals_np))
    eigenval_inv, eigenvec_inv = inverse_iteration(A, target)
    
    # QR algorithm
    eigenvals_qr = qr_algorithm(A)
    
    print(f"Power iteration: {eigenval_power:.6f}")
    print(f"Inverse iteration: {eigenval_inv:.6f}")
    print(f"QR algorithm (max): {np.max(eigenvals_qr):.6f}")
    print(f"Numpy (max): {np.max(eigenvals_np):.6f}")

# Exercise 5: Performance Analysis
def performance_analysis():
    sizes = [100, 500, 1000]
    operations = ['solve', 'eigenvalues', 'svd', 'inverse']
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        A = np.random.randn(size, size)
        
        for operation in operations:
            start_time = time.time()
            if operation == 'solve':
                b = np.random.randn(size)
                x = np.linalg.solve(A, b)
            elif operation == 'eigenvalues':
                eigenvals = np.linalg.eigvals(A)
            elif operation == 'svd':
                U, S, Vt = np.linalg.svd(A, full_matrices=False)
            elif operation == 'inverse':
                A_inv = np.linalg.inv(A)
            
            elapsed_time = time.time() - start_time
            print(f"  {operation}: {elapsed_time:.4f}s")
```

## Key Takeaways

- Numerical stability is crucial for reliable computations
- Condition number measures problem sensitivity to perturbations
- Iterative methods are efficient for large, sparse systems
- Sparse matrix formats save memory and computation time
- Different eigenvalue algorithms have different strengths
- Performance varies significantly with matrix size and structure

## Next Chapter

In the final chapter, we'll provide a comprehensive summary of all concepts covered and present practice problems to reinforce understanding. 