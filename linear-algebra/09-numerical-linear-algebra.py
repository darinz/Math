"""
Numerical Linear Algebra
========================

This module covers advanced numerical linear algebra concepts including:
- Numerical stability and conditioning
- Iterative methods for linear systems
- Sparse matrix operations
- Eigenvalue computation algorithms
- QR algorithm for eigenvalues
- SVD for large matrices
- Performance analysis and comparisons

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh
import time
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("NUMERICAL LINEAR ALGEBRA")
print("=" * 60)

# ============================================================================
# 1. NUMERICAL STABILITY AND CONDITIONING
# ============================================================================

print("\n1. NUMERICAL STABILITY AND CONDITIONING")
print("-" * 50)

def compute_condition_number(A):
    """Compute the condition number of a matrix"""
    return np.linalg.cond(A)

def analyze_numerical_stability(A, b, perturbation_level=1e-10):
    """
    Analyze numerical stability by solving Ax = b with perturbed b
    
    Parameters:
    -----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    perturbation_level : float
        Level of perturbation to add to b
        
    Returns:
    --------
    dict : Analysis results
    """
    # Exact solution
    x_exact = np.linalg.solve(A, b)
    
    # Perturbed right-hand side
    b_perturbed = b + perturbation_level * np.random.randn(len(b))
    x_perturbed = np.linalg.solve(A, b_perturbed)
    
    # Compute errors
    rel_error_b = np.linalg.norm(b_perturbed - b) / np.linalg.norm(b)
    rel_error_x = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)
    
    # Condition number
    condition_number = np.linalg.cond(A)
    
    # Theoretical bound
    theoretical_bound = condition_number * rel_error_b
    
    return {
        'condition_number': condition_number,
        'relative_error_b': rel_error_b,
        'relative_error_x': rel_error_x,
        'amplification_factor': rel_error_x / rel_error_b,
        'theoretical_bound': theoretical_bound,
        'satisfies_bound': rel_error_x <= theoretical_bound
    }

# Create test matrices with different condition numbers
def create_test_matrices(n=10):
    """Create matrices with different numerical properties"""
    # Identity matrix (well-conditioned)
    I = np.eye(n)
    
    # Hilbert matrix (ill-conditioned)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    
    # Random matrix with controlled condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.logspace(0, 6, n))  # Condition number ~10^6
    V, _ = np.linalg.qr(np.random.randn(n, n))
    A = U @ S @ V.T
    
    return I, H, A

# Test numerical stability
print("Testing numerical stability on different matrices...")
I, H, A = create_test_matrices(10)
b = np.random.randn(10)

matrices = [("Identity", I), ("Hilbert", H), ("Random", A)]

for name, matrix in matrices:
    result = analyze_numerical_stability(matrix, b)
    print(f"\n{name} Matrix:")
    print(f"  Condition number: {result['condition_number']:.2e}")
    print(f"  Amplification factor: {result['amplification_factor']:.2e}")
    print(f"  Theoretical bound: {result['theoretical_bound']:.2e}")
    print(f"  Satisfies bound: {result['satisfies_bound']}")

# ============================================================================
# 2. ITERATIVE METHODS FOR LINEAR SYSTEMS
# ============================================================================

print("\n\n2. ITERATIVE METHODS FOR LINEAR SYSTEMS")
print("-" * 50)

def jacobi_method(A, b, max_iter=1000, tol=1e-10):
    """
    Solve Ax = b using Jacobi iteration
    
    Parameters:
    -----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    x : ndarray
        Solution vector
    iterations : int
        Number of iterations performed
    residuals : list
        Residual norms at each iteration
    """
    n = len(A)
    x = np.zeros(n)
    residuals = []
    
    # Extract diagonal and off-diagonal parts
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    
    for k in range(max_iter):
        x_new = np.linalg.solve(D, b - L_plus_U @ x)
        
        # Compute residual
        residual = np.linalg.norm(A @ x_new - b)
        residuals.append(residual)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return x, k + 1, residuals

def gauss_seidel_method(A, b, max_iter=1000, tol=1e-10):
    """
    Solve Ax = b using Gauss-Seidel iteration
    """
    n = len(A)
    x = np.zeros(n)
    residuals = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
        
        # Compute residual
        residual = np.linalg.norm(A @ x - b)
        residuals.append(residual)
        
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x, k + 1, residuals

def conjugate_gradient_method(A, b, max_iter=None, tol=1e-10):
    """
    Solve Ax = b using Conjugate Gradient method (for symmetric positive definite A)
    """
    n = len(A)
    if max_iter is None:
        max_iter = n
    
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    residuals = [np.linalg.norm(r)]
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)
        
        if residual_norm < tol:
            break
        
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    
    return x, k + 1, residuals

# Test iterative methods
print("Testing iterative methods on different matrix types...")

def create_test_systems(n=100):
    """Create test matrices with different properties"""
    # Symmetric positive definite
    A_spd = np.random.randn(n, n)
    A_spd = A_spd.T @ A_spd + n * np.eye(n)
    
    # Diagonally dominant
    A_dd = np.random.randn(n, n)
    A_dd = A_dd + n * np.eye(n)
    
    # Random sparse
    A_sparse = np.random.randn(n, n)
    mask = np.random.random((n, n)) < 0.1  # 10% sparsity
    A_sparse[~mask] = 0
    A_sparse = A_sparse + n * np.eye(n)
    
    b = np.random.randn(n)
    return A_spd, A_dd, A_sparse, b

# Test all methods
A_spd, A_dd, A_sparse, b = create_test_systems(50)

test_matrices = [("SPD", A_spd), ("Diagonally Dominant", A_dd), ("Sparse", A_sparse)]

for name, A in test_matrices:
    print(f"\n{name} Matrix:")
    
    x_jacobi, iter_jacobi, res_jacobi = jacobi_method(A, b)
    x_gs, iter_gs, res_gs = gauss_seidel_method(A, b)
    x_cg, iter_cg, res_cg = conjugate_gradient_method(A, b)
    
    x_exact = np.linalg.solve(A, b)
    
    print(f"  Jacobi: {iter_jacobi} iterations, error: {np.linalg.norm(x_jacobi - x_exact):.2e}")
    print(f"  Gauss-Seidel: {iter_gs} iterations, error: {np.linalg.norm(x_gs - x_exact):.2e}")
    print(f"  Conjugate Gradient: {iter_cg} iterations, error: {np.linalg.norm(x_cg - x_exact):.2e}")

# ============================================================================
# 3. SPARSE MATRIX OPERATIONS
# ============================================================================

print("\n\n3. SPARSE MATRIX OPERATIONS")
print("-" * 50)

def dense_to_csr(A):
    """Convert dense matrix to CSR format"""
    n, m = A.shape
    data = []
    indices = []
    indptr = [0]
    
    for i in range(n):
        for j in range(m):
            if A[i, j] != 0:
                data.append(A[i, j])
                indices.append(j)
        indptr.append(len(data))
    
    return np.array(data), np.array(indices), np.array(indptr)

def csr_matvec(data, indices, indptr, x):
    """Matrix-vector multiplication for CSR format"""
    n = len(indptr) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(indptr[i], indptr[i+1]):
            y[i] += data[j] * x[indices[j]]
    
    return y

def create_sparse_matrix(n=1000, density=0.01):
    """Create random sparse matrix"""
    nnz = int(n * n * density)
    data = np.random.randn(nnz)
    row_indices = np.random.randint(0, n, nnz)
    col_indices = np.random.randint(0, n, nnz)
    
    # Create dense matrix
    A_dense = np.zeros((n, n))
    for i in range(nnz):
        A_dense[row_indices[i], col_indices[i]] = data[i]
    
    # Convert to CSR
    data_csr, indices_csr, indptr_csr = dense_to_csr(A_dense)
    
    return A_dense, data_csr, indices_csr, indptr_csr

# Performance test
print("Testing sparse matrix performance...")
n = 500
A_dense, data_csr, indices_csr, indptr_csr = create_sparse_matrix(n, 0.01)
x = np.random.randn(n)

# Time dense multiplication
start = time.time()
y_dense = A_dense @ x
time_dense = time.time() - start

# Time sparse multiplication
start = time.time()
y_sparse = csr_matvec(data_csr, indices_csr, indptr_csr, x)
time_sparse = time.time() - start

print(f"Dense multiplication: {time_dense:.4f}s")
print(f"Sparse multiplication: {time_sparse:.4f}s")
print(f"Speedup: {time_dense/time_sparse:.2f}x")
print(f"Memory dense: {A_dense.nbytes} bytes")
print(f"Memory sparse: {(data_csr.nbytes + indices_csr.nbytes + indptr_csr.nbytes)} bytes")

# ============================================================================
# 4. EIGENVALUE PROBLEMS
# ============================================================================

print("\n\n4. EIGENVALUE PROBLEMS")
print("-" * 50)

def power_iteration(A, max_iter=1000, tol=1e-10):
    """
    Find dominant eigenvalue and eigenvector using power iteration
    
    Parameters:
    -----------
    A : ndarray
        Matrix to find eigenvalues of
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    lambda_1 : float
        Dominant eigenvalue
    v_1 : ndarray
        Dominant eigenvector
    iterations : int
        Number of iterations performed
    """
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for k in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    lambda_1 = (x.T @ A @ x) / (x.T @ x)
    return lambda_1, x, k + 1

def inverse_iteration(A, mu, max_iter=1000, tol=1e-10):
    """
    Find eigenvalue closest to mu using inverse iteration
    
    Parameters:
    -----------
    A : ndarray
        Matrix to find eigenvalues of
    mu : float
        Shift parameter
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    lambda_i : float
        Eigenvalue closest to mu
    v_i : ndarray
        Corresponding eigenvector
    iterations : int
        Number of iterations performed
    """
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for k in range(max_iter):
        # Solve (A - mu*I)y = x
        y = np.linalg.solve(A - mu * np.eye(n), x)
        x_new = y / np.linalg.norm(y)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    lambda_i = mu + 1 / (x.T @ y)
    return lambda_i, x, k + 1

# Test eigenvalue methods
print("Testing eigenvalue computation methods...")

# Create test matrix
n = 50
A = np.random.randn(n, n)
A = A + A.T  # Make symmetric

# Find eigenvalues using different methods
eigenvals_exact, eigenvecs_exact = np.linalg.eig(A)
lambda_max_exact = np.max(np.abs(eigenvals_exact))
lambda_min_exact = np.min(np.abs(eigenvals_exact))

# Power iteration for largest eigenvalue
lambda_max_power, v_max, iter_max = power_iteration(A)
print(f"Largest eigenvalue:")
print(f"  Exact: {lambda_max_exact:.6f}")
print(f"  Power iteration: {lambda_max_power:.6f}")
print(f"  Error: {abs(lambda_max_power - lambda_max_exact):.2e}")
print(f"  Iterations: {iter_max}")

# Inverse iteration for smallest eigenvalue
lambda_min_inv, v_min, iter_min = inverse_iteration(A, 0)
print(f"\nSmallest eigenvalue:")
print(f"  Exact: {lambda_min_exact:.6f}")
print(f"  Inverse iteration: {lambda_min_inv:.6f}")
print(f"  Error: {abs(lambda_min_inv - lambda_min_exact):.2e}")
print(f"  Iterations: {iter_min}")

# ============================================================================
# 5. QR ALGORITHM FOR EIGENVALUES
# ============================================================================

print("\n\n5. QR ALGORITHM FOR EIGENVALUES")
print("-" * 50)

def qr_algorithm(A, max_iter=100, tol=1e-10):
    """
    Basic QR algorithm for eigenvalue computation
    
    Parameters:
    -----------
    A : ndarray
        Matrix to find eigenvalues of
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    eigenvals : ndarray
        Eigenvalues
    A_final : ndarray
        Final matrix (should be upper triangular)
    iterations : int
        Number of iterations performed
    """
    n = A.shape[0]
    A_k = A.copy()
    
    for k in range(max_iter):
        # QR decomposition
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        
        # Check convergence (off-diagonal elements)
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
    
    return np.diag(A_k), A_k, k + 1

def qr_algorithm_with_shift(A, max_iter=100, tol=1e-10):
    """
    QR algorithm with Wilkinson shift for faster convergence
    """
    n = A.shape[0]
    A_k = A.copy()
    
    for k in range(max_iter):
        # Wilkinson shift
        if n > 1:
            a = A_k[n-2, n-2]
            b = A_k[n-2, n-1]
            c = A_k[n-1, n-1]
            delta = (a - c) / 2
            mu = c - b**2 / (delta + np.sign(delta) * np.sqrt(delta**2 + b**2))
        else:
            mu = A_k[0, 0]
        
        # Shifted QR
        Q, R = np.linalg.qr(A_k - mu * np.eye(n))
        A_k = R @ Q + mu * np.eye(n)
        
        # Check convergence
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
    
    return np.diag(A_k), A_k, k + 1

# Test QR algorithm
print("Testing QR algorithm...")

# Create symmetric test matrix
n = 20
A = np.random.randn(n, n)
A = A + A.T

# Exact eigenvalues
eigenvals_exact = np.linalg.eigvals(A)
eigenvals_exact = np.sort(eigenvals_exact)

# QR algorithm without shift
eigenvals_qr, _, iter_qr = qr_algorithm(A)
eigenvals_qr = np.sort(eigenvals_qr)

# QR algorithm with shift
eigenvals_qr_shift, _, iter_qr_shift = qr_algorithm_with_shift(A)
eigenvals_qr_shift = np.sort(eigenvals_qr_shift)

print("Eigenvalue Comparison:")
print(f"Exact eigenvalues (first 5): {eigenvals_exact[:5]}")
print(f"QR without shift (first 5): {eigenvals_qr[:5]}")
print(f"QR with shift (first 5): {eigenvals_qr_shift[:5]}")
print(f"\nIterations:")
print(f"QR without shift: {iter_qr}")
print(f"QR with shift: {iter_qr_shift}")
print(f"\nErrors:")
print(f"QR without shift: {np.linalg.norm(eigenvals_qr - eigenvals_exact):.2e}")
print(f"QR with shift: {np.linalg.norm(eigenvals_qr_shift - eigenvals_exact):.2e}")

# ============================================================================
# 6. SVD FOR LARGE MATRICES
# ============================================================================

print("\n\n6. SVD FOR LARGE MATRICES")
print("-" * 50)

def randomized_svd(A, k, oversample=10):
    """
    Randomized SVD for large matrices
    
    Parameters:
    -----------
    A : ndarray
        Matrix to decompose
    k : int
        Number of singular values to compute
    oversample : int
        Number of extra samples for better accuracy
        
    Returns:
    --------
    U : ndarray
        Left singular vectors
    s : ndarray
        Singular values
    Vt : ndarray
        Right singular vectors (transposed)
    """
    m, n = A.shape
    l = k + oversample
    
    # Generate random matrix
    Omega = np.random.randn(n, l)
    
    # Compute Y = A * Omega
    Y = A @ Omega
    
    # QR decomposition of Y
    Q, _ = np.linalg.qr(Y)
    
    # Project A onto Q
    B = Q.T @ A
    
    # SVD of B
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Reconstruct U
    U = Q @ U_tilde
    
    return U[:, :k], s[:k], Vt[:k, :]

def randomized_svd_power_iteration(A, k, oversample=10, power_iter=2):
    """
    Randomized SVD with power iteration for better accuracy
    """
    m, n = A.shape
    l = k + oversample
    
    # Generate random matrix
    Omega = np.random.randn(n, l)
    
    # Power iteration
    Y = A @ Omega
    for _ in range(power_iter):
        Y = A.T @ Y
        Y = A @ Y
    
    # QR decomposition
    Q, _ = np.linalg.qr(Y)
    
    # Project and compute SVD
    B = Q.T @ A
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    
    return U[:, :k], s[:k], Vt[:k, :]

# Test randomized SVD
print("Testing randomized SVD...")

# Create large test matrix
m, n = 500, 300
A = np.random.randn(m, n)
k = 10

# Exact SVD
U_exact, s_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
U_exact = U_exact[:, :k]
s_exact = s_exact[:k]
Vt_exact = Vt_exact[:k, :]

# Randomized SVD
U_rand, s_rand, Vt_rand = randomized_svd(A, k)

# Randomized SVD with power iteration
U_rand_power, s_rand_power, Vt_rand_power = randomized_svd_power_iteration(A, k)

# Compare accuracy
A_exact = U_exact @ np.diag(s_exact) @ Vt_exact
A_rand = U_rand @ np.diag(s_rand) @ Vt_rand
A_rand_power = U_rand_power @ np.diag(s_rand_power) @ Vt_rand_power

error_exact = np.linalg.norm(A - A_exact, 'fro')
error_rand = np.linalg.norm(A - A_rand, 'fro')
error_rand_power = np.linalg.norm(A - A_rand_power, 'fro')

print("SVD Comparison:")
print(f"Exact SVD error: {error_exact:.2e}")
print(f"Randomized SVD error: {error_rand:.2e}")
print(f"Randomized SVD with power iteration error: {error_rand_power:.2e}")
print(f"Error ratio (rand/exact): {error_rand/error_exact:.2f}")
print(f"Error ratio (rand_power/exact): {error_rand_power/error_exact:.2f}")

# ============================================================================
# 7. PERFORMANCE COMPARISON AND VISUALIZATION
# ============================================================================

print("\n\n7. PERFORMANCE COMPARISON AND VISUALIZATION")
print("-" * 50)

def benchmark_methods(sizes=[50, 100, 200, 500]):
    """Benchmark different methods across matrix sizes"""
    results = {
        'size': [],
        'dense_solve_time': [],
        'sparse_solve_time': [],
        'power_iter_time': [],
        'qr_time': []
    }
    
    for n in sizes:
        print(f"Testing size n = {n}...")
        
        # Create test matrix
        A = np.random.randn(n, n)
        A = A + A.T + n * np.eye(n)  # Make SPD
        b = np.random.randn(n)
        
        # Dense solve
        start = time.time()
        x_dense = np.linalg.solve(A, b)
        dense_time = time.time() - start
        
        # Sparse solve
        A_sparse = sparse.csr_matrix(A)
        start = time.time()
        x_sparse = spsolve(A_sparse, b)
        sparse_time = time.time() - start
        
        # Power iteration
        start = time.time()
        lambda_max, _, _ = power_iteration(A)
        power_time = time.time() - start
        
        # QR algorithm
        start = time.time()
        eigenvals, _, _ = qr_algorithm(A, max_iter=50)
        qr_time = time.time() - start
        
        results['size'].append(n)
        results['dense_solve_time'].append(dense_time)
        results['sparse_solve_time'].append(sparse_time)
        results['power_iter_time'].append(power_time)
        results['qr_time'].append(qr_time)
    
    return results

# Run benchmarks
print("Running performance benchmarks...")
benchmark_results = benchmark_methods([50, 100, 200])

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Iterative methods convergence
ax1.semilogy(res_jacobi, label='Jacobi', marker='o')
ax1.semilogy(res_gs, label='Gauss-Seidel', marker='s')
ax1.semilogy(res_cg, label='Conjugate Gradient', marker='^')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Residual Norm')
ax1.set_title('Convergence of Iterative Methods')
ax1.legend()
ax1.grid(True)

# Plot 2: Performance comparison
sizes = benchmark_results['size']
ax2.loglog(sizes, benchmark_results['dense_solve_time'], 'o-', label='Dense Solve')
ax2.loglog(sizes, benchmark_results['sparse_solve_time'], 's-', label='Sparse Solve')
ax2.loglog(sizes, benchmark_results['power_iter_time'], '^-', label='Power Iteration')
ax2.loglog(sizes, benchmark_results['qr_time'], 'd-', label='QR Algorithm')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Performance Comparison')
ax2.legend()
ax2.grid(True)

# Plot 3: Condition number analysis
condition_numbers = []
matrix_names = []
for name, matrix in matrices:
    condition_numbers.append(np.linalg.cond(matrix))
    matrix_names.append(name)

ax3.bar(matrix_names, condition_numbers)
ax3.set_ylabel('Condition Number')
ax3.set_title('Condition Numbers of Test Matrices')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: SVD accuracy comparison
methods = ['Exact', 'Randomized', 'Randomized+Power']
errors = [error_exact, error_rand, error_rand_power]
colors = ['blue', 'red', 'green']

ax4.bar(methods, errors, color=colors)
ax4.set_ylabel('Frobenius Error')
ax4.set_title('SVD Approximation Accuracy')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 8. PRACTICAL APPLICATIONS AND ML EXAMPLES
# ============================================================================

print("\n\n8. PRACTICAL APPLICATIONS AND ML EXAMPLES")
print("-" * 50)

def pca_with_randomized_svd(X, n_components=2):
    """
    Perform PCA using randomized SVD
    
    Parameters:
    -----------
    X : ndarray
        Data matrix (samples x features)
    n_components : int
        Number of principal components
        
    Returns:
    --------
    X_reduced : ndarray
        Reduced data
    explained_variance_ratio : ndarray
        Explained variance ratio for each component
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Randomized SVD
    U, s, Vt = randomized_svd(X_centered, n_components)
    
    # Project data
    X_reduced = X_centered @ Vt.T
    
    # Explained variance ratio
    explained_variance_ratio = s**2 / np.sum(s**2)
    
    return X_reduced, explained_variance_ratio

def sparse_linear_regression(X, y, alpha=0.1, max_iter=1000):
    """
    Sparse linear regression using iterative methods
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    alpha : float
        Regularization parameter
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    beta : ndarray
        Regression coefficients
    """
    n_samples, n_features = X.shape
    
    # Normal equation: (X^T X + alpha I) beta = X^T y
    A = X.T @ X + alpha * np.eye(n_features)
    b = X.T @ y
    
    # Solve using conjugate gradient
    beta, _, _ = conjugate_gradient_method(A, b, max_iter=max_iter)
    
    return beta

# Example: PCA on synthetic data
print("Example: PCA with Randomized SVD")
n_samples, n_features = 1000, 100
X = np.random.randn(n_samples, n_features)

# Add some structure
X[:, 0] = 2 * X[:, 1] + 0.1 * np.random.randn(n_samples)

X_reduced, explained_var = pca_with_randomized_svd(X, n_components=5)
print(f"Explained variance ratios: {explained_var[:3]}")

# Example: Sparse linear regression
print("\nExample: Sparse Linear Regression")
n_samples, n_features = 500, 100
X = np.random.randn(n_samples, n_features)
beta_true = np.zeros(n_features)
beta_true[:10] = np.random.randn(10)  # Only first 10 features matter
y = X @ beta_true + 0.1 * np.random.randn(n_samples)

beta_estimated = sparse_linear_regression(X, y, alpha=0.1)
print(f"True non-zero coefficients: {np.sum(beta_true != 0)}")
print(f"Estimated non-zero coefficients: {np.sum(np.abs(beta_estimated) > 0.01)}")

# ============================================================================
# 9. EXERCISES AND PRACTICE PROBLEMS
# ============================================================================

print("\n\n9. EXERCISES AND PRACTICE PROBLEMS")
print("-" * 50)

def exercise_1_condition_number():
    """Exercise 1: Condition Number Analysis"""
    print("Exercise 1: Condition Number Analysis")
    print("Create matrices with different condition numbers and analyze stability...")
    
    # Create matrices with known condition numbers
    n = 20
    
    # Well-conditioned matrix
    A_good = np.eye(n) + 0.1 * np.random.randn(n, n)
    
    # Ill-conditioned matrix
    U, _ = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.logspace(0, 8, n))  # Condition number ~10^8
    V, _ = np.linalg.qr(np.random.randn(n, n))
    A_bad = U @ S @ V.T
    
    # Test stability
    b = np.random.randn(n)
    
    for name, A in [("Well-conditioned", A_good), ("Ill-conditioned", A_bad)]:
        result = analyze_numerical_stability(A, b)
        print(f"\n{name} matrix:")
        print(f"  Condition number: {result['condition_number']:.2e}")
        print(f"  Error amplification: {result['amplification_factor']:.2e}")

def exercise_2_iterative_comparison():
    """Exercise 2: Iterative Methods Comparison"""
    print("\nExercise 2: Iterative Methods Comparison")
    print("Compare different iterative methods on various matrix types...")
    
    # Create different matrix types
    n = 100
    
    # Symmetric positive definite
    A_spd = np.random.randn(n, n)
    A_spd = A_spd.T @ A_spd + n * np.eye(n)
    
    # Diagonally dominant
    A_dd = np.random.randn(n, n)
    A_dd = A_dd + 2 * n * np.eye(n)
    
    b = np.random.randn(n)
    
    for name, A in [("SPD", A_spd), ("Diagonally Dominant", A_dd)]:
        print(f"\n{name} matrix:")
        
        x_jacobi, iter_jacobi, _ = jacobi_method(A, b)
        x_gs, iter_gs, _ = gauss_seidel_method(A, b)
        x_cg, iter_cg, _ = conjugate_gradient_method(A, b)
        
        x_exact = np.linalg.solve(A, b)
        
        print(f"  Jacobi: {iter_jacobi} iterations")
        print(f"  Gauss-Seidel: {iter_gs} iterations")
        print(f"  Conjugate Gradient: {iter_cg} iterations")

def exercise_3_sparse_operations():
    """Exercise 3: Sparse Matrix Operations"""
    print("\nExercise 3: Sparse Matrix Operations")
    print("Implement and test sparse matrix operations...")
    
    # Create sparse matrix
    n = 1000
    density = 0.01
    A_dense, data_csr, indices_csr, indptr_csr = create_sparse_matrix(n, density)
    
    # Test matrix-vector multiplication
    x = np.random.randn(n)
    
    # Dense multiplication
    start = time.time()
    y_dense = A_dense @ x
    time_dense = time.time() - start
    
    # Sparse multiplication
    start = time.time()
    y_sparse = csr_matvec(data_csr, indices_csr, indptr_csr, x)
    time_sparse = time.time() - start
    
    print(f"Matrix size: {n}x{n}")
    print(f"Sparsity: {density:.1%}")
    print(f"Dense time: {time_dense:.4f}s")
    print(f"Sparse time: {time_sparse:.4f}s")
    print(f"Speedup: {time_dense/time_sparse:.2f}x")
    print(f"Memory dense: {A_dense.nbytes} bytes")
    print(f"Memory sparse: {(data_csr.nbytes + indices_csr.nbytes + indptr_csr.nbytes)} bytes")

# Run exercises
exercise_1_condition_number()
exercise_2_iterative_comparison()
exercise_3_sparse_operations()

# ============================================================================
# 10. SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print("\n\n10. SUMMARY AND KEY TAKEAWAYS")
print("-" * 50)

print("""
Key Takeaways:

1. NUMERICAL STABILITY
   - Condition number measures problem sensitivity
   - Ill-conditioned problems amplify errors
   - Use stable algorithms for critical computations

2. ITERATIVE METHODS
   - Efficient for large, sparse systems
   - Conjugate gradient optimal for symmetric positive definite
   - Convergence depends on matrix properties

3. SPARSE MATRICES
   - CSR format efficient for matrix-vector operations
   - Memory savings significant for sparse matrices
   - Performance depends on sparsity pattern

4. EIGENVALUE COMPUTATION
   - Power iteration finds dominant eigenvalue
   - Inverse iteration finds eigenvalues near shift
   - QR algorithm finds all eigenvalues

5. RANDOMIZED METHODS
   - Provide good approximations for large matrices
   - Power iteration improves accuracy
   - Useful for PCA and other applications

6. PERFORMANCE CONSIDERATIONS
   - Choose algorithm based on matrix size and structure
   - Hardware considerations impact performance
   - Memory access patterns affect efficiency
""")

print("\n" + "=" * 60)
print("NUMERICAL LINEAR ALGEBRA COMPLETE")
print("=" * 60) 