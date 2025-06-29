# Matrix Decompositions

[![Chapter](https://img.shields.io/badge/Chapter-7-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Matrix_Decompositions-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Matrix decompositions are fundamental tools in linear algebra that break down complex matrices into simpler, more manageable components. These decompositions reveal the underlying structure of matrices and enable efficient algorithms for solving systems of equations, understanding data patterns, and implementing machine learning algorithms.

**Mathematical Foundation:**
Matrix decompositions express a matrix A as a product of simpler matrices:
A = B₁ × B₂ × ... × Bₖ

where each Bᵢ has a special structure (triangular, orthogonal, diagonal, etc.) that makes certain operations computationally efficient.

**Key Benefits:**
1. **Computational Efficiency**: Decompositions enable fast algorithms for solving systems, computing inverses, and finding eigenvalues
2. **Numerical Stability**: Some decompositions provide better numerical properties than direct methods
3. **Structural Insight**: Decompositions reveal important properties like rank, condition number, and geometric structure
4. **Machine Learning Applications**: Essential for dimensionality reduction, feature extraction, and model optimization

**Geometric Interpretation:**
Matrix decompositions can be viewed as coordinate transformations that reveal the "natural" structure of the data. For example:
- **LU decomposition**: Breaks down a matrix into elementary row operations
- **QR decomposition**: Orthogonalizes the columns of a matrix
- **SVD**: Finds the principal axes of variation in the data
- **Eigenvalue decomposition**: Diagonalizes a matrix in its eigenbasis

## LU Decomposition

LU decomposition factors a square matrix A into A = LU, where L is lower triangular and U is upper triangular. This decomposition is fundamental for solving systems of linear equations efficiently.

### Mathematical Foundation

**Definition:**
For a square matrix A ∈ ℝ^(n×n), the LU decomposition is:
A = LU

where:
- L ∈ ℝ^(n×n) is lower triangular (Lᵢⱼ = 0 for i < j)
- U ∈ ℝ^(n×n) is upper triangular (Uᵢⱼ = 0 for i > j)

**Existence and Uniqueness:**
- **Existence**: LU decomposition exists if and only if all leading principal minors of A are non-zero
- **Uniqueness**: If A is invertible, the LU decomposition is unique when L has ones on the diagonal
- **Generalization**: For any matrix, we can find A = PLU where P is a permutation matrix

**Geometric Interpretation:**
LU decomposition represents A as a sequence of elementary row operations:
1. **L matrix**: Represents the row operations needed to eliminate entries below the diagonal
2. **U matrix**: The resulting upper triangular form after elimination
3. **P matrix**: Represents row exchanges needed for numerical stability

**Key Properties:**
1. **Determinant**: det(A) = det(L) × det(U) = ∏ᵢ Lᵢᵢ × ∏ᵢ Uᵢᵢ
2. **Inverse**: A⁻¹ = U⁻¹L⁻¹ (if A is invertible)
3. **Linear Systems**: Ax = b becomes LUx = b, solved by forward/backward substitution

```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
import matplotlib.pyplot as plt

def lu_decomposition_detailed(A, pivot=True):
    """
    Perform detailed LU decomposition with comprehensive analysis
    
    Mathematical approach:
    A = PLU where P is permutation matrix, L is lower triangular, U is upper triangular
    
    Parameters:
    A: numpy array - square matrix to decompose
    pivot: bool - whether to use partial pivoting for numerical stability
    
    Returns:
    tuple - (P, L, U, analysis_results)
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square for LU decomposition")
    
    # Perform LU decomposition
    if pivot:
        P, L, U = lu(A)
    else:
        # Manual LU without pivoting (for educational purposes)
        P, L, U = lu_decomposition_manual(A)
    
    # Comprehensive analysis
    analysis = analyze_lu_decomposition(A, P, L, U)
    
    return P, L, U, analysis

def lu_decomposition_manual(A):
    """
    Manual LU decomposition without pivoting (educational implementation)
    
    Mathematical approach:
    Use Gaussian elimination to construct L and U matrices
    """
    n = A.shape[0]
    A_copy = A.copy().astype(float)
    L = np.eye(n)
    U = np.zeros_like(A_copy)
    
    for k in range(n):
        # Check if pivot is zero
        if abs(A_copy[k, k]) < 1e-10:
            raise ValueError("Zero pivot encountered - matrix may not have LU decomposition")
        
        # Store diagonal element in U
        U[k, k] = A_copy[k, k]
        
        # Eliminate column k
        for i in range(k+1, n):
            # Compute multiplier
            multiplier = A_copy[i, k] / A_copy[k, k]
            
            # Store multiplier in L
            L[i, k] = multiplier
            
            # Eliminate entry in A
            for j in range(k, n):
                A_copy[i, j] -= multiplier * A_copy[k, j]
        
        # Copy row k to U
        U[k, k:] = A_copy[k, k:]
    
    return np.eye(n), L, U  # No permutation matrix in manual version

def analyze_lu_decomposition(A, P, L, U):
    """
    Comprehensive analysis of LU decomposition
    
    Parameters:
    A: numpy array - original matrix
    P: numpy array - permutation matrix
    L: numpy array - lower triangular matrix
    U: numpy array - upper triangular matrix
    
    Returns:
    dict - analysis results
    """
    # Verify decomposition
    A_reconstructed = P @ L @ U
    decomposition_error = np.linalg.norm(A - A_reconstructed, 'fro')
    
    # Check triangular structure
    L_upper_entries = np.triu(L, k=1)
    U_lower_entries = np.tril(U, k=-1)
    L_triangular_error = np.linalg.norm(L_upper_entries)
    U_triangular_error = np.linalg.norm(U_lower_entries)
    
    # Check L has ones on diagonal (if applicable)
    L_diagonal = np.diag(L)
    L_diagonal_error = np.linalg.norm(L_diagonal - np.ones_like(L_diagonal))
    
    # Compute determinants
    det_A = np.linalg.det(A)
    det_L = np.linalg.det(L)
    det_U = np.linalg.det(U)
    det_P = np.linalg.det(P)
    det_product = det_P * det_L * det_U
    
    # Condition numbers
    cond_A = np.linalg.cond(A)
    cond_L = np.linalg.cond(L)
    cond_U = np.linalg.cond(U)
    
    # Growth factor (measure of numerical stability)
    max_U_element = np.max(np.abs(U))
    max_A_element = np.max(np.abs(A))
    growth_factor = max_U_element / max_A_element if max_A_element > 0 else float('inf')
    
    return {
        'decomposition_error': decomposition_error,
        'L_triangular_error': L_triangular_error,
        'U_triangular_error': U_triangular_error,
        'L_diagonal_error': L_diagonal_error,
        'det_A': det_A,
        'det_L': det_L,
        'det_U': det_U,
        'det_P': det_P,
        'det_product': det_product,
        'det_preserved': abs(det_A - det_product) < 1e-10,
        'cond_A': cond_A,
        'cond_L': cond_L,
        'cond_U': cond_U,
        'growth_factor': growth_factor,
        'numerically_stable': growth_factor < 100  # Heuristic threshold
    }

def solve_system_lu_detailed(A, b, method='scipy'):
    """
    Solve linear system Ax = b using LU decomposition with detailed analysis
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    method: str - 'scipy' or 'manual'
    
    Returns:
    tuple - (x, analysis_results)
    """
    if method == 'scipy':
        # Use scipy's optimized implementation
        lu_factor_result = lu_factor(A)
        x = lu_solve(lu_factor_result, b)
        
        # Extract L and U from lu_factor result
        L = np.tril(lu_factor_result[0])
        U = np.triu(lu_factor_result[0])
        np.fill_diagonal(L, 1.0)  # Ensure L has ones on diagonal
        
        analysis = analyze_lu_decomposition(A, np.eye(A.shape[0]), L, U)
        
    else:
        # Manual implementation
        P, L, U, analysis = lu_decomposition_detailed(A)
        
        # Solve PLUx = b
        # Step 1: Solve Py = b
        y1 = P.T @ b
        
        # Step 2: Solve Ly = y1 (forward substitution)
        y2 = np.linalg.solve(L, y1)
        
        # Step 3: Solve Ux = y2 (backward substitution)
        x = np.linalg.solve(U, y2)
    
    # Verify solution
    b_computed = A @ x
    residual = np.linalg.norm(b - b_computed)
    relative_residual = residual / np.linalg.norm(b) if np.linalg.norm(b) > 0 else residual
    
    analysis['solution_residual'] = residual
    analysis['relative_residual'] = relative_residual
    analysis['solution_accurate'] = relative_residual < 1e-10
    
    return x, analysis

def compare_lu_methods(A, b):
    """
    Compare different LU decomposition methods
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    
    Returns:
    dict - comparison results
    """
    methods = ['scipy_pivot', 'manual_no_pivot']
    results = {}
    
    for method in methods:
        try:
            if method == 'scipy_pivot':
                x, analysis = solve_system_lu_detailed(A, b, method='scipy')
            else:
                x, analysis = solve_system_lu_detailed(A, b, method='manual')
            
            results[method] = {
                'x': x,
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            results[method] = {
                'error': str(e),
                'success': False
            }
    
    return results

# Example: Comprehensive LU decomposition analysis
print("=== Comprehensive LU Decomposition Analysis ===")

# Test matrix
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
b = np.array([5, -2, 9], dtype=float)

print(f"Matrix A:\n{A}")
print(f"Right-hand side b: {b}")

# Perform detailed LU decomposition
P, L, U, analysis = lu_decomposition_detailed(A)

print(f"\nPermutation matrix P:\n{P}")
print(f"Lower triangular matrix L:\n{L}")
print(f"Upper triangular matrix U:\n{U}")

# Display analysis results
print(f"\nLU Decomposition Analysis:")
for key, value in analysis.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Verify decomposition
A_reconstructed = P @ L @ U
print(f"\nVerification:")
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Decomposition error: {analysis['decomposition_error']:.2e}")

# Solve system with detailed analysis
x, solution_analysis = solve_system_lu_detailed(A, b)

print(f"\nSystem Solution:")
print(f"x = {x}")
print(f"Solution residual: {solution_analysis['solution_residual']:.2e}")
print(f"Relative residual: {solution_analysis['relative_residual']:.2e}")
print(f"Solution accurate: {solution_analysis['solution_accurate']}")

# Verify solution
b_computed = A @ x
print(f"\nVerification:")
print(f"Original b: {b}")
print(f"Computed b: {b_computed}")
print(f"Verification error: {np.linalg.norm(b - b_computed):.2e}")

# Compare methods
print(f"\n=== Method Comparison ===")
comparison = compare_lu_methods(A, b)

for method, result in comparison.items():
    print(f"\n{method}:")
    if result['success']:
        analysis = result['analysis']
        print(f"  Success: {result['success']}")
        print(f"  Decomposition error: {analysis['decomposition_error']:.2e}")
        print(f"  Growth factor: {analysis['growth_factor']:.2f}")
        print(f"  Solution residual: {analysis['solution_residual']:.2e}")
    else:
        print(f"  Success: {result['success']}")
        print(f"  Error: {result['error']}")

# Test with different matrix types
print(f"\n=== Testing Different Matrix Types ===")

# Well-conditioned matrix
A_well = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=float)
b_well = np.array([1, 2, 3], dtype=float)

print(f"Well-conditioned matrix (condition number: {np.linalg.cond(A_well):.2f}):")
x_well, analysis_well = solve_system_lu_detailed(A_well, b_well)
print(f"  Solution residual: {analysis_well['solution_residual']:.2e}")

# Ill-conditioned matrix
A_ill = np.array([[1, 1], [1, 1.0001]], dtype=float)
b_ill = np.array([2, 2.0001], dtype=float)

print(f"Ill-conditioned matrix (condition number: {np.linalg.cond(A_ill):.2e}):")
x_ill, analysis_ill = solve_system_lu_detailed(A_ill, b_ill)
print(f"  Solution residual: {analysis_ill['solution_residual']:.2e}")
print(f"  Growth factor: {analysis_ill['growth_factor']:.2f}")

# Singular matrix (should fail)
A_singular = np.array([[1, 1], [1, 1]], dtype=float)
b_singular = np.array([1, 1], dtype=float)

print(f"Singular matrix:")
try:
    x_singular, analysis_singular = solve_system_lu_detailed(A_singular, b_singular)
    print(f"  Unexpected success!")
except Exception as e:
    print(f"  Expected failure: {e}")
```

### Solving Systems with LU Decomposition

**Mathematical Foundation:**
The LU decomposition enables efficient solution of linear systems Ax = b through forward and backward substitution:

1. **Decomposition**: A = LU
2. **Forward Substitution**: Solve Ly = b for y
3. **Backward Substitution**: Solve Ux = y for x

**Computational Complexity:**
- **Decomposition**: O(n³) operations (one-time cost)
- **Forward/Backward Substitution**: O(n²) operations per right-hand side
- **Multiple Right-hand Sides**: Very efficient after initial decomposition

**Numerical Stability:**
- **Partial Pivoting**: Exchanging rows to avoid small pivots
- **Growth Factor**: Measure of numerical stability
- **Condition Number**: Relationship between input and output perturbations

```python
def solve_multiple_systems_lu(A, B):
    """
    Solve multiple systems AX = B using LU decomposition
    
    Mathematical approach:
    After computing A = LU, solve LUx_i = b_i for each column b_i of B
    
    Parameters:
    A: numpy array - coefficient matrix (n × n)
    B: numpy array - right-hand side matrix (n × k)
    
    Returns:
    numpy array - solution matrix X (n × k)
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square")
    
    k = B.shape[1] if len(B.shape) > 1 else 1
    
    # Perform LU decomposition once
    lu_factor_result = lu_factor(A)
    
    # Solve for each right-hand side
    if k == 1:
        X = lu_solve(lu_factor_result, B)
    else:
        X = np.zeros((n, k))
        for i in range(k):
            X[:, i] = lu_solve(lu_factor_result, B[:, i])
    
    return X

def analyze_system_sensitivity(A, b, perturbation_magnitude=1e-6):
    """
    Analyze sensitivity of linear system solution to perturbations
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    perturbation_magnitude: float - magnitude of perturbations to test
    
    Returns:
    dict - sensitivity analysis results
    """
    # Solve original system
    x_original, _ = solve_system_lu_detailed(A, b)
    
    # Test perturbations in A
    A_perturbed = A + perturbation_magnitude * np.random.randn(*A.shape)
    x_A_perturbed, _ = solve_system_lu_detailed(A_perturbed, b)
    sensitivity_A = np.linalg.norm(x_original - x_A_perturbed) / perturbation_magnitude
    
    # Test perturbations in b
    b_perturbed = b + perturbation_magnitude * np.random.randn(*b.shape)
    x_b_perturbed, _ = solve_system_lu_detailed(A, b_perturbed)
    sensitivity_b = np.linalg.norm(x_original - x_b_perturbed) / perturbation_magnitude
    
    # Theoretical bounds
    cond_A = np.linalg.cond(A)
    norm_A_inv = np.linalg.norm(np.linalg.inv(A))
    norm_x = np.linalg.norm(x_original)
    norm_b = np.linalg.norm(b)
    
    return {
        'sensitivity_A': sensitivity_A,
        'sensitivity_b': sensitivity_b,
        'condition_number': cond_A,
        'norm_A_inv': norm_A_inv,
        'theoretical_bound_A': cond_A * norm_x / norm_A,
        'theoretical_bound_b': cond_A / norm_A,
        'well_conditioned': cond_A < 100
    }

# Example: Multiple systems and sensitivity analysis
print("\n=== Multiple Systems and Sensitivity Analysis ===")

# Create multiple right-hand sides
B = np.column_stack([
    np.array([5, -2, 9]),
    np.array([1, 0, 1]),
    np.array([0, 1, 0])
])

print(f"Multiple right-hand sides B:\n{B}")

# Solve all systems efficiently
X = solve_multiple_systems_lu(A, B)

print(f"Solution matrix X:\n{X}")

# Verify solutions
for i in range(B.shape[1]):
    b_i = B[:, i]
    x_i = X[:, i]
    residual = np.linalg.norm(A @ x_i - b_i)
    print(f"System {i+1} residual: {residual:.2e}")

# Sensitivity analysis
sensitivity_results = analyze_system_sensitivity(A, b)

print(f"\nSensitivity Analysis:")
for key, value in sensitivity_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Compare with theoretical bounds
print(f"\nSensitivity Comparison:")
print(f"  Observed sensitivity to A: {sensitivity_results['sensitivity_A']:.2e}")
print(f"  Theoretical bound for A: {sensitivity_results['theoretical_bound_A']:.2e}")
print(f"  Observed sensitivity to b: {sensitivity_results['sensitivity_b']:.2e}")
print(f"  Theoretical bound for b: {sensitivity_results['theoretical_bound_b']:.2e}")
```

## QR Decomposition

QR decomposition factors a matrix A into A = QR, where Q is orthogonal and R is upper triangular. This decomposition is fundamental for least squares problems, eigenvalue computation, and numerical stability.

### Mathematical Foundation

**Definition:**
For a matrix A ∈ ℝ^(m×n) with m ≥ n, the QR decomposition is:
A = QR

where:
- Q ∈ ℝ^(m×m) is orthogonal (QᵀQ = I)
- R ∈ ℝ^(m×n) is upper triangular (Rᵢⱼ = 0 for i > j)

**Existence and Uniqueness:**
- **Existence**: QR decomposition always exists for any matrix A
- **Uniqueness**: If A has full column rank, the decomposition is unique when R has positive diagonal entries
- **Reduced Form**: For m > n, we can write A = Q₁R₁ where Q₁ ∈ ℝ^(m×n) has orthonormal columns

**Geometric Interpretation:**
QR decomposition represents A as:
1. **Q matrix**: Orthonormal basis for the column space of A
2. **R matrix**: Coordinates of A's columns in the Q basis
3. **Gram-Schmidt Connection**: QR decomposition is essentially Gram-Schmidt orthogonalization applied to A's columns

**Key Properties:**
1. **Orthogonality**: QᵀQ = I (Q preserves lengths and angles)
2. **Upper Triangular**: R is upper triangular, enabling efficient back-substitution
3. **Rank Preservation**: rank(A) = rank(R) = number of non-zero diagonal elements of R
4. **Least Squares**: QR decomposition provides numerically stable solution to least squares problems

```python
def qr_decomposition_detailed(A, method='numpy'):
    """
    Perform detailed QR decomposition with comprehensive analysis
    
    Mathematical approach:
    A = QR where Q is orthogonal and R is upper triangular
    
    Parameters:
    A: numpy array - matrix to decompose
    method: str - 'numpy', 'gram_schmidt', or 'householder'
    
    Returns:
    tuple - (Q, R, analysis_results)
    """
    m, n = A.shape
    
    if method == 'numpy':
        # Use numpy's optimized implementation
        Q, R = np.linalg.qr(A)
        
    elif method == 'gram_schmidt':
        # Manual Gram-Schmidt implementation
        Q, R = qr_gram_schmidt(A)
        
    elif method == 'householder':
        # Manual Householder implementation
        Q, R = qr_householder(A)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Comprehensive analysis
    analysis = analyze_qr_decomposition(A, Q, R)
    
    return Q, R, analysis

def qr_gram_schmidt(A):
    """
    QR decomposition using Gram-Schmidt orthogonalization
    
    Mathematical approach:
    Apply Gram-Schmidt to columns of A to obtain Q, then compute R = Q^T A
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        # Start with column j of A
        v = A[:, j].copy()
        
        # Subtract projections onto previous orthogonal vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # Normalize to get next column of Q
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            # Handle zero or near-zero vectors
            Q[:, j] = np.zeros(m)
            R[j, j] = 0
    
    return Q, R

def qr_householder(A):
    """
    QR decomposition using Householder reflections
    
    Mathematical approach:
    Use Householder matrices to zero out subdiagonal elements
    """
    m, n = A.shape
    A_copy = A.copy().astype(float)
    Q = np.eye(m)
    
    for j in range(min(m-1, n)):
        # Extract column j from row j onwards
        x = A_copy[j:, j]
        
        # Compute Householder vector
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-10:
            # Choose sign to avoid cancellation
            if x[0] >= 0:
                x[0] += norm_x
            else:
                x[0] -= norm_x
            
            # Normalize Householder vector
            u = x / np.linalg.norm(x)
            
            # Apply Householder reflection to A
            A_copy[j:, j:] = A_copy[j:, j:] - 2 * np.outer(u, u.T @ A_copy[j:, j:])
            
            # Update Q matrix
            Q[j:, :] = Q[j:, :] - 2 * np.outer(u, u.T @ Q[j:, :])
    
    R = A_copy
    return Q.T, R

def analyze_qr_decomposition(A, Q, R):
    """
    Comprehensive analysis of QR decomposition
    
    Parameters:
    A: numpy array - original matrix
    Q: numpy array - orthogonal matrix
    R: numpy array - upper triangular matrix
    
    Returns:
    dict - analysis results
    """
    # Verify decomposition
    A_reconstructed = Q @ R
    decomposition_error = np.linalg.norm(A - A_reconstructed, 'fro')
    
    # Check orthogonality of Q
    Q_orthogonal_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 'fro')
    
    # Check upper triangular structure of R
    R_lower_entries = np.tril(R, k=-1)
    R_triangular_error = np.linalg.norm(R_lower_entries)
    
    # Check diagonal positivity of R (for uniqueness)
    R_diagonal = np.diag(R)
    negative_diagonal_count = np.sum(R_diagonal < -1e-10)
    
    # Rank analysis
    rank_A = np.linalg.matrix_rank(A)
    rank_R = np.sum(np.abs(R_diagonal) > 1e-10)
    
    # Condition numbers
    cond_A = np.linalg.cond(A)
    cond_R = np.linalg.cond(R)
    
    # Norm preservation
    norm_A = np.linalg.norm(A, 'fro')
    norm_QR = np.linalg.norm(Q @ R, 'fro')
    norm_preservation_error = abs(norm_A - norm_QR)
    
    return {
        'decomposition_error': decomposition_error,
        'Q_orthogonal_error': Q_orthogonal_error,
        'R_triangular_error': R_triangular_error,
        'negative_diagonal_count': negative_diagonal_count,
        'rank_A': rank_A,
        'rank_R': rank_R,
        'rank_preserved': rank_A == rank_R,
        'cond_A': cond_A,
        'cond_R': cond_R,
        'norm_preservation_error': norm_preservation_error,
        'numerically_stable': decomposition_error < 1e-10 and Q_orthogonal_error < 1e-10
    }

def solve_least_squares_qr_detailed(A, b, method='numpy'):
    """
    Solve least squares problem using QR decomposition with detailed analysis
    
    Mathematical approach:
    min ||Ax - b||₂ is solved by QR decomposition:
    1. A = QR
    2. min ||QRx - b||₂ = min ||Rx - Q^T b||₂
    3. Solve Rx = Q^T b by back-substitution
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    method: str - QR decomposition method
    
    Returns:
    tuple - (x, analysis_results)
    """
    m, n = A.shape
    
    # Perform QR decomposition
    Q, R, qr_analysis = qr_decomposition_detailed(A, method=method)
    
    # Solve least squares problem
    # Step 1: Compute Q^T b
    Qtb = Q.T @ b
    
    # Step 2: Solve Rx = Q^T b
    # Use only the first n rows of R and Q^T b
    R_n = R[:n, :n]
    Qtb_n = Qtb[:n]
    
    try:
        x = np.linalg.solve(R_n, Qtb_n)
        solution_exists = True
    except np.linalg.LinAlgError:
        # Handle rank-deficient case
        x = np.linalg.lstsq(R_n, Qtb_n, rcond=None)[0]
        solution_exists = False
    
    # Analysis
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    relative_residual = residual_norm / np.linalg.norm(b) if np.linalg.norm(b) > 0 else residual_norm
    
    # Check if solution is unique
    rank_A = qr_analysis['rank_A']
    solution_unique = rank_A == n
    
    # Compare with numpy's least squares
    x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
    numpy_residual = A @ x_numpy - b
    numpy_residual_norm = np.linalg.norm(numpy_residual)
    
    analysis = {
        'x': x,
        'residual_norm': residual_norm,
        'relative_residual': relative_residual,
        'solution_exists': solution_exists,
        'solution_unique': solution_unique,
        'rank_A': rank_A,
        'numpy_solution': x_numpy,
        'numpy_residual_norm': numpy_residual_norm,
        'solution_difference': np.linalg.norm(x - x_numpy),
        'qr_analysis': qr_analysis
    }
    
    return x, analysis

def compare_qr_methods(A, b):
    """
    Compare different QR decomposition methods for least squares
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    
    Returns:
    dict - comparison results
    """
    methods = ['numpy', 'gram_schmidt', 'householder']
    results = {}
    
    for method in methods:
        try:
            x, analysis = solve_least_squares_qr_detailed(A, b, method=method)
            results[method] = {
                'x': x,
                'analysis': analysis,
                'success': True
            }
        except Exception as e:
            results[method] = {
                'error': str(e),
                'success': False
            }
    
    return results

# Example: Comprehensive QR decomposition analysis
print("\n=== Comprehensive QR Decomposition Analysis ===")

# Test matrix (overdetermined system)
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
b = np.array([2, 3, 4, 5], dtype=float)

print(f"Matrix A:\n{A}")
print(f"Right-hand side b: {b}")
print(f"System shape: {A.shape}")

# Perform detailed QR decomposition
Q, R, analysis = qr_decomposition_detailed(A)

print(f"\nOrthogonal matrix Q:\n{Q}")
print(f"Upper triangular matrix R:\n{R}")

# Display analysis results
print(f"\nQR Decomposition Analysis:")
for key, value in analysis.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Verify decomposition
A_reconstructed = Q @ R
print(f"\nVerification:")
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Decomposition error: {analysis['decomposition_error']:.2e}")

# Verify orthogonality
Q_orthogonal = Q.T @ Q
print(f"\nOrthogonality check (Q^T @ Q):")
print(Q_orthogonal)
print(f"Orthogonality error: {analysis['Q_orthogonal_error']:.2e}")

# Solve least squares with detailed analysis
x, ls_analysis = solve_least_squares_qr_detailed(A, b)

print(f"\nLeast Squares Solution:")
print(f"x = {x}")
print(f"Residual norm: {ls_analysis['residual_norm']:.6f}")
print(f"Relative residual: {ls_analysis['relative_residual']:.6f}")
print(f"Solution unique: {ls_analysis['solution_unique']}")

# Compare with numpy
print(f"\nComparison with numpy:")
print(f"QR solution: {x}")
print(f"Numpy solution: {ls_analysis['numpy_solution']}")
print(f"Solution difference: {ls_analysis['solution_difference']:.2e}")

# Compare methods
print(f"\n=== Method Comparison ===")
comparison = compare_qr_methods(A, b)

for method, result in comparison.items():
    print(f"\n{method}:")
    if result['success']:
        analysis = result['analysis']
        print(f"  Success: {result['success']}")
        print(f"  Decomposition error: {analysis['qr_analysis']['decomposition_error']:.2e}")
        print(f"  Residual norm: {analysis['residual_norm']:.6f}")
        print(f"  Solution difference from numpy: {analysis['solution_difference']:.2e}")
    else:
        print(f"  Success: {result['success']}")
        print(f"  Error: {result['error']}")

# Test with different matrix types
print(f"\n=== Testing Different Matrix Types ===")

# Well-conditioned overdetermined system
A_well = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
b_well = np.array([1, 2, 3], dtype=float)

print(f"Well-conditioned system:")
x_well, analysis_well = solve_least_squares_qr_detailed(A_well, b_well)
print(f"  Residual norm: {analysis_well['residual_norm']:.6f}")
print(f"  Solution unique: {analysis_well['solution_unique']}")

# Ill-conditioned system
A_ill = np.array([[1, 1], [1, 1.0001], [1, 1.0002]], dtype=float)
b_ill = np.array([2, 2.0001, 2.0002], dtype=float)

print(f"Ill-conditioned system:")
x_ill, analysis_ill = solve_least_squares_qr_detailed(A_ill, b_ill)
print(f"  Residual norm: {analysis_ill['residual_norm']:.6f}")
print(f"  Condition number: {analysis_ill['qr_analysis']['cond_A']:.2e}")

# Rank-deficient system
A_rank_def = np.array([[1, 1], [1, 1], [1, 1]], dtype=float)
b_rank_def = np.array([1, 2, 3], dtype=float)

print(f"Rank-deficient system:")
x_rank_def, analysis_rank_def = solve_least_squares_qr_detailed(A_rank_def, b_rank_def)
print(f"  Residual norm: {analysis_rank_def['residual_norm']:.6f}")
print(f"  Rank: {analysis_rank_def['rank_A']}")
print(f"  Solution unique: {analysis_rank_def['solution_unique']}")

# Test with square system (should give exact solution)
print(f"\n=== Square System Test ===")
A_square = np.array([[2, 1], [1, 3]], dtype=float)
b_square = np.array([5, 6], dtype=float)

print(f"Square system:")
x_square, analysis_square = solve_least_squares_qr_detailed(A_square, b_square)
print(f"  Solution: {x_square}")
print(f"  Residual norm: {analysis_square['residual_norm']:.6f}")
print(f"  Should be exact: {analysis_square['residual_norm'] < 1e-10}")

# Verify exact solution
x_exact = np.linalg.solve(A_square, b_square)
print(f"  Exact solution: {x_exact}")
print(f"  Difference: {np.linalg.norm(x_square - x_exact):.2e}")
```

### Least Squares Applications

**Mathematical Foundation:**
QR decomposition provides a numerically stable method for solving least squares problems:

min ||Ax - b||₂

The solution is obtained by:
1. **Decomposition**: A = QR
2. **Transformation**: ||Ax - b||₂ = ||QRx - b||₂ = ||Rx - Qᵀb||₂
3. **Solution**: Solve Rx = Qᵀb by back-substitution

**Key Advantages:**
1. **Numerical Stability**: QR decomposition is more stable than normal equations
2. **Rank Deficiency**: Handles rank-deficient matrices gracefully
3. **Multiple Right-hand Sides**: Efficient for solving multiple least squares problems
4. **Conditioning**: Preserves conditioning better than other methods

```python
def solve_multiple_least_squares_qr(A, B):
    """
    Solve multiple least squares problems AX ≈ B using QR decomposition
    
    Mathematical approach:
    After computing A = QR, solve Rx_i = Q^T b_i for each column b_i of B
    
    Parameters:
    A: numpy array - coefficient matrix (m × n)
    B: numpy array - right-hand side matrix (m × k)
    
    Returns:
    numpy array - solution matrix X (n × k)
    """
    m, n = A.shape
    k = B.shape[1] if len(B.shape) > 1 else 1
    
    # Perform QR decomposition once
    Q, R = np.linalg.qr(A)
    
    # Solve for each right-hand side
    if k == 1:
        Qtb = Q.T @ B
        X = np.linalg.solve(R[:n, :n], Qtb[:n])
    else:
        Qtb = Q.T @ B
        X = np.zeros((n, k))
        for i in range(k):
            X[:, i] = np.linalg.solve(R[:n, :n], Qtb[:n, i])
    
    return X

def analyze_least_squares_sensitivity(A, b, perturbation_magnitude=1e-6):
    """
    Analyze sensitivity of least squares solution to perturbations
    
    Parameters:
    A: numpy array - coefficient matrix
    b: numpy array - right-hand side vector
    perturbation_magnitude: float - magnitude of perturbations to test
    
    Returns:
    dict - sensitivity analysis results
    """
    # Solve original problem
    x_original, _ = solve_least_squares_qr_detailed(A, b)
    
    # Test perturbations in A
    A_perturbed = A + perturbation_magnitude * np.random.randn(*A.shape)
    x_A_perturbed, _ = solve_least_squares_qr_detailed(A_perturbed, b)
    sensitivity_A = np.linalg.norm(x_original - x_A_perturbed) / perturbation_magnitude
    
    # Test perturbations in b
    b_perturbed = b + perturbation_magnitude * np.random.randn(*b.shape)
    x_b_perturbed, _ = solve_least_squares_qr_detailed(A, b_perturbed)
    sensitivity_b = np.linalg.norm(x_original - x_b_perturbed) / perturbation_magnitude
    
    # Theoretical bounds
    cond_A = np.linalg.cond(A)
    norm_A_pinv = np.linalg.norm(np.linalg.pinv(A))
    norm_x = np.linalg.norm(x_original)
    norm_b = np.linalg.norm(b)
    
    return {
        'sensitivity_A': sensitivity_A,
        'sensitivity_b': sensitivity_b,
        'condition_number': cond_A,
        'norm_A_pinv': norm_A_pinv,
        'theoretical_bound_A': cond_A * norm_x / norm_A,
        'theoretical_bound_b': cond_A / norm_A,
        'well_conditioned': cond_A < 100
    }

# Example: Multiple least squares and sensitivity analysis
print("\n=== Multiple Least Squares and Sensitivity Analysis ===")

# Create multiple right-hand sides
B = np.column_stack([
    np.array([2, 3, 4, 5]),
    np.array([1, 2, 3, 4]),
    np.array([0, 1, 2, 3])
])

print(f"Multiple right-hand sides B:\n{B}")

# Solve all least squares problems efficiently
X = solve_multiple_least_squares_qr(A, B)

print(f"Solution matrix X:\n{X}")

# Verify solutions
for i in range(B.shape[1]):
    b_i = B[:, i]
    x_i = X[:, i]
    residual = np.linalg.norm(A @ x_i - b_i)
    print(f"System {i+1} residual: {residual:.6f}")

# Sensitivity analysis
sensitivity_results = analyze_least_squares_sensitivity(A, b)

print(f"\nSensitivity Analysis:")
for key, value in sensitivity_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Compare with theoretical bounds
print(f"\nSensitivity Comparison:")
print(f"  Observed sensitivity to A: {sensitivity_results['sensitivity_A']:.2e}")
print(f"  Theoretical bound for A: {sensitivity_results['theoretical_bound_A']:.2e}")
print(f"  Observed sensitivity to b: {sensitivity_results['sensitivity_b']:.2e}")
print(f"  Theoretical bound for b: {sensitivity_results['theoretical_bound_b']:.2e}")
```

## Singular Value Decomposition (SVD)

SVD decomposes a matrix A into A = UΣVᵀ, where U and V are orthogonal and Σ is diagonal.

### Mathematical Definition
A = UΣVᵀ where:
- U is orthogonal (left singular vectors)
- Σ is diagonal (singular values)
- V is orthogonal (right singular vectors)

```python
def svd_decomposition_example():
    """Demonstrate SVD decomposition"""
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Matrix A:")
    print(A)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A)
    
    print("\nLeft singular vectors U:")
    print(U)
    print("\nSingular values S:")
    print(S)
    print("\nRight singular vectors V^T:")
    print(Vt)
    
    # Verify orthogonality
    U_orthogonal = U.T @ U
    V_orthogonal = Vt @ Vt.T
    print(f"\nU orthogonality error: {np.linalg.norm(U_orthogonal - np.eye(U.shape[0])):.2e}")
    print(f"V orthogonality error: {np.linalg.norm(V_orthogonal - np.eye(Vt.shape[1])):.2e}")
    
    # Reconstruct matrix
    Sigma = np.zeros_like(A, dtype=float)
    Sigma[:len(S), :len(S)] = np.diag(S)
    A_reconstructed = U @ Sigma @ Vt
    
    print("\nA reconstructed (U @ Σ @ V^T):")
    print(A_reconstructed)
    print(f"Decomposition error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return U, S, Vt

U, S, Vt = svd_decomposition_example()
```

### Low-Rank Approximation
```python
def low_rank_approximation(A, k):
    """Compute rank-k approximation of matrix A using SVD"""
    U, S, Vt = np.linalg.svd(A)
    
    # Keep only k singular values
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct rank-k approximation
    A_k = U_k @ np.diag(S_k) @ Vt_k
    
    return A_k, U_k, S_k, Vt_k

# Example: Image compression simulation
np.random.seed(42)
# Create a "low-rank" matrix (simulating an image)
n, m = 50, 50
A_original = np.random.randn(n, m)
# Make it approximately low-rank by adding some structure
A_original = A_original @ A_original.T / m

print(f"Original matrix shape: {A_original.shape}")
print(f"Original rank: {np.linalg.matrix_rank(A_original)}")

# Compute different rank approximations
ranks = [5, 10, 20]
for k in ranks:
    A_k, U_k, S_k, Vt_k = low_rank_approximation(A_original, k)
    
    # Calculate compression ratio
    original_size = A_original.size
    compressed_size = U_k.size + S_k.size + Vt_k.size
    compression_ratio = original_size / compressed_size
    
    # Calculate reconstruction error
    error = np.linalg.norm(A_original - A_k) / np.linalg.norm(A_original)
    
    print(f"\nRank {k} approximation:")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Relative error: {error:.4f}")
```

## Cholesky Decomposition

Cholesky decomposition factors a positive definite matrix A into A = LLᵀ, where L is lower triangular.

### Mathematical Definition
A = LLᵀ where:
- L is lower triangular
- A is symmetric positive definite

```python
def cholesky_decomposition_example():
    """Demonstrate Cholesky decomposition"""
    # Create a positive definite matrix
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    print("Positive definite matrix A:")
    print(A)
    
    # Verify positive definiteness
    eigenvals = np.linalg.eigvals(A)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Is positive definite: {np.all(eigenvals > 0)}")
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    print("\nLower triangular matrix L:")
    print(L)
    
    # Verify decomposition
    A_reconstructed = L @ L.T
    print("\nA reconstructed (L @ L^T):")
    print(A_reconstructed)
    print(f"Decomposition error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return L

L = cholesky_decomposition_example()
```

### Solving Systems with Cholesky
```python
def solve_with_cholesky(A, b):
    """Solve Ax = b using Cholesky decomposition"""
    # Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    # Solve L y = b (forward substitution)
    y = np.linalg.solve(L, b)
    
    # Solve L^T x = y (backward substitution)
    x = np.linalg.solve(L.T, y)
    
    return x

# Example: Solve system with positive definite matrix
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
b = np.array([1, 2, 3])

print("System Ax = b (A positive definite):")
print(f"A:\n{A}")
print(f"b: {b}")

x = solve_with_cholesky(A, b)
print(f"\nSolution x: {x}")

# Verify solution
b_computed = A @ x
print(f"Computed b: {b_computed}")
print(f"Verification error: {np.linalg.norm(b - b_computed):.2e}")
```

## Eigenvalue Decomposition

Eigenvalue decomposition factors a diagonalizable matrix A into A = PDP⁻¹, where P contains eigenvectors and D is diagonal.

### Mathematical Definition
A = PDP⁻¹ where:
- P contains eigenvectors as columns
- D is diagonal with eigenvalues
- A must be diagonalizable

```python
def eigenvalue_decomposition_example():
    """Demonstrate eigenvalue decomposition"""
    A = np.array([[4, -2], [1, 1]])
    print("Matrix A:")
    print(A)
    
    # Perform eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    print(f"\nEigenvalues: {eigenvals}")
    print("\nEigenvectors (columns):")
    print(eigenvecs)
    
    # Construct diagonal matrix
    D = np.diag(eigenvals)
    P = eigenvecs
    P_inv = np.linalg.inv(P)
    
    print("\nDiagonal matrix D:")
    print(D)
    print("\nEigenvector matrix P:")
    print(P)
    
    # Verify decomposition
    A_reconstructed = P @ D @ P_inv
    print("\nA reconstructed (P @ D @ P^(-1)):")
    print(A_reconstructed)
    print(f"Decomposition error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return eigenvals, eigenvecs

eigenvals, eigenvecs = eigenvalue_decomposition_example()
```

## Applications in Machine Learning

### Principal Component Analysis (PCA)
```python
def pca_with_svd(data, n_components=None):
    """Perform PCA using SVD"""
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # SVD of centered data
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    
    if n_components is None:
        n_components = len(S)
    
    # Principal components are right singular vectors
    principal_components = Vt[:n_components, :]
    
    # Project data onto principal components
    data_pca = U[:, :n_components] @ np.diag(S[:n_components])
    
    return data_pca, principal_components, S

# Example: PCA on synthetic data
np.random.seed(42)
n_samples, n_features = 100, 3
data = np.random.randn(n_samples, n_features)
# Add correlation
data[:, 2] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n_samples)

print(f"Original data shape: {data.shape}")

# Perform PCA
data_pca, components, singular_values = pca_with_svd(data, n_components=2)

print(f"PCA data shape: {data_pca.shape}")
print(f"Principal components shape: {components.shape}")

# Explained variance
explained_variance = singular_values**2 / (n_samples - 1)
total_variance = np.sum(explained_variance)
explained_variance_ratio = explained_variance / total_variance

print(f"\nExplained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio[:2]):
    print(f"PC{i+1}: {ratio:.3f}")
```

### Matrix Factorization for Recommender Systems
```python
def matrix_factorization(R, k, max_iter=100, learning_rate=0.01, reg=0.1):
    """Simple matrix factorization for recommender systems"""
    m, n = R.shape
    
    # Initialize factors
    U = np.random.randn(m, k) * 0.1
    V = np.random.randn(n, k) * 0.1
    
    # Find non-zero entries
    non_zero = R != 0
    
    for iteration in range(max_iter):
        # Compute prediction
        R_pred = U @ V.T
        
        # Compute error only for non-zero entries
        error = R_pred - R
        error[~non_zero] = 0
        
        # Update factors
        U_grad = error @ V + reg * U
        V_grad = error.T @ U + reg * V
        
        U -= learning_rate * U_grad
        V -= learning_rate * V_grad
        
        # Compute loss
        if iteration % 20 == 0:
            loss = np.sum(error**2) + reg * (np.sum(U**2) + np.sum(V**2))
            print(f"Iteration {iteration}, Loss: {loss:.4f}")
    
    return U, V

# Example: Simple recommender system
np.random.seed(42)
# Create rating matrix (users x items)
n_users, n_items = 10, 15
R = np.random.randint(1, 6, (n_users, n_items))
# Add some missing ratings
R[np.random.rand(n_users, n_items) < 0.3] = 0

print("Rating matrix (0 = missing rating):")
print(R)

# Perform matrix factorization
k = 3  # Number of latent factors
U, V = matrix_factorization(R, k, max_iter=100)

# Predict missing ratings
R_pred = U @ V.T
print(f"\nPredicted ratings:")
print(R_pred.round(2))

# Evaluate on non-zero entries
non_zero = R != 0
mse = np.mean((R_pred[non_zero] - R[non_zero])**2)
print(f"\nMean squared error on observed ratings: {mse:.4f}")
```

## Exercises

### Exercise 1: LU Decomposition
```python
# Perform LU decomposition on matrix
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])

# Your code here:
# 1. Perform LU decomposition
# 2. Solve the system Ax = [5, -2, 9] using LU
# 3. Verify the solution
```

### Exercise 2: QR Decomposition
```python
# Perform QR decomposition and least squares
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([2, 3, 4, 5])

# Your code here:
# 1. Perform QR decomposition
# 2. Solve least squares problem
# 3. Compare with numpy's lstsq
```

### Exercise 3: SVD and Low-Rank Approximation
```python
# Create a matrix and perform SVD
A = np.random.randn(10, 8)

# Your code here:
# 1. Perform SVD decomposition
# 2. Create rank-3 approximation
# 3. Calculate compression ratio and error
```

## Solutions

### Solution 1: LU Decomposition
```python
# 1. Perform LU decomposition
P, L, U = lu(A)
print("LU decomposition:")
print(f"L:\n{L}")
print(f"U:\n{U}")

# 2. Solve system using LU
b = np.array([5, -2, 9])
x = lu_solve(lu_factor(A), b)
print(f"\nSolution x: {x}")

# 3. Verify solution
b_computed = A @ x
print(f"Computed b: {b_computed}")
print(f"Verification error: {np.linalg.norm(b - b_computed):.2e}")
```

### Solution 2: QR Decomposition
```python
# 1. Perform QR decomposition
Q, R = np.linalg.qr(A)
print("QR decomposition:")
print(f"Q:\n{Q}")
print(f"R:\n{R}")

# 2. Solve least squares
Qtb = Q.T @ b
x = np.linalg.solve(R, Qtb)
print(f"\nLeast squares solution: {x}")

# 3. Compare with numpy
x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Numpy solution: {x_numpy}")
print(f"Difference: {np.linalg.norm(x - x_numpy):.2e}")
```

### Solution 3: SVD and Low-Rank Approximation
```python
# 1. Perform SVD
U, S, Vt = np.linalg.svd(A)
print(f"Singular values: {S}")

# 2. Create rank-3 approximation
k = 3
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]
A_k = U_k @ np.diag(S_k) @ Vt_k

# 3. Calculate metrics
original_size = A.size
compressed_size = U_k.size + S_k.size + Vt_k.size
compression_ratio = original_size / compressed_size
error = np.linalg.norm(A - A_k) / np.linalg.norm(A)

print(f"Compression ratio: {compression_ratio:.2f}x")
print(f"Relative error: {error:.4f}")
```

## Summary

In this chapter, we covered:
- LU decomposition for solving linear systems
- QR decomposition for least squares problems
- SVD for dimensionality reduction and matrix approximation
- Cholesky decomposition for positive definite matrices
- Eigenvalue decomposition for diagonalizable matrices
- Applications in PCA and recommender systems

Matrix decompositions are essential tools for understanding matrix structure and implementing efficient algorithms in machine learning.

## Next Steps

In the next chapter, we'll explore applications of linear algebra in machine learning, including linear regression, neural networks, and optimization. 