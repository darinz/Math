# Matrix Decompositions

[![Chapter](https://img.shields.io/badge/Chapter-7-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Matrix_Decompositions-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Matrix decompositions are powerful tools that break down matrices into simpler, more manageable components. They are fundamental for solving systems of equations, understanding matrix structure, and implementing efficient algorithms in machine learning and data science.

## LU Decomposition

LU decomposition factors a matrix A into A = LU, where L is lower triangular and U is upper triangular.

### Mathematical Definition
A = LU where:
- L is lower triangular (Lᵢⱼ = 0 for i < j)
- U is upper triangular (Uᵢⱼ = 0 for i > j)

```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

def lu_decomposition_example():
    """Demonstrate LU decomposition"""
    A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    print("Matrix A:")
    print(A)
    
    # Perform LU decomposition
    P, L, U = lu(A)
    
    print("\nPermutation matrix P:")
    print(P)
    print("\nLower triangular matrix L:")
    print(L)
    print("\nUpper triangular matrix U:")
    print(U)
    
    # Verify decomposition
    A_reconstructed = P @ L @ U
    print("\nA reconstructed (P @ L @ U):")
    print(A_reconstructed)
    print(f"Decomposition error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return P, L, U

P, L, U = lu_decomposition_example()
```

### Solving Systems with LU
```python
def solve_with_lu(A, b):
    """Solve Ax = b using LU decomposition"""
    # Factor A
    lu_factor_result = lu_factor(A)
    
    # Solve system
    x = lu_solve(lu_factor_result, b)
    
    return x

# Example: Solve system of equations
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
b = np.array([5, -2, 9])

print("System Ax = b:")
print(f"A:\n{A}")
print(f"b: {b}")

x = solve_with_lu(A, b)
print(f"\nSolution x: {x}")

# Verify solution
b_computed = A @ x
print(f"Computed b: {b_computed}")
print(f"Verification error: {np.linalg.norm(b - b_computed):.2e}")
```

## QR Decomposition

QR decomposition factors a matrix A into A = QR, where Q is orthogonal and R is upper triangular.

### Mathematical Definition
A = QR where:
- Q is orthogonal (QᵀQ = I)
- R is upper triangular

```python
def qr_decomposition_example():
    """Demonstrate QR decomposition"""
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Matrix A:")
    print(A)
    
    # Perform QR decomposition
    Q, R = np.linalg.qr(A)
    
    print("\nOrthogonal matrix Q:")
    print(Q)
    print("\nUpper triangular matrix R:")
    print(R)
    
    # Verify orthogonality of Q
    Q_orthogonal = Q.T @ Q
    print("\nQ^T @ Q (should be identity):")
    print(Q_orthogonal)
    print(f"Orthogonality error: {np.linalg.norm(Q_orthogonal - np.eye(Q.shape[0])):.2e}")
    
    # Verify decomposition
    A_reconstructed = Q @ R
    print("\nA reconstructed (Q @ R):")
    print(A_reconstructed)
    print(f"Decomposition error: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    return Q, R

Q, R = qr_decomposition_example()
```

### Least Squares with QR
```python
def least_squares_qr(A, b):
    """Solve least squares problem using QR decomposition"""
    # QR decomposition
    Q, R = np.linalg.qr(A)
    
    # Solve R x = Q^T b
    Qtb = Q.T @ b
    x = np.linalg.solve(R, Qtb)
    
    return x

# Example: Overdetermined system
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([2, 3, 4, 5])

print("Overdetermined system Ax ≈ b:")
print(f"A:\n{A}")
print(f"b: {b}")

x_ls = least_squares_qr(A, b)
print(f"\nLeast squares solution: {x_ls}")

# Compare with numpy's least squares
x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Numpy least squares: {x_numpy}")
print(f"Difference: {np.linalg.norm(x_ls - x_numpy):.2e}")
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