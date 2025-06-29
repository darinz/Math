# Numerical Linear Algebra

[![Chapter](https://img.shields.io/badge/Chapter-9-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Numerical_Linear_Algebra-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Numerical linear algebra deals with the practical implementation of linear algebra algorithms on computers, considering finite precision arithmetic and computational efficiency. This chapter covers numerical stability, conditioning, and efficient algorithms for large-scale problems.

## 1. Numerical Stability and Conditioning

### Condition Number
The condition number measures how sensitive a problem is to perturbations in the input data.

For a matrix $A$, the condition number is:
$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def condition_number(A):
    """Compute the condition number of a matrix"""
    try:
        A_inv = np.linalg.inv(A)
        cond = np.linalg.norm(A) * np.linalg.norm(A_inv)
        return cond
    except np.linalg.LinAlgError:
        return np.inf

# Example: Well-conditioned vs ill-conditioned matrices
A_well = np.array([[1, 0], [0, 1]])
A_ill = np.array([[1, 1], [1, 1.0001]])

print("Well-conditioned matrix:")
print(A_well)
print(f"Condition number: {condition_number(A_well):.2f}")

print("\nIll-conditioned matrix:")
print(A_ill)
print(f"Condition number: {condition_number(A_ill):.2f}")

# Test sensitivity to perturbations
b = np.array([1, 1])
x_exact = np.linalg.solve(A_ill, b)

# Add small perturbation to b
b_perturbed = b + np.array([1e-6, 1e-6])
x_perturbed = np.linalg.solve(A_ill, b_perturbed)

print(f"\nRelative change in b: {np.linalg.norm(b_perturbed - b) / np.linalg.norm(b):.2e}")
print(f"Relative change in x: {np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact):.2e}")
```

### Numerical Stability Examples

```python
def demonstrate_numerical_instability():
    """Demonstrate numerical instability in matrix operations"""
    
    # Hilbert matrix (classic example of ill-conditioned matrix)
    def hilbert_matrix(n):
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)
        return H
    
    # Test different sizes
    sizes = [5, 10, 15, 20]
    condition_numbers = []
    
    for n in sizes:
        H = hilbert_matrix(n)
        cond = condition_number(H)
        condition_numbers.append(cond)
        print(f"Hilbert matrix {n}x{n}: condition number = {cond:.2e}")
    
    # Visualize growth of condition number
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(sizes, condition_numbers, 'bo-')
    plt.xlabel('Matrix Size')
    plt.ylabel('Condition Number')
    plt.title('Condition Number Growth for Hilbert Matrix')
    plt.grid(True)
    
    # Test solving linear system
    n = 10
    H = hilbert_matrix(n)
    x_true = np.ones(n)
    b = H @ x_true
    
    # Solve with different methods
    x_lu = np.linalg.solve(H, b)
    x_qr = np.linalg.solve(H.T @ H, H.T @ b)  # Normal equations
    
    error_lu = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
    error_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)
    
    print(f"\nRelative error (LU): {error_lu:.2e}")
    print(f"Relative error (QR): {error_qr:.2e}")
    
    plt.subplot(1, 2, 2)
    methods = ['LU', 'QR']
    errors = [error_lu, error_qr]
    plt.bar(methods, errors)
    plt.ylabel('Relative Error')
    plt.title('Solution Error Comparison')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

demonstrate_numerical_instability()
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