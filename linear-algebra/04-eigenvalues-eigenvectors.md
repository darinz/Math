# Eigenvalues and Eigenvectors

[![Chapter](https://img.shields.io/badge/Chapter-4-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Eigenvalues_Eigenvectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Eigenvalues and eigenvectors are fundamental concepts in linear algebra that reveal the intrinsic properties of matrices. They are crucial for understanding matrix behavior, diagonalization, and applications in machine learning like Principal Component Analysis (PCA), spectral clustering, and dimensionality reduction. Eigenvalues and eigenvectors provide a "natural coordinate system" for understanding how matrices transform space.

### Why Eigenvalues and Eigenvectors Matter in AI/ML

1. **Dimensionality Reduction**: PCA uses eigenvectors to find the most important directions in data
2. **Spectral Clustering**: Uses eigenvectors of similarity matrices for clustering
3. **PageRank Algorithm**: Uses eigenvectors to rank web pages
4. **Neural Networks**: Weight matrices have eigenvalues that affect training dynamics
5. **Optimization**: Hessian eigenvalues determine convergence properties
6. **Signal Processing**: Fourier transforms and other spectral methods use eigenvectors

## What are Eigenvalues and Eigenvectors?

### Mathematical Definition

For a square matrix $A$ of size $n \times n$, a non-zero vector $\vec{v} \in \mathbb{R}^n$ is an **eigenvector** if:
$$A\vec{v} = \lambda\vec{v}$$

where $\lambda$ is a scalar called the **eigenvalue** corresponding to $\vec{v}$.

### Geometric Interpretation

Eigenvectors are special vectors that don't change direction when transformed by the matrix $A$. They only get scaled by their corresponding eigenvalue:

- If $\lambda > 0$: The eigenvector is stretched by factor $\lambda$
- If $\lambda < 0$: The eigenvector is stretched by factor $|\lambda|$ and flipped
- If $\lambda = 0$: The eigenvector is mapped to the zero vector
- If $|\lambda| > 1$: The eigenvector is expanded
- If $|\lambda| < 1$: The eigenvector is contracted

### Fundamental Properties

1. **Eigenvectors are not unique**: If $\vec{v}$ is an eigenvector, then $c\vec{v}$ is also an eigenvector for any scalar $c \neq 0$
2. **Eigenvalues are unique**: Each eigenvector corresponds to exactly one eigenvalue
3. **Eigenvectors can be complex**: Even for real matrices, eigenvalues and eigenvectors can be complex numbers
4. **Number of eigenvalues**: An $n \times n$ matrix has exactly $n$ eigenvalues (counting multiplicities)

```python
import numpy as np
import matplotlib.pyplot as plt

# Example matrix
A = np.array([[4, -2], [1, 1]])
print("Matrix A:")
print(A)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:")
print(eigenvectors)

# Verify the eigenvalue equation: Av = λv
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ eigenvector
    lambda_v = eigenvalue * eigenvector
    print(f"\nEigenvalue {i+1}: {eigenvalue:.4f}")
    print(f"Eigenvector {i+1}: {eigenvector}")
    print(f"A × v: {Av}")
    print(f"λ × v: {lambda_v}")
    print(f"Are they equal? {np.allclose(Av, lambda_v)}")
    print(f"Relative error: {np.linalg.norm(Av - lambda_v) / np.linalg.norm(lambda_v):.2e}")

# Geometric interpretation
print(f"\nGeometric interpretation:")
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
    if eigenvalue > 0:
        if abs(eigenvalue) > 1:
            print(f"Eigenvector {i+1}: Stretched by factor {abs(eigenvalue):.2f}")
        else:
            print(f"Eigenvector {i+1}: Contracted by factor {abs(eigenvalue):.2f}")
    else:
        print(f"Eigenvector {i+1}: Flipped and scaled by factor {abs(eigenvalue):.2f}")
```

## Finding Eigenvalues and Eigenvectors

### Characteristic Equation

The eigenvalues are solutions to the characteristic equation:
$$\det(A - \lambda I) = 0$$

This is a polynomial equation of degree $n$ in $\lambda$, called the characteristic polynomial.

### Step-by-Step Process

1. **Form the matrix** $A - \lambda I$
2. **Compute the determinant** $\det(A - \lambda I)$
3. **Set equal to zero** and solve for $\lambda$
4. **For each eigenvalue** $\lambda_i$, solve $(A - \lambda_i I)\vec{v} = \vec{0}$ for $\vec{v}$

```python
def find_eigenvalues_manually(A):
    """Find eigenvalues by solving the characteristic equation"""
    n = A.shape[0]
    
    # For 2x2 matrices, we can solve analytically
    if n == 2:
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]
        
        # Characteristic equation: λ² - (a+d)λ + (ad-bc) = 0
        trace = a + d
        det = a * d - b * c
        
        # Quadratic formula
        discriminant = trace**2 - 4 * det
        if discriminant >= 0:
            lambda1 = (trace + np.sqrt(discriminant)) / 2
            lambda2 = (trace - np.sqrt(discriminant)) / 2
            return np.array([lambda1, lambda2])
        else:
            # Complex eigenvalues
            real_part = trace / 2
            imag_part = np.sqrt(-discriminant) / 2
            return np.array([complex(real_part, imag_part), complex(real_part, -imag_part)])
    
    # For larger matrices, use numerical methods
    return np.linalg.eigvals(A)

def find_eigenvectors_manually(A, eigenvalues):
    """Find eigenvectors for given eigenvalues"""
    eigenvectors = []
    
    for eigenvalue in eigenvalues:
        # Solve (A - λI)v = 0
        B = A - eigenvalue * np.eye(A.shape[0])
        
        # Use SVD to find null space
        U, S, Vt = np.linalg.svd(B)
        
        # Find vectors in null space (corresponding to zero singular values)
        tol = 1e-10
        null_space_indices = np.where(S < tol)[0]
        
        if len(null_space_indices) > 0:
            # Take the first vector in the null space
            eigenvector = Vt[null_space_indices[0]]
            eigenvectors.append(eigenvector)
        else:
            # If no exact null space, take the vector with smallest singular value
            eigenvector = Vt[-1]
            eigenvectors.append(eigenvector)
    
    return np.array(eigenvectors).T

# Test manual eigenvalue/eigenvector calculation
A = np.array([[4, -2], [1, 1]])
print("Matrix A:")
print(A)

# Manual calculation
eigenvalues_manual = find_eigenvalues_manually(A)
eigenvectors_manual = find_eigenvectors_manually(A, eigenvalues_manual)

print(f"\nManual calculation:")
print(f"Eigenvalues: {eigenvalues_manual}")
print(f"Eigenvectors:")
print(eigenvectors_manual)

# Compare with NumPy
eigenvalues_numpy, eigenvectors_numpy = np.linalg.eig(A)
print(f"\nNumPy calculation:")
print(f"Eigenvalues: {eigenvalues_numpy}")
print(f"Eigenvectors:")
print(eigenvectors_numpy)

# Verify both methods give same results
print(f"\nEigenvalues match: {np.allclose(eigenvalues_manual, eigenvalues_numpy)}")
print(f"Eigenvectors match: {np.allclose(np.abs(eigenvectors_manual), np.abs(eigenvectors_numpy))}")
```

## Properties of Eigenvalues and Eigenvectors

### Basic Properties

1. **Trace and Sum**: $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$
2. **Determinant and Product**: $\det(A) = \prod_{i=1}^{n} \lambda_i$
3. **Powers**: If $\lambda$ is an eigenvalue of $A$, then $\lambda^k$ is an eigenvalue of $A^k$
4. **Inverse**: If $\lambda$ is an eigenvalue of $A$, then $\frac{1}{\lambda}$ is an eigenvalue of $A^{-1}$ (if $A$ is invertible)

```python
def verify_eigenvalue_properties(A):
    """Verify fundamental eigenvalue properties"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Eigenvalue Properties Verification:")
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    
    # Property 1: Trace equals sum of eigenvalues
    trace_A = np.trace(A)
    sum_eigenvalues = np.sum(eigenvalues)
    print(f"\n1. Trace property:")
    print(f"   tr(A) = {trace_A:.4f}")
    print(f"   Sum of eigenvalues = {sum_eigenvalues:.4f}")
    print(f"   Match: {np.isclose(trace_A, sum_eigenvalues)}")
    
    # Property 2: Determinant equals product of eigenvalues
    det_A = np.linalg.det(A)
    prod_eigenvalues = np.prod(eigenvalues)
    print(f"\n2. Determinant property:")
    print(f"   det(A) = {det_A:.4f}")
    print(f"   Product of eigenvalues = {prod_eigenvalues:.4f}")
    print(f"   Match: {np.isclose(det_A, prod_eigenvalues)}")
    
    # Property 3: Powers of eigenvalues
    A_squared = A @ A
    eigenvalues_squared, _ = np.linalg.eig(A_squared)
    eigenvalues_powered = eigenvalues**2
    print(f"\n3. Power property:")
    print(f"   Eigenvalues of A²: {eigenvalues_squared}")
    print(f"   Eigenvalues²: {eigenvalues_powered}")
    print(f"   Match: {np.allclose(eigenvalues_squared, eigenvalues_powered)}")
    
    # Property 4: Inverse eigenvalues (if A is invertible)
    if np.linalg.det(A) != 0:
        A_inv = np.linalg.inv(A)
        eigenvalues_inv, _ = np.linalg.eig(A_inv)
        eigenvalues_reciprocal = 1 / eigenvalues
        print(f"\n4. Inverse property:")
        print(f"   Eigenvalues of A⁻¹: {eigenvalues_inv}")
        print(f"   1/eigenvalues: {eigenvalues_reciprocal}")
        print(f"   Match: {np.allclose(eigenvalues_inv, eigenvalues_reciprocal)}")
    
    return eigenvalues, eigenvectors

# Test with different matrices
A1 = np.array([[2, 1], [1, 3]])
A2 = np.array([[4, -2], [1, 1]])
A3 = np.array([[1, 2], [3, 4]])

for i, A in enumerate([A1, A2, A3]):
    print(f"\n{'='*50}")
    print(f"Matrix {i+1}:")
    verify_eigenvalue_properties(A)
```

### Eigenvalues of Special Matrices

```python
def analyze_special_matrices():
    """Analyze eigenvalues of special matrix types"""
    
    # Identity matrix
    I = np.eye(3)
    eigenvals_I = np.linalg.eigvals(I)
    print("Identity matrix eigenvalues:")
    print(f"Matrix:\n{I}")
    print(f"Eigenvalues: {eigenvals_I}")
    print(f"All eigenvalues = 1: {np.allclose(eigenvals_I, 1)}")
    
    # Diagonal matrix
    D = np.diag([1, 2, 3])
    eigenvals_D = np.linalg.eigvals(D)
    print(f"\nDiagonal matrix eigenvalues:")
    print(f"Matrix:\n{D}")
    print(f"Eigenvalues: {eigenvals_D}")
    print(f"Eigenvalues = diagonal elements: {np.allclose(eigenvals_D, [1, 2, 3])}")
    
    # Triangular matrix
    T = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    eigenvals_T = np.linalg.eigvals(T)
    print(f"\nTriangular matrix eigenvalues:")
    print(f"Matrix:\n{T}")
    print(f"Eigenvalues: {eigenvals_T}")
    print(f"Eigenvalues = diagonal elements: {np.allclose(eigenvals_T, [1, 4, 6])}")
    
    # Symmetric matrix
    S = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
    eigenvals_S, eigenvecs_S = np.linalg.eigh(S)  # Use eigh for symmetric matrices
    print(f"\nSymmetric matrix eigenvalues:")
    print(f"Matrix:\n{S}")
    print(f"Eigenvalues: {eigenvals_S}")
    print(f"All eigenvalues real: {np.all(np.isreal(eigenvals_S))}")
    
    # Orthogonal matrix (rotation)
    theta = np.pi/4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    eigenvals_R = np.linalg.eigvals(R)
    print(f"\nOrthogonal matrix (rotation) eigenvalues:")
    print(f"Matrix:\n{R}")
    print(f"Eigenvalues: {eigenvals_R}")
    print(f"Magnitude of eigenvalues = 1: {np.allclose(np.abs(eigenvals_R), 1)}")
    
    # Nilpotent matrix
    N = np.array([[0, 1], [0, 0]])
    eigenvals_N = np.linalg.eigvals(N)
    print(f"\nNilpotent matrix eigenvalues:")
    print(f"Matrix:\n{N}")
    print(f"Eigenvalues: {eigenvals_N}")
    print(f"All eigenvalues = 0: {np.allclose(eigenvals_N, 0)}")

analyze_special_matrices()
```

## Diagonalization

### Mathematical Definition

A matrix $A$ is **diagonalizable** if it can be written as:
$$A = PDP^{-1}$$

where:
- $P$ is an invertible matrix whose columns are eigenvectors of $A$
- $D$ is a diagonal matrix whose diagonal elements are the corresponding eigenvalues
- $P^{-1}$ is the inverse of $P$

### Conditions for Diagonalization

A matrix is diagonalizable if and only if:
1. It has $n$ linearly independent eigenvectors (where $n$ is the size of the matrix)
2. The geometric multiplicity equals the algebraic multiplicity for each eigenvalue

### Geometric Interpretation

Diagonalization represents a change of basis to the "eigenvector basis" where the matrix becomes diagonal. This makes many operations much simpler:

- **Powers**: $A^k = PD^kP^{-1}$
- **Exponential**: $e^A = Pe^DP^{-1}$
- **Functions**: $f(A) = Pf(D)P^{-1}$

```python
def analyze_diagonalization(A):
    """Analyze diagonalization of a matrix"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Diagonalization Analysis:")
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Check if matrix is diagonalizable
    det_P = np.linalg.det(eigenvectors)
    is_diagonalizable = abs(det_P) > 1e-10
    
    print(f"\nDeterminant of eigenvector matrix: {det_P:.6f}")
    print(f"Matrix is diagonalizable: {is_diagonalizable}")
    
    if is_diagonalizable:
        # Perform diagonalization
        P = eigenvectors
        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)
        
        # Reconstruct A
        A_reconstructed = P @ D @ P_inv
        
        print(f"\nDiagonalization:")
        print(f"P (eigenvectors):\n{P}")
        print(f"D (eigenvalues):\n{D}")
        print(f"P⁻¹:\n{P_inv}")
        print(f"A reconstructed: P @ D @ P⁻¹\n{A_reconstructed}")
        print(f"Original A:\n{A}")
        print(f"Reconstruction successful: {np.allclose(A, A_reconstructed)}")
        
        # Demonstrate power calculation
        k = 3
        A_power_direct = np.linalg.matrix_power(A, k)
        A_power_diagonal = P @ np.linalg.matrix_power(D, k) @ P_inv
        
        print(f"\nPower calculation (A^{k}):")
        print(f"Direct calculation:\n{A_power_direct}")
        print(f"Using diagonalization:\n{A_power_diagonal}")
        print(f"Results match: {np.allclose(A_power_direct, A_power_diagonal)}")
        
        return P, D, P_inv
    else:
        print("Matrix is not diagonalizable")
        return None, None, None

# Test diagonalization with different matrices
A_diagonalizable = np.array([[4, -2], [1, 1]])
A_non_diagonalizable = np.array([[1, 1], [0, 1]])  # Jordan block

print("="*60)
print("Diagonalizable matrix:")
P1, D1, P1_inv = analyze_diagonalization(A_diagonalizable)

print("\n" + "="*60)
print("Non-diagonalizable matrix:")
P2, D2, P2_inv = analyze_diagonalization(A_non_diagonalizable)
```

## Power Method

The power method is an iterative algorithm to find the dominant eigenvalue and eigenvector of a matrix.

### Algorithm

1. Start with a random vector $\vec{v}_0$
2. Iterate: $\vec{v}_{k+1} = \frac{A\vec{v}_k}{\|A\vec{v}_k\|}$
3. The eigenvalue is approximated by: $\lambda \approx \frac{\vec{v}_k^T A \vec{v}_k}{\vec{v}_k^T \vec{v}_k}$

### Convergence

The power method converges to the eigenvalue with the largest magnitude, provided:
- The matrix has a unique dominant eigenvalue
- The initial vector has a non-zero component in the direction of the dominant eigenvector

```python
def power_method(A, max_iter=100, tol=1e-6, verbose=True):
    """Power method to find dominant eigenvalue and eigenvector"""
    n = A.shape[0]
    
    # Initialize random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenvalues_history = []
    eigenvectors_history = []
    
    for i in range(max_iter):
        v_old = v.copy()
        
        # Apply matrix transformation
        v_new = A @ v
        
        # Normalize
        v = v_new / np.linalg.norm(v_new)
        
        # Calculate eigenvalue approximation
        eigenvalue = (v.T @ A @ v) / (v.T @ v)
        eigenvalues_history.append(eigenvalue)
        eigenvectors_history.append(v.copy())
        
        # Check convergence
        if np.linalg.norm(v - v_old) < tol:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break
    
    if verbose:
        print(f"Final eigenvalue: {eigenvalue:.6f}")
        print(f"Final eigenvector: {v}")
    
    return eigenvalue, v, eigenvalues_history, eigenvectors_history

def inverse_power_method(A, shift=0, max_iter=100, tol=1e-6):
    """Inverse power method to find eigenvalue closest to shift"""
    n = A.shape[0]
    
    # Initialize random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    # Shift the matrix
    A_shifted = A - shift * np.eye(n)
    
    for i in range(max_iter):
        v_old = v.copy()
        
        # Solve linear system (A - shift*I)v_new = v
        v_new = np.linalg.solve(A_shifted, v)
        
        # Normalize
        v = v_new / np.linalg.norm(v_new)
        
        # Check convergence
        if np.linalg.norm(v - v_old) < tol:
            break
    
    # Calculate eigenvalue
    eigenvalue = (v.T @ A @ v) / (v.T @ v)
    
    return eigenvalue, v

# Test power method
A = np.array([[4, -2], [1, 1]])
print("Testing Power Method:")
print(f"Matrix A:\n{A}")

# Power method
dominant_eigenvalue, dominant_eigenvector, history_eigenvalues, history_eigenvectors = power_method(A)

# Compare with exact values
exact_eigenvalues, exact_eigenvectors = np.linalg.eig(A)
dominant_exact_idx = np.argmax(np.abs(exact_eigenvalues))
exact_dominant_eigenvalue = exact_eigenvalues[dominant_exact_idx]
exact_dominant_eigenvector = exact_eigenvectors[:, dominant_exact_idx]

print(f"\nComparison with exact values:")
print(f"Power method eigenvalue: {dominant_eigenvalue:.6f}")
print(f"Exact dominant eigenvalue: {exact_dominant_eigenvalue:.6f}")
print(f"Relative errors: {abs(dominant_eigenvalue - exact_dominant_eigenvalue) / abs(exact_dominant_eigenvalue):.2e}")

print(f"\nPower method eigenvector: {dominant_eigenvector}")
print(f"Exact dominant eigenvector: {exact_dominant_eigenvector}")
print(f"Eigenvector alignment: {abs(np.dot(dominant_eigenvector, exact_dominant_eigenvector)):.6f}")

# Test inverse power method
smallest_eigenvalue, smallest_eigenvector = inverse_power_method(A)
smallest_exact_idx = np.argmin(np.abs(exact_eigenvalues))
exact_smallest_eigenvalue = exact_eigenvalues[smallest_exact_idx]

print(f"\nInverse power method:")
print(f"Smallest eigenvalue: {smallest_eigenvalue:.6f}")
```

## Applications in Machine Learning

### Principal Component Analysis (PCA)

PCA uses eigenvectors of the covariance matrix to find the principal components (directions of maximum variance).

```python
def pca_analysis():
    """Demonstrate PCA using eigenvalues and eigenvectors"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create correlated data
    mean = [0, 0, 0]
    cov = [[1, 0.8, 0.6], 
           [0.8, 1, 0.7], 
           [0.6, 0.7, 1]]
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    print("PCA Analysis:")
    print(f"Data shape: {data.shape}")
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print(f"\nEigenvalues (variances): {eigenvalues}")
    print(f"Eigenvectors (principal components):\n{eigenvectors}")
    
    # Explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"\nExplained variance ratio: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_variance_ratio}")
    
    # Project data onto principal components
    data_pca = data_centered @ eigenvectors
    
    print(f"\nTransformed data shape: {data_pca.shape}")
    print(f"Variance in each component: {np.var(data_pca, axis=0)}")
    
    # Verify that variances match eigenvalues
    print(f"Variance verification: {np.allclose(np.var(data_pca, axis=0), eigenvalues)}")
    
    return data, data_pca, eigenvectors, eigenvalues

data, data_pca, eigenvectors, eigenvalues = pca_analysis()
```

### Spectral Clustering

Spectral clustering uses eigenvectors of the Laplacian matrix to perform clustering.

```python
def spectral_clustering_demo():
    """Demonstrate spectral clustering using eigenvalues/eigenvectors"""
    
    # Generate sample data with clear clusters
    np.random.seed(42)
    n_samples = 200
    
    # Create two clusters
    cluster1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
    cluster2 = np.random.multivariate_normal([4, 4], [[1, -0.3], [-0.3, 1]], n_samples//2)
    
    data = np.vstack([cluster1, cluster2])
    
    # Compute similarity matrix (Gaussian kernel)
    def gaussian_kernel(X, sigma=1.0):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                K[i, j] = np.exp(-np.dot(diff, diff) / (2 * sigma**2))
        return K
    
    similarity_matrix = gaussian_kernel(data, sigma=1.0)
    
    # Compute Laplacian matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix
    
    # Eigendecomposition of Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print("Spectral Clustering Analysis:")
    print(f"Data shape: {data.shape}")
    print(f"Number of clusters: 2")
    print(f"Laplacian eigenvalues: {eigenvalues[:5]}")  # Show first 5
    
    # Use second eigenvector for clustering (first is constant)
    cluster_indicator = eigenvectors[:, 1]
    
    # Simple clustering based on sign
    clusters = (cluster_indicator > 0).astype(int)
    
    print(f"\nClustering results:")
    print(f"Cluster sizes: {np.bincount(clusters)}")
    
    return data, clusters, eigenvalues, eigenvectors

data_clusters, clusters, eigenvalues_laplacian, eigenvectors_laplacian = spectral_clustering_demo()
```

### PageRank Algorithm

PageRank uses the dominant eigenvector of the transition matrix to rank web pages.

```python
def pagerank_demo():
    """Demonstrate PageRank algorithm using eigenvalues/eigenvectors"""
    
    # Simple web graph (adjacency matrix)
    # Pages: A, B, C, D
    # A links to B, C
    # B links to C
    # C links to A
    # D links to A, C
    
    adjacency_matrix = np.array([
        [0, 1, 1, 0],  # A
        [0, 0, 1, 0],  # B
        [1, 0, 0, 0],  # C
        [1, 0, 1, 0]   # D
    ])
    
    # Create transition matrix
    out_degrees = np.sum(adjacency_matrix, axis=1)
    transition_matrix = np.zeros_like(adjacency_matrix, dtype=float)
    
    for i in range(adjacency_matrix.shape[0]):
        if out_degrees[i] > 0:
            transition_matrix[i] = adjacency_matrix[i] / out_degrees[i]
        else:
            # Handle dangling nodes (pages with no outlinks)
            transition_matrix[i] = 1.0 / adjacency_matrix.shape[0]
    
    # Add damping factor (teleportation)
    damping = 0.85
    n_pages = adjacency_matrix.shape[0]
    transition_matrix = damping * transition_matrix + (1 - damping) / n_pages
    
    print("PageRank Analysis:")
    print(f"Transition matrix:\n{transition_matrix}")
    
    # Find dominant eigenvector (PageRank scores)
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
    
    # Find eigenvector corresponding to eigenvalue 1
    dominant_idx = np.argmin(np.abs(eigenvalues - 1))
    pagerank_scores = np.real(eigenvectors[:, dominant_idx])
    pagerank_scores = pagerank_scores / np.sum(pagerank_scores)  # Normalize
    
    print(f"\nPageRank scores:")
    pages = ['A', 'B', 'C', 'D']
    for page, score in zip(pages, pagerank_scores):
        print(f"Page {page}: {score:.4f}")
    
    # Rank pages
    page_rankings = np.argsort(pagerank_scores)[::-1]
    print(f"\nPage rankings (highest to lowest):")
    for i, rank in enumerate(page_rankings):
        print(f"{i+1}. Page {pages[rank]} (score: {pagerank_scores[rank]:.4f})")
    
    return transition_matrix, pagerank_scores

transition_matrix, pagerank_scores = pagerank_demo()
```

## Visualization of Eigenvalues and Eigenvectors

```python
def visualize_eigenvalues_eigenvectors():
    """Visualize eigenvalues and eigenvectors"""
    
    # Create a 2x2 matrix
    A = np.array([[3, 1], [1, 2]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original vectors and their transformations
    ax1 = axes[0]
    
    # Unit vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transform unit vectors
    Ae1 = A @ e1
    Ae2 = A @ e2
    
    # Plot original unit vectors
    ax1.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', alpha=0.7, label='e₁', width=0.02)
    ax1.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
              color='green', alpha=0.7, label='e₂', width=0.02)
    
    # Plot transformed unit vectors
    ax1.quiver(0, 0, Ae1[0], Ae1[1], angles='xy', scale_units='xy', scale=1, 
              color='red', alpha=0.7, label='Ae₁', width=0.02)
    ax1.quiver(0, 0, Ae2[0], Ae2[1], angles='xy', scale_units='xy', scale=1, 
              color='orange', alpha=0.7, label='Ae₂', width=0.02)
    
    # Plot eigenvectors
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Normalize eigenvector for visualization
        eigenvector_norm = eigenvector / np.linalg.norm(eigenvector)
        ax1.quiver(0, 0, eigenvector_norm[0], eigenvector_norm[1], angles='xy', scale_units='xy', scale=1, 
                  color='purple', alpha=0.9, label=f'Eigenvector {i+1} (λ={eigenvalue:.2f})', width=0.03)
        
        # Show eigenvalue scaling
        scaled_eigenvector = eigenvalue * eigenvector_norm
        ax1.quiver(0, 0, scaled_eigenvector[0], scaled_eigenvector[1], angles='xy', scale_units='xy', scale=1, 
                  color='magenta', alpha=0.7, label=f'λ{i+1}×eigenvector {i+1}', width=0.02)
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Matrix Transformation and Eigenvectors')
    ax1.legend()
    
    # Plot 2: Eigenvalue spectrum
    ax2 = axes[1]
    
    # Plot eigenvalues in complex plane
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    ax2.scatter(real_parts, imag_parts, c=['red', 'blue'], s=100, alpha=0.7)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax2.plot(circle_x, circle_y, 'k--', alpha=0.3, label='Unit circle')
    
    # Label eigenvalues
    for i, (real, imag) in enumerate(zip(real_parts, imag_parts)):
        ax2.annotate(f'λ{i+1}={real:.2f}+{imag:.2f}i', 
                    (real, imag), xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('Eigenvalue Spectrum')
    ax2.set_xlabel('Real part')
    ax2.set_ylabel('Imaginary part')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

visualize_eigenvalues_eigenvectors()
```

## Exercises

### Exercise 1: Eigenvalue Properties
```python
# Verify eigenvalue properties for different matrices
def exercise_eigenvalue_properties():
    matrices = {
        "Symmetric": np.array([[2, 1], [1, 3]]),
        "Skew-symmetric": np.array([[0, 1], [-1, 0]]),
        "Triangular": np.array([[1, 2], [0, 3]]),
        "Random": np.random.randn(2, 2)
    }
    
    for name, A in matrices.items():
        print(f"\n{name} matrix:")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Verify trace property
        trace_property = np.isclose(np.trace(A), np.sum(eigenvalues))
        print(f"  Trace property: {trace_property}")
        
        # Verify determinant property
        det_property = np.isclose(np.linalg.det(A), np.prod(eigenvalues))
        print(f"  Determinant property: {det_property}")
        
        # Check if eigenvalues are real
        real_eigenvalues = np.all(np.isreal(eigenvalues))
        print(f"  Real eigenvalues: {real_eigenvalues}")

exercise_eigenvalue_properties()
```

### Exercise 2: Power Method Implementation
```python
# Implement and test power method variations
def exercise_power_method():
    A = np.array([[4, -2], [1, 1]])
    
    # Standard power method
    lambda1, v1, _, _ = power_method(A, verbose=False)
    
    # Inverse power method
    lambda2, v2 = inverse_power_method(A, shift=0)
    
    # Exact values
    exact_eigenvalues, exact_eigenvectors = np.linalg.eig(A)
    
    print("Power Method Exercise:")
    print(f"Matrix A:\n{A}")
    print(f"Exact eigenvalues: {exact_eigenvalues}")
    print(f"Power method (largest): {lambda1:.6f}")
    print(f"Inverse power method (smallest): {lambda2:.6f}")
    print(f"Relative errors: {abs(lambda1 - max(exact_eigenvalues)) / abs(max(exact_eigenvalues)):.2e}, {abs(lambda2 - min(exact_eigenvalues)) / abs(min(exact_eigenvalues)):.2e}")

exercise_power_method()
```

### Exercise 3: Diagonalization
```python
# Test diagonalization with different matrices
def exercise_diagonalization():
    matrices = {
        "Diagonalizable": np.array([[4, -2], [1, 1]]),
        "Non-diagonalizable": np.array([[1, 1], [0, 1]]),
        "Symmetric": np.array([[2, 1], [1, 3]])
    }
    
    for name, A in matrices.items():
        print(f"\n{name} matrix:")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        det_P = np.linalg.det(eigenvectors)
        
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Det(P): {det_P:.6f}")
        print(f"  Diagonalizable: {abs(det_P) > 1e-10}")
        
        if abs(det_P) > 1e-10:
            P = eigenvectors
            D = np.diag(eigenvalues)
            P_inv = np.linalg.inv(P)
            A_reconstructed = P @ D @ P_inv
            reconstruction_success = np.allclose(A, A_reconstructed)
            print(f"  Reconstruction successful: {reconstruction_success}")

exercise_diagonalization()
```

## Summary

In this chapter, we've covered:

1. **Eigenvalue/Eigenvector Fundamentals**: Definition, geometric interpretation, and mathematical properties
2. **Finding Eigenvalues/Eigenvectors**: Characteristic equation, manual calculation, and numerical methods
3. **Properties**: Trace, determinant, powers, and special matrix properties
4. **Diagonalization**: Conditions, process, and applications for simplifying matrix operations
5. **Power Method**: Iterative algorithm for finding dominant eigenvalues and eigenvectors
6. **AI/ML Applications**: PCA, spectral clustering, and PageRank algorithm
7. **Visualization**: Geometric interpretation and eigenvalue spectrum analysis

### Key Takeaways:
- Eigenvalues and eigenvectors reveal the fundamental structure of matrices
- Eigenvectors provide a natural coordinate system for understanding matrix transformations
- Diagonalization simplifies many matrix operations and calculations
- Eigenvalues and eigenvectors are fundamental to many machine learning algorithms
- Understanding these concepts is crucial for advanced linear algebra and AI/ML applications

### Next Steps:
In the next chapter, we'll explore vector spaces and subspaces, understanding the mathematical foundations of linear algebra and how they relate to eigenvalues and eigenvectors. 