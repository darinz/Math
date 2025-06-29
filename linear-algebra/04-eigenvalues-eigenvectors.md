# Eigenvalues and Eigenvectors

[![Chapter](https://img.shields.io/badge/Chapter-4-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Eigenvalues_Eigenvectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Eigenvalues and eigenvectors are fundamental concepts in linear algebra that reveal the intrinsic properties of matrices. They are crucial for understanding matrix behavior, diagonalization, and applications in machine learning like Principal Component Analysis (PCA) and spectral clustering.

## What are Eigenvalues and Eigenvectors?

For a square matrix A, a non-zero vector v is an eigenvector if:
Av = λv

where λ is the corresponding eigenvalue.

### Mathematical Definition
- **Eigenvector**: A non-zero vector v such that Av = λv
- **Eigenvalue**: A scalar λ such that Av = λv for some eigenvector v

## Finding Eigenvalues and Eigenvectors

### Characteristic Equation
The eigenvalues are solutions to the characteristic equation:
det(A - λI) = 0

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

# Verify: Av = λv
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ eigenvector
    lambda_v = eigenvalue * eigenvector
    print(f"\nEigenvalue {i+1}: {eigenvalue}")
    print(f"Av: {Av}")
    print(f"λv: {lambda_v}")
    print(f"Are they equal? {np.allclose(Av, lambda_v)}")
```

## Properties of Eigenvalues and Eigenvectors

### Basic Properties
```python
# Create a symmetric matrix
A = np.array([[2, 1], [1, 3]])
print("Symmetric matrix A:")
print(A)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:")
print(eigenvectors)

# Properties
print(f"\nTrace of A: {np.trace(A)}")
print(f"Sum of eigenvalues: {np.sum(eigenvalues)}")
print(f"Are they equal? {np.isclose(np.trace(A), np.sum(eigenvalues))}")

print(f"\nDeterminant of A: {np.linalg.det(A)}")
print(f"Product of eigenvalues: {np.prod(eigenvalues)}")
print(f"Are they equal? {np.isclose(np.linalg.det(A), np.prod(eigenvalues))}")
```

### Eigenvalues of Special Matrices
```python
# Identity matrix
I = np.eye(3)
eigenvals_I = np.linalg.eigvals(I)
print(f"Eigenvalues of identity matrix: {eigenvals_I}")

# Diagonal matrix
D = np.diag([1, 2, 3])
eigenvals_D = np.linalg.eigvals(D)
print(f"Eigenvalues of diagonal matrix: {eigenvals_D}")

# Triangular matrix
T = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
eigenvals_T = np.linalg.eigvals(T)
print(f"Eigenvalues of triangular matrix: {eigenvals_T}")
```

## Diagonalization

A matrix A is diagonalizable if it can be written as:
A = PDP⁻¹

where P contains eigenvectors and D is a diagonal matrix of eigenvalues.

```python
# Check if matrix is diagonalizable
def is_diagonalizable(A, tol=1e-10):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Check if eigenvectors are linearly independent
    P = eigenvectors
    det_P = np.linalg.det(P)
    
    return abs(det_P) > tol

# Test diagonalization
A = np.array([[4, -2], [1, 1]])
print("Matrix A:")
print(A)

if is_diagonalizable(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    
    # Reconstruct A
    A_reconstructed = P @ D @ P_inv
    print(f"\nA is diagonalizable")
    print(f"P (eigenvectors):")
    print(P)
    print(f"D (eigenvalues):")
    print(D)
    print(f"P⁻¹:")
    print(P_inv)
    print(f"A reconstructed: P @ D @ P⁻¹")
    print(A_reconstructed)
    print(f"Original A:")
    print(A)
    print(f"Are they equal? {np.allclose(A, A_reconstructed)}")
else:
    print("A is not diagonalizable")
```

## Power Method

The power method is an iterative algorithm to find the dominant eigenvalue and eigenvector.

```python
def power_method(A, max_iter=100, tol=1e-6):
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for i in range(max_iter):
        v_old = v.copy()
        v_new = A @ v
        v = v_new / np.linalg.norm(v_new)
        
        # Check convergence
        if np.linalg.norm(v - v_old) < tol:
            break
    
    # Calculate eigenvalue
    eigenvalue = (v.T @ A @ v) / (v.T @ v)
    
    return eigenvalue, v

# Test power method
A = np.array([[4, -2], [1, 1]])
dominant_eigenvalue, dominant_eigenvector = power_method(A)

print("Power method results:")
print(f"Dominant eigenvalue: {dominant_eigenvalue}")
print(f"Dominant eigenvector: {dominant_eigenvector}")

# Compare with exact values
exact_eigenvalues, exact_eigenvectors = np.linalg.eig(A)
print(f"\nExact dominant eigenvalue: {np.max(exact_eigenvalues)}")
print(f"Exact dominant eigenvector: {exact_eigenvectors[:, np.argmax(exact_eigenvalues)]}")
```

## Applications in Machine Learning

### Principal Component Analysis (PCA)
PCA uses eigenvectors of the covariance matrix to find principal components.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA results:")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Principal components (eigenvectors):")
print(pca.components_)

# Visualize PCA
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
```

### Spectral Clustering
Spectral clustering uses eigenvectors of the Laplacian matrix.

```python
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel

# Create similarity matrix
similarity_matrix = rbf_kernel(X, gamma=0.5)

# Apply spectral clustering
spectral = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
labels = spectral.fit_predict(similarity_matrix)

# Visualize clustering
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.6)
plt.title('Spectral Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### PageRank Algorithm
PageRank uses eigenvectors of the transition matrix.

```python
def create_transition_matrix(adjacency_matrix):
    """Create transition matrix from adjacency matrix"""
    n = adjacency_matrix.shape[0]
    transition_matrix = np.zeros((n, n))
    
    for i in range(n):
        out_degree = np.sum(adjacency_matrix[i, :])
        if out_degree > 0:
            transition_matrix[i, :] = adjacency_matrix[i, :] / out_degree
        else:
            # Handle dangling nodes
            transition_matrix[i, :] = 1.0 / n
    
    return transition_matrix

# Example: Simple web graph
adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

transition_matrix = create_transition_matrix(adjacency_matrix)
print("Transition matrix:")
print(transition_matrix)

# Find PageRank (eigenvector with eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
pagerank_eigenvector = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))]
pagerank_scores = np.abs(pagerank_eigenvector) / np.sum(np.abs(pagerank_eigenvector))

print(f"\nPageRank scores: {pagerank_scores}")
```

## Visualization of Eigenvalues and Eigenvectors

```python
def visualize_eigenvectors(A, title="Eigenvectors"):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    plt.figure(figsize=(10, 5))
    
    # Original vectors
    plt.subplot(1, 2, 1)
    unit_vectors = np.array([[1, 0], [0, 1]])
    for i, v in enumerate(unit_vectors):
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='blue', alpha=0.5, label=f'Unit vector {i+1}' if i == 0 else "")
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.title('Original Unit Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Transformed vectors
    plt.subplot(1, 2, 2)
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Transform unit vectors
        transformed_vectors = A @ unit_vectors.T
        
        for j, v in enumerate(transformed_vectors.T):
            plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                      color='red', alpha=0.5, label=f'Transformed {j+1}' if j == 0 else "")
        
        # Plot eigenvector direction
        plt.quiver(0, 0, eigenvector[0], eigenvector[1], angles='xy', scale_units='xy', scale=1,
                  color='green', linewidth=3, label=f'Eigenvector {i+1} (λ={eigenvalue:.2f})')
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Visualize eigenvectors
A = np.array([[2, 1], [1, 3]])
visualize_eigenvectors(A, "Eigenvectors of Matrix A")
```

## Exercises

### Exercise 1: Basic Eigenvalue Computation
```python
# Find eigenvalues and eigenvectors
A = np.array([[3, 1], [0, 2]])

# Your code here:
# 1. Find eigenvalues and eigenvectors
# 2. Verify that Av = λv for each eigenpair
# 3. Check if A is diagonalizable
# 4. Find A¹⁰ using diagonalization
```

### Exercise 2: Eigenvalue Properties
```python
# Create a 3×3 matrix and analyze its eigenvalues
B = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])

# Your code here:
# 1. Find eigenvalues and eigenvectors
# 2. Verify trace and determinant properties
# 3. Check if eigenvalues are real
# 4. Find the characteristic polynomial
```

### Exercise 3: PCA Implementation
```python
# Implement PCA from scratch using eigenvalues
data = np.random.randn(50, 3)

# Your code here:
# 1. Center the data
# 2. Compute covariance matrix
# 3. Find eigenvalues and eigenvectors
# 4. Project data onto principal components
# 5. Compare with sklearn PCA
```

## Solutions

### Solution 1: Basic Eigenvalue Computation
```python
# 1. Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:")
print(eigenvectors)

# 2. Verify Av = λv
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ eigenvector
    lambda_v = eigenvalue * eigenvector
    print(f"\nEigenpair {i+1}:")
    print(f"Av: {Av}")
    print(f"λv: {lambda_v}")
    print(f"Are equal? {np.allclose(Av, lambda_v)}")

# 3. Check diagonalizability
P = eigenvectors
det_P = np.linalg.det(P)
is_diagonalizable = abs(det_P) > 1e-10
print(f"\nIs A diagonalizable? {is_diagonalizable}")

# 4. Find A¹⁰ using diagonalization
if is_diagonalizable:
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    A_10 = P @ (D**10) @ P_inv
    print(f"\nA¹⁰:")
    print(A_10)
```

### Solution 2: Eigenvalue Properties
```python
# 1. Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:")
print(eigenvectors)

# 2. Verify properties
trace_B = np.trace(B)
det_B = np.linalg.det(B)
sum_eigenvals = np.sum(eigenvalues)
prod_eigenvals = np.prod(eigenvalues)

print(f"\nTrace of B: {trace_B}")
print(f"Sum of eigenvalues: {sum_eigenvals}")
print(f"Are equal? {np.isclose(trace_B, sum_eigenvals)}")

print(f"\nDeterminant of B: {det_B}")
print(f"Product of eigenvalues: {prod_eigenvals}")
print(f"Are equal? {np.isclose(det_B, prod_eigenvals)}")

# 3. Check if eigenvalues are real
are_real = np.allclose(eigenvalues.imag, 0)
print(f"\nAre eigenvalues real? {are_real}")

# 4. Characteristic polynomial (coefficients)
char_poly = np.poly(eigenvalues)
print(f"\nCharacteristic polynomial coefficients: {char_poly}")
```

### Solution 3: PCA Implementation
```python
# 1. Center the data
data_centered = data - np.mean(data, axis=0)

# 2. Compute covariance matrix
cov_matrix = np.cov(data_centered.T)

# 3. Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# 4. Project data onto principal components
data_pca = data_centered @ eigenvectors_sorted

print("Manual PCA results:")
print(f"Explained variance ratio: {eigenvalues_sorted / np.sum(eigenvalues_sorted)}")
print(f"Principal components:")
print(eigenvectors_sorted)

# 5. Compare with sklearn
from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=3)
data_pca_sklearn = pca_sklearn.fit_transform(data)

print(f"\nSklearn PCA explained variance ratio: {pca_sklearn.explained_variance_ratio_}")
print(f"Are results similar? {np.allclose(data_pca, data_pca_sklearn, atol=1e-10)}")
```

## Summary

In this chapter, we covered:
- Definition and computation of eigenvalues and eigenvectors
- Properties of eigenvalues and eigenvectors
- Matrix diagonalization
- Power method for finding dominant eigenvalues
- Applications in machine learning (PCA, spectral clustering, PageRank)
- Visualization techniques

Eigenvalues and eigenvectors are fundamental for understanding matrix behavior and are essential in many machine learning algorithms.

## Next Steps

In the next chapter, we'll explore vector spaces and subspaces, which provide the theoretical foundation for understanding linear algebra concepts. 