# Matrices and Matrix Operations

[![Chapter](https://img.shields.io/badge/Chapter-2-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Matrices-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-brightgreen.svg)]()

## Introduction

Matrices are rectangular arrays of numbers that represent linear transformations and systems of linear equations. They are fundamental in machine learning for representing data, transformations, and computations.

## What is a Matrix?

A matrix is a 2D array of numbers arranged in rows and columns. An m×n matrix has m rows and n columns.

### Mathematical Notation
A matrix A is written as:
A = [aᵢⱼ] where i = 1,2,...,m and j = 1,2,...,n

### Python Implementation
```python
import numpy as np

# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("Matrix A:")
print(A)
print("\nShape of A:", A.shape)
print("\nMatrix B:")
print(B)
```

## Matrix Operations

### Addition and Subtraction
Matrices are added/subtracted element-wise (must have same dimensions):
(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ

```python
# Matrix addition
C = A + B
print("A + B:")
print(C)

# Matrix subtraction
D = A - B
print("\nA - B:")
print(D)

# Broadcasting with scalar
E = A + 5
print("\nA + 5:")
print(E)
```

### Scalar Multiplication
Multiplying a matrix by a scalar multiplies each element:
(cA)ᵢⱼ = c × Aᵢⱼ

```python
# Scalar multiplication
scaled_A = 2 * A
print("2 * A:")
print(scaled_A)

# Division
divided_A = A / 2
print("\nA / 2:")
print(divided_A)
```

### Matrix Multiplication
Matrix multiplication is more complex than element-wise multiplication:
(AB)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

```python
# Matrix multiplication
AB = np.dot(A, B)
print("A × B:")
print(AB)

# Alternative syntax
AB_alt = A @ B
print("\nA @ B (same result):")
print(AB_alt)

# Element-wise multiplication (Hadamard product)
element_wise = A * B
print("\nElement-wise multiplication (A * B):")
print(element_wise)
```

## Special Matrices

### Identity Matrix
The identity matrix I has 1s on the diagonal and 0s elsewhere.

```python
# Identity matrix
I = np.eye(3)
print("3×3 Identity matrix:")
print(I)

# Verify: AI = A
result = A @ I
print("\nA × I = A:")
print(result)
print("\nIs A × I equal to A?", np.array_equal(result, A))
```

### Zero Matrix
A zero matrix has all elements equal to zero.

```python
# Zero matrix
Z = np.zeros((3, 3))
print("3×3 Zero matrix:")
print(Z)
```

### Diagonal Matrix
A diagonal matrix has non-zero elements only on the main diagonal.

```python
# Diagonal matrix
D = np.diag([1, 2, 3])
print("Diagonal matrix:")
print(D)

# Extract diagonal from matrix
diagonal = np.diag(A)
print("\nDiagonal of A:")
print(diagonal)
```

## Matrix Properties

### Transpose
The transpose of a matrix flips rows and columns:
(Aᵀ)ᵢⱼ = Aⱼᵢ

```python
# Matrix transpose
A_transpose = A.T
print("A:")
print(A)
print("\nAᵀ:")
print(A_transpose)

# Properties of transpose
print("\n(Aᵀ)ᵀ = A:", np.array_equal((A.T).T, A))
print("(A + B)ᵀ = Aᵀ + Bᵀ:", np.array_equal((A + B).T, A.T + B.T))
```

### Trace
The trace is the sum of diagonal elements:
tr(A) = Σᵢ Aᵢᵢ

```python
# Matrix trace
trace_A = np.trace(A)
print("Trace of A:", trace_A)

# Verify: tr(A + B) = tr(A) + tr(B)
trace_sum = np.trace(A + B)
trace_A_plus_trace_B = np.trace(A) + np.trace(B)
print("tr(A + B) = tr(A) + tr(B):", trace_sum == trace_A_plus_trace_B)
```

## Matrix Types

### Symmetric Matrix
A symmetric matrix satisfies A = Aᵀ.

```python
# Create symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

print("Symmetric matrix S:")
print(S)
print("\nS = Sᵀ:", np.array_equal(S, S.T))
```

### Skew-Symmetric Matrix
A skew-symmetric matrix satisfies A = -Aᵀ.

```python
# Create skew-symmetric matrix
K = np.array([[0, 2, -3],
              [-2, 0, 4],
              [3, -4, 0]])

print("Skew-symmetric matrix K:")
print(K)
print("\nK = -Kᵀ:", np.array_equal(K, -K.T))
```

### Orthogonal Matrix
An orthogonal matrix satisfies AᵀA = AAᵀ = I.

```python
# Create orthogonal matrix (rotation matrix)
theta = np.pi/4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

print("Orthogonal matrix Q (rotation by 45°):")
print(Q)
print("\nQᵀQ:")
print(Q.T @ Q)
print("\nIs Q orthogonal?", np.allclose(Q.T @ Q, np.eye(2)))
```

## Matrix Operations in Machine Learning

### Data Representation
Matrices are used to represent datasets where rows are samples and columns are features.

```python
# Sample dataset matrix
# Rows: samples (houses)
# Columns: features (sqft, bedrooms, bathrooms, price)
dataset = np.array([
    [2000, 3, 2, 300000],
    [1500, 2, 1, 250000],
    [3000, 4, 3, 450000],
    [1200, 2, 1, 200000]
])

print("House dataset matrix:")
print("Rows: houses, Columns: [sqft, bedrooms, bathrooms, price]")
print(dataset)

# Extract features and target
X = dataset[:, :3]  # Features
y = dataset[:, 3]   # Target (price)

print("\nFeature matrix X:")
print(X)
print("\nTarget vector y:")
print(y)
```

### Feature Scaling
```python
# Standardize features (z-score normalization)
def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_scaled = standardize_features(X)
print("Standardized features:")
print(X_scaled)
print("\nMean of scaled features:", np.mean(X_scaled, axis=0))
print("Std of scaled features:", np.std(X_scaled, axis=0))
```

### Distance Matrix
```python
# Calculate pairwise distances between samples
from scipy.spatial.distance import pdist, squareform

distances = squareform(pdist(X_scaled, 'euclidean'))
print("Pairwise distance matrix:")
print(distances)
```

## Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of matrix
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
sns.heatmap(A, annot=True, cmap='viridis', cbar=True)
plt.title('Matrix A')

plt.subplot(2, 2, 2)
sns.heatmap(B, annot=True, cmap='plasma', cbar=True)
plt.title('Matrix B')

plt.subplot(2, 2, 3)
sns.heatmap(A @ B, annot=True, cmap='magma', cbar=True)
plt.title('A × B')

plt.subplot(2, 2, 4)
sns.heatmap(X_scaled, annot=True, cmap='coolwarm', cbar=True)
plt.title('Standardized Features')

plt.tight_layout()
plt.show()
```

## Exercises

### Exercise 1: Basic Matrix Operations
```python
# Create two 3×3 matrices and perform operations
M1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
M2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Your code here:
# 1. Add M1 and M2
# 2. Multiply M1 by M2
# 3. Calculate M1ᵀ
# 4. Find the trace of M1
# 5. Check if M1 is symmetric
```

### Exercise 2: Matrix Properties
```python
# Create a 2×2 matrix and verify properties
A = np.array([[2, 1], [1, 3]])

# Your code here:
# 1. Verify (Aᵀ)ᵀ = A
# 2. Check if A is symmetric
# 3. Calculate A²
# 4. Find eigenvalues (we'll learn this in Chapter 4)
```

### Exercise 3: Data Matrix Operations
```python
# Create a sample dataset
data = np.array([
    [1, 2, 3, 10],
    [4, 5, 6, 20],
    [7, 8, 9, 30],
    [10, 11, 12, 40]
])

# Your code here:
# 1. Separate features and target
# 2. Standardize the features
# 3. Calculate correlation matrix
# 4. Find the most correlated features
```

## Solutions

### Solution 1: Basic Matrix Operations
```python
# 1. Addition
result = M1 + M2
print("M1 + M2:")
print(result)

# 2. Multiplication
result = M1 @ M2
print("\nM1 × M2:")
print(result)

# 3. Transpose
M1_transpose = M1.T
print("\nM1ᵀ:")
print(M1_transpose)

# 4. Trace
trace = np.trace(M1)
print(f"\nTrace of M1: {trace}")

# 5. Symmetry check
is_symmetric = np.array_equal(M1, M1.T)
print(f"M1 is symmetric: {is_symmetric}")
```

### Solution 2: Matrix Properties
```python
# 1. Verify (Aᵀ)ᵀ = A
transpose_check = np.array_equal((A.T).T, A)
print("(Aᵀ)ᵀ = A:", transpose_check)

# 2. Symmetry check
is_symmetric = np.array_equal(A, A.T)
print("A is symmetric:", is_symmetric)

# 3. A²
A_squared = A @ A
print("\nA²:")
print(A_squared)

# 4. Eigenvalues (preview)
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalues of A: {eigenvalues}")
```

### Solution 3: Data Matrix Operations
```python
# 1. Separate features and target
X = data[:, :3]
y = data[:, 3]
print("Features X:")
print(X)
print("\nTarget y:")
print(y)

# 2. Standardize features
X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print("\nStandardized features:")
print(X_scaled)

# 3. Correlation matrix
correlation_matrix = np.corrcoef(X.T)
print("\nCorrelation matrix:")
print(correlation_matrix)

# 4. Most correlated features
# Find off-diagonal elements
n_features = X.shape[1]
max_corr = 0
max_pair = (0, 0)

for i in range(n_features):
    for j in range(i+1, n_features):
        corr = abs(correlation_matrix[i, j])
        if corr > max_corr:
            max_corr = corr
            max_pair = (i, j)

print(f"\nMost correlated features: {max_pair} with correlation {max_corr:.3f}")
```

## Summary

In this chapter, we covered:
- Matrix creation and basic operations
- Special matrices (identity, zero, diagonal)
- Matrix properties (transpose, trace)
- Matrix types (symmetric, skew-symmetric, orthogonal)
- Applications in machine learning
- Matrix visualization techniques

These concepts are essential for understanding linear transformations and solving systems of linear equations.

## Next Steps

In the next chapter, we'll explore linear transformations and how matrices represent geometric operations in space. 