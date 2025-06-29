# Vector Spaces and Subspaces

[![Chapter](https://img.shields.io/badge/Chapter-5-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Vector_Spaces-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Vector spaces provide the mathematical foundation for linear algebra. They are sets of vectors that satisfy certain axioms and are fundamental for understanding linear transformations, subspaces, and the structure of linear systems.

## What is a Vector Space?

A vector space V over a field F (usually ℝ or ℂ) is a set of vectors with two operations:
1. **Vector addition**: u + v ∈ V for all u, v ∈ V
2. **Scalar multiplication**: cu ∈ V for all c ∈ F, u ∈ V

### Axioms of a Vector Space
- **Commutativity**: u + v = v + u
- **Associativity**: (u + v) + w = u + (v + w)
- **Identity**: There exists 0 ∈ V such that v + 0 = v
- **Inverse**: For each v ∈ V, there exists -v such that v + (-v) = 0
- **Distributivity**: c(u + v) = cu + cv and (c + d)u = cu + du
- **Scalar associativity**: c(du) = (cd)u
- **Scalar identity**: 1u = u

## Common Vector Spaces

### ℝⁿ (Real Vector Space)
```python
import numpy as np

# ℝ² - 2D real vectors
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Vector addition
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")

# Scalar multiplication
scaled_v = 2 * v1
print(f"2 * v1 = {scaled_v}")

# Zero vector
zero_vector = np.zeros(2)
print(f"Zero vector: {zero_vector}")

# Verify axioms
print(f"v1 + v2 = v2 + v1: {np.array_equal(v1 + v2, v2 + v1)}")
print(f"(v1 + v2) + v3 = v1 + (v2 + v3): {np.array_equal((v1 + v2) + np.array([5, 6]), v1 + (v2 + np.array([5, 6])))}")
```

### Function Spaces
```python
# Space of polynomials of degree ≤ 2
def polynomial_space_example():
    # Basis: {1, x, x²}
    p1 = lambda x: 1 + 2*x + 3*x**2  # 1 + 2x + 3x²
    p2 = lambda x: 2 - x + x**2      # 2 - x + x²
    
    # Vector addition (function addition)
    p_sum = lambda x: p1(x) + p2(x)
    
    # Scalar multiplication
    p_scaled = lambda x: 2 * p1(x)
    
    # Test at x = 1
    x_test = 1
    print(f"p1({x_test}) = {p1(x_test)}")
    print(f"p2({x_test}) = {p2(x_test)}")
    print(f"(p1 + p2)({x_test}) = {p_sum(x_test)}")
    print(f"(2*p1)({x_test}) = {p_scaled(x_test)}")

polynomial_space_example()
```

## Subspaces

A subspace W of a vector space V is a subset that is itself a vector space.

### Subspace Criteria
For W to be a subspace of V:
1. **Closure under addition**: u, v ∈ W → u + v ∈ W
2. **Closure under scalar multiplication**: u ∈ W, c ∈ F → cu ∈ W
3. **Contains zero vector**: 0 ∈ W

```python
def is_subspace(vectors, test_vectors=None):
    """Check if a set of vectors forms a subspace"""
    if test_vectors is None:
        test_vectors = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([0, 0])
        ]
    
    # Check if zero vector is in the span
    zero_in_span = False
    for v in test_vectors:
        if np.allclose(v, np.zeros_like(v)):
            zero_in_span = True
            break
    
    # Check closure under addition and scalar multiplication
    # (This is a simplified check - in practice, you'd need to check all combinations)
    closed = True
    
    return zero_in_span and closed

# Example: Line through origin in ℝ²
line_vectors = [np.array([1, 2]), np.array([2, 4]), np.array([0, 0])]
print(f"Line through origin is subspace: {is_subspace(line_vectors)}")

# Example: Plane not through origin
plane_vectors = [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 2])]
print(f"Plane not through origin is subspace: {is_subspace(plane_vectors)}")
```

## Span

The span of a set of vectors is the set of all linear combinations of those vectors.

```python
def span_vectors(vectors, coefficients):
    """Compute linear combination of vectors"""
    result = np.zeros_like(vectors[0])
    for v, c in zip(vectors, coefficients):
        result += c * v
    return result

# Example: Span of two vectors in ℝ²
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Different linear combinations
combinations = [
    [1, 0],   # 1*v1 + 0*v2 = [1, 0]
    [0, 1],   # 0*v1 + 1*v2 = [0, 1]
    [2, 3],   # 2*v1 + 3*v2 = [2, 3]
    [-1, 2]   # -1*v1 + 2*v2 = [-1, 2]
]

print("Linear combinations:")
for coeffs in combinations:
    result = span_vectors([v1, v2], coeffs)
    print(f"{coeffs[0]}*v1 + {coeffs[1]}*v2 = {result}")
```

## Linear Independence

A set of vectors is linearly independent if no vector can be written as a linear combination of the others.

```python
def is_linearly_independent(vectors, tol=1e-10):
    """Check if vectors are linearly independent"""
    # Convert to matrix
    matrix = np.column_stack(vectors)
    
    # Check rank
    rank = np.linalg.matrix_rank(matrix)
    
    return rank == len(vectors)

# Test linear independence
independent_vectors = [
    np.array([1, 0]),
    np.array([0, 1])
]
print(f"Independent vectors: {is_linearly_independent(independent_vectors)}")

dependent_vectors = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])  # This is a linear combination of the first two
]
print(f"Dependent vectors: {is_linearly_independent(dependent_vectors)}")
```

## Basis and Dimension

A basis for a vector space is a linearly independent set that spans the space.

```python
def find_basis(vectors):
    """Find a basis for the span of vectors"""
    matrix = np.column_stack(vectors)
    
    # Use QR decomposition to find basis
    Q, R, P = np.linalg.qr(matrix, mode='full', pivoting=True)
    
    # Find rank
    rank = np.linalg.matrix_rank(matrix)
    
    # Return first 'rank' vectors as basis
    basis = [vectors[P[i]] for i in range(rank)]
    
    return basis

# Example: Find basis for ℝ³
vectors_3d = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
    np.array([1, 1, 1])  # Redundant vector
]

basis = find_basis(vectors_3d)
print("Basis vectors:")
for i, v in enumerate(basis):
    print(f"v{i+1} = {v}")

print(f"Dimension: {len(basis)}")
```

## Null Space and Column Space

### Null Space (Kernel)
The null space of a matrix A is the set of vectors x such that Ax = 0.

```python
def null_space(A, tol=1e-10):
    """Find null space of matrix A"""
    U, S, Vt = np.linalg.svd(A)
    
    # Find singular values close to zero
    null_indices = np.where(S < tol)[0]
    
    if len(null_indices) == 0:
        return np.array([]).reshape(A.shape[1], 0)
    
    # Null space basis
    null_basis = Vt[null_indices, :].T
    
    return null_basis

# Example
A = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix A:")
print(A)

null_basis = null_space(A)
print(f"\nNull space dimension: {null_basis.shape[1]}")
if null_basis.size > 0:
    print("Null space basis:")
    print(null_basis)
    
    # Verify: A * null_basis ≈ 0
    verification = A @ null_basis
    print(f"\nVerification (should be close to zero):")
    print(verification)
```

### Column Space (Range)
The column space of a matrix A is the span of its columns.

```python
def column_space(A):
    """Find column space of matrix A"""
    Q, R, P = np.linalg.qr(A, mode='full', pivoting=True)
    
    # Find rank
    rank = np.linalg.matrix_rank(A)
    
    # Return first 'rank' columns of Q as basis
    col_basis = Q[:, :rank]
    
    return col_basis

# Example
col_basis = column_space(A)
print(f"Column space dimension: {col_basis.shape[1]}")
print("Column space basis:")
print(col_basis)
```

## Rank-Nullity Theorem

For any matrix A: rank(A) + nullity(A) = n

```python
def rank_nullity_theorem(A):
    """Verify rank-nullity theorem"""
    rank = np.linalg.matrix_rank(A)
    nullity = null_space(A).shape[1]
    n = A.shape[1]
    
    print(f"Rank: {rank}")
    print(f"Nullity: {nullity}")
    print(f"n: {n}")
    print(f"Rank + Nullity = n: {rank + nullity == n}")
    
    return rank, nullity

# Test with different matrices
A1 = np.array([[1, 2], [3, 4]])  # Full rank
A2 = np.array([[1, 2], [2, 4]])  # Rank deficient

print("Matrix A1:")
rank_nullity_theorem(A1)

print("\nMatrix A2:")
rank_nullity_theorem(A2)
```

## Applications in Machine Learning

### Feature Space
```python
# Example: Feature vectors in machine learning
feature_vectors = [
    np.array([1, 2, 3]),  # Sample 1: [feature1, feature2, feature3]
    np.array([4, 5, 6]),  # Sample 2
    np.array([7, 8, 9]),  # Sample 3
    np.array([2, 4, 6])   # Sample 4
]

# Check if features are linearly independent
feature_matrix = np.column_stack(feature_vectors)
rank = np.linalg.matrix_rank(feature_matrix)
print(f"Feature matrix rank: {rank}")
print(f"Number of features: {feature_matrix.shape[0]}")
print(f"Are features linearly independent? {rank == feature_matrix.shape[0]}")

# Find feature subspace
feature_basis = column_space(feature_matrix.T)
print(f"Feature subspace dimension: {feature_basis.shape[1]}")
```

### Kernel Methods
```python
def rbf_kernel(x1, x2, gamma=1.0):
    """Radial Basis Function kernel"""
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# Example: Kernel feature space
X = np.array([[1, 2], [3, 4], [5, 6]])
n_samples = X.shape[0]

# Compute kernel matrix
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i, j] = rbf_kernel(X[i], X[j])

print("Kernel matrix:")
print(K)

# Check if kernel matrix is positive definite
eigenvalues = np.linalg.eigvals(K)
print(f"Kernel matrix eigenvalues: {eigenvalues}")
print(f"Is positive definite? {np.all(eigenvalues > 0)}")
```

## Exercises

### Exercise 1: Subspace Verification
```python
# Check if given sets are subspaces
set1 = [np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]  # ℝ²
set2 = [np.array([1, 1]), np.array([2, 2]), np.array([0, 0])]  # Line through origin

# Your code here:
# 1. Check if each set is a subspace
# 2. Find the dimension of each subspace
# 3. Find a basis for each subspace
```

### Exercise 2: Linear Independence
```python
# Test linear independence of vectors
vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
    np.array([2, 4, 6])
]

# Your code here:
# 1. Check if vectors are linearly independent
# 2. Find a basis for their span
# 3. Find the dimension of the span
```

### Exercise 3: Matrix Spaces
```python
# Analyze matrix spaces
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Your code here:
# 1. Find null space of A
# 2. Find column space of A
# 3. Verify rank-nullity theorem
# 4. Find row space of A
```

## Solutions

### Solution 1: Subspace Verification
```python
# 1. Check if each set is a subspace
def check_subspace(vectors):
    # Check if zero vector is present
    has_zero = any(np.allclose(v, np.zeros_like(v)) for v in vectors)
    
    # Check closure under addition (simplified)
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    
    return has_zero and rank <= len(vectors)

print(f"Set 1 is subspace: {check_subspace(set1)}")
print(f"Set 2 is subspace: {check_subspace(set2)}")

# 2. Find dimension
def subspace_dimension(vectors):
    matrix = np.column_stack(vectors)
    return np.linalg.matrix_rank(matrix)

print(f"Set 1 dimension: {subspace_dimension(set1)}")
print(f"Set 2 dimension: {subspace_dimension(set2)}")

# 3. Find basis
basis1 = find_basis(set1)
basis2 = find_basis(set2)

print("Basis for set 1:")
for v in basis1:
    print(v)
print("Basis for set 2:")
for v in basis2:
    print(v)
```

### Solution 2: Linear Independence
```python
# 1. Check linear independence
is_independent = is_linearly_independent(vectors)
print(f"Vectors are linearly independent: {is_independent}")

# 2. Find basis
basis = find_basis(vectors)
print(f"Basis vectors:")
for i, v in enumerate(basis):
    print(f"v{i+1} = {v}")

# 3. Find dimension
dimension = len(basis)
print(f"Dimension of span: {dimension}")
```

### Solution 3: Matrix Spaces
```python
# 1. Null space
null_basis = null_space(A)
print(f"Null space dimension: {null_basis.shape[1]}")
if null_basis.size > 0:
    print("Null space basis:")
    print(null_basis)

# 2. Column space
col_basis = column_space(A)
print(f"\nColumn space dimension: {col_basis.shape[1]}")
print("Column space basis:")
print(col_basis)

# 3. Verify rank-nullity theorem
rank, nullity = rank_nullity_theorem(A)

# 4. Row space (same as column space of A^T)
row_basis = column_space(A.T)
print(f"\nRow space dimension: {row_basis.shape[1]}")
print("Row space basis:")
print(row_basis)
```

## Summary

In this chapter, we covered:
- Definition and axioms of vector spaces
- Subspaces and their properties
- Span and linear independence
- Basis and dimension
- Null space and column space
- Rank-nullity theorem
- Applications in machine learning

Vector spaces provide the theoretical foundation for understanding linear algebra concepts and their applications.

## Next Steps

In the next chapter, we'll explore linear independence and basis in more detail, focusing on coordinate systems and change of basis. 