# Linear Independence and Basis

[![Chapter](https://img.shields.io/badge/Chapter-6-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Linear_Independence-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Linear independence and basis are fundamental concepts that determine the structure and dimension of vector spaces. Understanding these concepts is crucial for solving systems of equations, performing coordinate transformations, and analyzing data in machine learning.

## Linear Independence

A set of vectors {v₁, v₂, ..., vₙ} is linearly independent if the only solution to:
c₁v₁ + c₂v₂ + ... + cₙvₙ = 0
is c₁ = c₂ = ... = cₙ = 0.

### Mathematical Definition
Vectors v₁, v₂, ..., vₙ are linearly independent if:
- The equation c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 has only the trivial solution
- No vector can be written as a linear combination of the others
- The rank of the matrix [v₁ v₂ ... vₙ] equals n

```python
import numpy as np

def is_linearly_independent(vectors, tol=1e-10):
    """Check if vectors are linearly independent"""
    if not vectors:
        return True
    
    # Convert to matrix
    matrix = np.column_stack(vectors)
    
    # Check rank
    rank = np.linalg.matrix_rank(matrix)
    
    return rank == len(vectors)

# Example 1: Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

independent_vectors = [v1, v2, v3]
print("Example 1: Standard basis vectors")
print(f"Are linearly independent: {is_linearly_independent(independent_vectors)}")

# Example 2: Linearly dependent vectors
v4 = np.array([1, 1, 0])  # This is v1 + v2
dependent_vectors = [v1, v2, v4]
print("\nExample 2: Including dependent vector")
print(f"Are linearly independent: {is_linearly_independent(dependent_vectors)}")
```

### Testing Linear Independence
```python
def test_linear_independence_detailed(vectors):
    """Detailed test of linear independence"""
    if not vectors:
        return True, "Empty set is linearly independent"
    
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    n_vectors = len(vectors)
    
    print(f"Number of vectors: {n_vectors}")
    print(f"Matrix rank: {rank}")
    print(f"Matrix shape: {matrix.shape}")
    
    if rank == n_vectors:
        print("✓ Vectors are linearly independent")
        return True, "Full rank"
    else:
        print("✗ Vectors are linearly dependent")
        
        # Find dependent vectors using SVD
        U, S, Vt = np.linalg.svd(matrix)
        tol = 1e-10
        independent_cols = np.where(S > tol)[0]
        dependent_cols = np.where(S <= tol)[0]
        
        print(f"Independent columns: {independent_cols}")
        print(f"Dependent columns: {dependent_cols}")
        
        return False, f"Rank {rank} < {n_vectors}"

# Test with different sets
print("=== Testing Linear Independence ===")
test_vectors1 = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9])
]
is_indep1, msg1 = test_linear_independence_detailed(test_vectors1)

print("\n" + "="*50)
test_vectors2 = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0])
]
is_indep2, msg2 = test_linear_independence_detailed(test_vectors2)
```

## Basis

A basis for a vector space V is a linearly independent set that spans V.

### Properties of a Basis
1. **Linear Independence**: All vectors in the basis are linearly independent
2. **Spanning**: Every vector in V can be written as a linear combination of basis vectors
3. **Minimal**: No proper subset spans V
4. **Unique Representation**: Each vector has a unique representation in terms of the basis

```python
def find_basis(vectors):
    """Find a basis for the span of vectors"""
    if not vectors:
        return []
    
    matrix = np.column_stack(vectors)
    
    # Use QR decomposition with pivoting
    Q, R, P = np.linalg.qr(matrix, mode='full', pivoting=True)
    
    # Find rank
    rank = np.linalg.matrix_rank(matrix)
    
    # Return first 'rank' vectors as basis
    basis = [vectors[P[i]] for i in range(rank)]
    
    return basis

# Example: Find basis for a set of vectors
vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
    np.array([2, 4, 6])  # This is 2 * first vector
]

print("Original vectors:")
for i, v in enumerate(vectors):
    print(f"v{i+1} = {v}")

basis = find_basis(vectors)
print(f"\nBasis vectors (dimension: {len(basis)}):")
for i, v in enumerate(basis):
    print(f"b{i+1} = {v}")
```

### Standard Basis
```python
def standard_basis(n):
    """Generate standard basis for ℝⁿ"""
    basis = []
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        basis.append(e_i)
    return basis

# Generate standard basis for ℝ³
std_basis_3d = standard_basis(3)
print("Standard basis for ℝ³:")
for i, e in enumerate(std_basis_3d):
    print(f"e{i+1} = {e}")

# Verify it's a basis
print(f"\nIs standard basis linearly independent: {is_linearly_independent(std_basis_3d)}")
```

## Coordinate Systems

### Vector Representation in Different Bases
```python
def vector_in_basis(vector, basis):
    """Find coordinates of vector in given basis"""
    # Solve: vector = c₁b₁ + c₂b₂ + ... + cₙbₙ
    # This is equivalent to: basis_matrix @ coordinates = vector
    
    basis_matrix = np.column_stack(basis)
    coordinates = np.linalg.solve(basis_matrix, vector)
    
    return coordinates

def reconstruct_vector(coordinates, basis):
    """Reconstruct vector from coordinates in basis"""
    basis_matrix = np.column_stack(basis)
    vector = basis_matrix @ coordinates
    return vector

# Example: Vector in different bases
v = np.array([3, 4])

# Standard basis
std_basis = [np.array([1, 0]), np.array([0, 1])]
coords_std = vector_in_basis(v, std_basis)
print(f"Vector {v} in standard basis: {coords_std}")

# Different basis
new_basis = [np.array([1, 1]), np.array([1, -1])]
coords_new = vector_in_basis(v, new_basis)
print(f"Vector {v} in new basis: {coords_new}")

# Verify reconstruction
v_reconstructed = reconstruct_vector(coords_new, new_basis)
print(f"Reconstructed vector: {v_reconstructed}")
print(f"Reconstruction error: {np.linalg.norm(v - v_reconstructed)}")
```

## Change of Basis

### Change of Basis Matrix
```python
def change_of_basis_matrix(old_basis, new_basis):
    """Find change of basis matrix from old_basis to new_basis"""
    # P = [new_basis]_old_basis
    # This matrix satisfies: [v]_new = P^(-1) [v]_old
    
    old_matrix = np.column_stack(old_basis)
    new_matrix = np.column_stack(new_basis)
    
    # P = new_matrix @ old_matrix^(-1)
    P = new_matrix @ np.linalg.inv(old_matrix)
    
    return P

def change_coordinates(vector, old_basis, new_basis):
    """Change vector coordinates from old_basis to new_basis"""
    P = change_of_basis_matrix(old_basis, new_basis)
    
    # Get coordinates in old basis
    coords_old = vector_in_basis(vector, old_basis)
    
    # Change to new basis
    coords_new = np.linalg.inv(P) @ coords_old
    
    return coords_new

# Example: Change of basis
old_basis = [np.array([1, 0]), np.array([0, 1])]  # Standard basis
new_basis = [np.array([1, 1]), np.array([1, -1])]  # New basis

vector = np.array([3, 4])

# Method 1: Direct computation
coords_old = vector_in_basis(vector, old_basis)
coords_new = vector_in_basis(vector, new_basis)

print(f"Vector {vector}")
print(f"Coordinates in old basis: {coords_old}")
print(f"Coordinates in new basis: {coords_new}")

# Method 2: Using change of basis matrix
P = change_of_basis_matrix(old_basis, new_basis)
coords_new_via_P = np.linalg.inv(P) @ coords_old

print(f"Change of basis matrix P:")
print(P)
print(f"Coordinates via change of basis: {coords_new_via_P}")
```

## Gram-Schmidt Orthogonalization

### Orthogonal Basis
```python
def gram_schmidt(vectors):
    """Apply Gram-Schmidt orthogonalization to vectors"""
    if not vectors:
        return []
    
    orthogonal_basis = []
    
    for i, v in enumerate(vectors):
        # Start with original vector
        u = v.copy()
        
        # Subtract projections onto previous orthogonal vectors
        for j in range(i):
            proj = np.dot(v, orthogonal_basis[j]) / np.dot(orthogonal_basis[j], orthogonal_basis[j])
            u = u - proj * orthogonal_basis[j]
        
        # Normalize
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid zero vectors
            u = u / norm
            orthogonal_basis.append(u)
    
    return orthogonal_basis

# Example: Gram-Schmidt orthogonalization
vectors = [
    np.array([1, 1, 0]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1])
]

print("Original vectors:")
for i, v in enumerate(vectors):
    print(f"v{i+1} = {v}")

orthogonal_basis = gram_schmidt(vectors)
print(f"\nOrthogonal basis:")
for i, u in enumerate(orthogonal_basis):
    print(f"u{i+1} = {u}")

# Verify orthogonality
print(f"\nOrthogonality check:")
for i in range(len(orthogonal_basis)):
    for j in range(i+1, len(orthogonal_basis)):
        dot_product = np.dot(orthogonal_basis[i], orthogonal_basis[j])
        print(f"u{i+1} · u{j+1} = {dot_product:.2e}")
```

## Applications in Machine Learning

### Feature Selection
```python
def select_linearly_independent_features(feature_matrix, tol=1e-10):
    """Select linearly independent features from matrix"""
    # Use QR decomposition to find independent columns
    Q, R, P = np.linalg.qr(feature_matrix, mode='full', pivoting=True)
    
    # Find rank
    rank = np.linalg.matrix_rank(feature_matrix)
    
    # Select independent features
    independent_indices = P[:rank]
    independent_features = feature_matrix[:, independent_indices]
    
    return independent_features, independent_indices

# Example: Feature selection
np.random.seed(42)
# Create feature matrix with some linear dependencies
n_samples, n_features = 100, 5
features = np.random.randn(n_samples, n_features)

# Add linear dependencies
features[:, 2] = 2 * features[:, 0] + 3 * features[:, 1]  # Feature 2 is dependent
features[:, 4] = features[:, 0] - features[:, 1]          # Feature 4 is dependent

print(f"Original feature matrix shape: {features.shape}")
print(f"Original rank: {np.linalg.matrix_rank(features)}")

# Select independent features
independent_features, indices = select_linearly_independent_features(features)
print(f"Independent features shape: {independent_features.shape}")
print(f"Selected feature indices: {indices}")
print(f"Independent features rank: {np.linalg.matrix_rank(independent_features)}")
```

### Principal Component Analysis (PCA)
```python
def pca_basis(data, n_components=None):
    """Find PCA basis for data"""
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    if n_components is None:
        n_components = len(eigenvalues)
    
    # Return basis vectors (principal components)
    basis = [eigenvectors_sorted[:, i] for i in range(n_components)]
    
    return basis, eigenvalues_sorted[:n_components]

# Example: PCA basis
np.random.seed(42)
# Create correlated data
n_samples = 100
data = np.random.randn(n_samples, 3)
data[:, 2] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n_samples)  # Add correlation

pca_basis_vectors, eigenvalues = pca_basis(data, n_components=2)

print("PCA basis vectors:")
for i, v in enumerate(pca_basis_vectors):
    print(f"PC{i+1} = {v}")

print(f"\nExplained variance ratios:")
total_variance = np.sum(eigenvalues)
for i, eigenval in enumerate(eigenvalues):
    print(f"PC{i+1}: {eigenval/total_variance:.3f}")
```

## Exercises

### Exercise 1: Linear Independence Testing
```python
# Test linear independence of different vector sets
vectors_set1 = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9])
]

vectors_set2 = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0])
]

# Your code here:
# 1. Test linear independence of both sets
# 2. Find bases for both sets
# 3. Determine the dimension of each span
```

### Exercise 2: Change of Basis
```python
# Perform change of basis
old_basis = [np.array([1, 0]), np.array([0, 1])]
new_basis = [np.array([2, 1]), np.array([1, 2])]
vector = np.array([5, 3])

# Your code here:
# 1. Find coordinates of vector in old basis
# 2. Find change of basis matrix
# 3. Find coordinates of vector in new basis
# 4. Verify the transformation
```

### Exercise 3: Gram-Schmidt Process
```python
# Apply Gram-Schmidt to vectors
vectors = [
    np.array([1, 1, 1]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1])
]

# Your code here:
# 1. Apply Gram-Schmidt orthogonalization
# 2. Verify orthogonality of result
# 3. Check if the result is a basis
```

## Solutions

### Solution 1: Linear Independence Testing
```python
# 1. Test linear independence
print("Set 1:")
is_indep1, msg1 = test_linear_independence_detailed(vectors_set1)

print("\nSet 2:")
is_indep2, msg2 = test_linear_independence_detailed(vectors_set2)

# 2. Find bases
basis1 = find_basis(vectors_set1)
basis2 = find_basis(vectors_set2)

print(f"\nBasis for set 1 (dimension: {len(basis1)}):")
for v in basis1:
    print(v)

print(f"\nBasis for set 2 (dimension: {len(basis2)}):")
for v in basis2:
    print(v)

# 3. Determine dimensions
print(f"\nDimension of span 1: {len(basis1)}")
print(f"Dimension of span 2: {len(basis2)}")
```

### Solution 2: Change of Basis
```python
# 1. Coordinates in old basis
coords_old = vector_in_basis(vector, old_basis)
print(f"Coordinates in old basis: {coords_old}")

# 2. Change of basis matrix
P = change_of_basis_matrix(old_basis, new_basis)
print(f"Change of basis matrix:")
print(P)

# 3. Coordinates in new basis
coords_new = np.linalg.inv(P) @ coords_old
print(f"Coordinates in new basis: {coords_new}")

# 4. Verify transformation
vector_reconstructed = reconstruct_vector(coords_new, new_basis)
print(f"Reconstructed vector: {vector_reconstructed}")
print(f"Original vector: {vector}")
print(f"Verification error: {np.linalg.norm(vector - vector_reconstructed)}")
```

### Solution 3: Gram-Schmidt Process
```python
# 1. Apply Gram-Schmidt
orthogonal_basis = gram_schmidt(vectors)
print("Orthogonal basis:")
for i, v in enumerate(orthogonal_basis):
    print(f"u{i+1} = {v}")

# 2. Verify orthogonality
print(f"\nOrthogonality check:")
for i in range(len(orthogonal_basis)):
    for j in range(i+1, len(orthogonal_basis)):
        dot_product = np.dot(orthogonal_basis[i], orthogonal_basis[j])
        print(f"u{i+1} · u{j+1} = {dot_product:.2e}")

# 3. Check if it's a basis
is_basis = is_linearly_independent(orthogonal_basis)
print(f"\nIs orthogonal set a basis? {is_basis}")
print(f"Dimension: {len(orthogonal_basis)}")
```

## Summary

In this chapter, we covered:
- Definition and testing of linear independence
- Basis and coordinate systems
- Change of basis transformations
- Gram-Schmidt orthogonalization
- Applications in feature selection and PCA

Linear independence and basis are fundamental for understanding vector spaces and coordinate transformations in machine learning.

## Next Steps

In the next chapter, we'll explore matrix decompositions, which are powerful tools for understanding matrix structure and solving systems of equations. 