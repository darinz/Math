# Linear Independence and Basis

[![Chapter](https://img.shields.io/badge/Chapter-6-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Linear_Independence-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Linear independence and basis are fundamental concepts that determine the structure and dimension of vector spaces. Understanding these concepts is crucial for solving systems of equations, performing coordinate transformations, and analyzing data in machine learning. In ML, the concepts of independence and basis underlie feature selection, dimensionality reduction, and the expressiveness of models.

### Why Linear Independence and Basis Matter in AI/ML

1. **Feature Selection**: Redundant features are linearly dependent; removing them improves model efficiency
2. **Dimensionality Reduction**: PCA finds a new basis of independent directions (principal components)
3. **Model Expressiveness**: The basis determines the space of possible solutions
4. **Coordinate Systems**: Changing basis is essential for understanding embeddings and transformations

## Linear Independence

A set of vectors $\{v_1, v_2, \ldots, v_n\}$ is linearly independent if the only solution to:
$$c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = 0$$
is $c_1 = c_2 = \cdots = c_n = 0$ (the trivial solution).

### Mathematical Definition
Vectors $v_1, v_2, \ldots, v_n$ are linearly independent if:
- The equation $c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = 0$ has only the trivial solution
- No vector in the set can be written as a linear combination of the others
- The rank of the matrix $[v_1\ v_2\ \ldots\ v_n]$ equals $n$

**Geometric Interpretation:**
- In $\mathbb{R}^2$, two vectors are independent if they are not collinear
- In $\mathbb{R}^3$, three vectors are independent if they do not all lie in the same plane

**Why It Matters:**
- The maximum number of linearly independent vectors in a space is its dimension
- Basis vectors must be linearly independent
- In ML, linearly dependent features do not add new information

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

# Example 1: Linearly independent vectors (standard basis in ℝ³)
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

independent_vectors = [v1, v2, v3]
print("Example 1: Standard basis vectors")
print(f"Are linearly independent: {is_linearly_independent(independent_vectors)}")

# Example 2: Linearly dependent vectors (v4 = v1 + v2)
v4 = np.array([1, 1, 0])
dependent_vectors = [v1, v2, v4]
print("\nExample 2: Including dependent vector")
print(f"Are linearly independent: {is_linearly_independent(dependent_vectors)}")
```

### Testing Linear Independence (Detailed)
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

A **basis** for a vector space $V$ is a linearly independent set that spans $V$.

### Properties of a Basis
1. **Linear Independence**: All vectors in the basis are linearly independent
2. **Spanning**: Every vector in $V$ can be written as a linear combination of basis vectors
3. **Minimal**: No proper subset spans $V$
4. **Unique Representation**: Each vector has a unique representation in terms of the basis

**Geometric Interpretation:**
- In $\mathbb{R}^2$, any two non-collinear vectors form a basis
- In $\mathbb{R}^3$, any three non-coplanar vectors form a basis
- The standard basis for $\mathbb{R}^n$ is $\{e_1, \ldots, e_n\}$

**Why It Matters:**
- The basis provides a coordinate system for the space
- Dimensionality reduction (PCA) finds a new basis for the data
- The number of features in ML is the dimension of the feature space

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

The concept of representing vectors in different bases is fundamental to linear algebra and has profound implications in machine learning. When we represent a vector in different bases, we're essentially changing our "coordinate system" or "perspective" for describing the same mathematical object.

**Mathematical Foundation:**
Given a vector space V with basis B = {b₁, b₂, ..., bₙ}, any vector v ∈ V can be uniquely written as:
v = c₁b₁ + c₂b₂ + ... + cₙbₙ

The coefficients c₁, c₂, ..., cₙ are called the coordinates of v with respect to basis B, denoted [v]B.

**Key Properties:**
1. **Uniqueness**: Each vector has exactly one representation in a given basis
2. **Completeness**: Every vector can be represented in any basis
3. **Linearity**: The coordinate mapping preserves vector operations

**Geometric Interpretation:**
Think of changing bases as changing the "ruler" or "measuring stick" we use to describe vectors. Just as we can describe a location using different coordinate systems (Cartesian, polar, etc.), we can describe vectors using different bases.

```python
def vector_in_basis(vector, basis):
    """
    Find coordinates of vector in given basis
    
    Mathematical approach:
    We solve the system: vector = c₁b₁ + c₂b₂ + ... + cₙbₙ
    This is equivalent to solving: B @ coordinates = vector
    where B is the matrix whose columns are the basis vectors
    
    Parameters:
    vector: numpy array - the vector to represent
    basis: list of numpy arrays - the basis vectors
    
    Returns:
    numpy array - coordinates of vector in the basis
    """
    # Construct basis matrix B = [b₁ | b₂ | ... | bₙ]
    basis_matrix = np.column_stack(basis)
    
    # Solve B @ coordinates = vector
    # This gives us the unique coordinates
    coordinates = np.linalg.solve(basis_matrix, vector)
    
    return coordinates

def reconstruct_vector(coordinates, basis):
    """
    Reconstruct vector from coordinates in basis
    
    This is the inverse operation: v = c₁b₁ + c₂b₂ + ... + cₙbₙ
    
    Parameters:
    coordinates: numpy array - coordinates in the basis
    basis: list of numpy arrays - the basis vectors
    
    Returns:
    numpy array - the reconstructed vector
    """
    basis_matrix = np.column_stack(basis)
    vector = basis_matrix @ coordinates
    return vector

def verify_basis_representation(vector, basis, coordinates):
    """
    Verify that coordinates correctly represent vector in basis
    
    Parameters:
    vector: numpy array - original vector
    basis: list of numpy arrays - basis vectors
    coordinates: numpy array - coordinates to verify
    
    Returns:
    bool - True if representation is correct
    """
    reconstructed = reconstruct_vector(coordinates, basis)
    error = np.linalg.norm(vector - reconstructed)
    return error < 1e-10, error

# Example: Vector in different bases
print("=== Vector Representation in Different Bases ===")

v = np.array([3, 4])
print(f"Original vector: {v}")

# Standard basis (canonical basis)
std_basis = [np.array([1, 0]), np.array([0, 1])]
coords_std = vector_in_basis(v, std_basis)
print(f"\nStandard basis: {[b.tolist() for b in std_basis]}")
print(f"Coordinates in standard basis: {coords_std}")

# Verify reconstruction
is_correct, error = verify_basis_representation(v, std_basis, coords_std)
print(f"Reconstruction correct: {is_correct} (error: {error:.2e})")

# Different basis (non-orthogonal)
new_basis = [np.array([1, 1]), np.array([1, -1])]
coords_new = vector_in_basis(v, new_basis)
print(f"\nNew basis: {[b.tolist() for b in new_basis]}")
print(f"Coordinates in new basis: {coords_new}")

# Verify reconstruction
is_correct, error = verify_basis_representation(v, new_basis, coords_new)
print(f"Reconstruction correct: {is_correct} (error: {error:.2e})")

# Geometric interpretation
print(f"\nGeometric interpretation:")
print(f"In standard basis: {coords_std[0]:.2f} * (1,0) + {coords_std[1]:.2f} * (0,1)")
print(f"In new basis: {coords_new[0]:.2f} * (1,1) + {coords_new[1]:.2f} * (1,-1)")

# Test with multiple vectors
test_vectors = [np.array([1, 0]), np.array([0, 1]), np.array([2, 3])]
print(f"\n=== Testing Multiple Vectors ===")

for i, test_v in enumerate(test_vectors):
    coords_std = vector_in_basis(test_v, std_basis)
    coords_new = vector_in_basis(test_v, new_basis)
    
    print(f"Vector {test_v}:")
    print(f"  Standard coords: {coords_std}")
    print(f"  New basis coords: {coords_new}")
    
    # Verify both representations
    is_correct_std, _ = verify_basis_representation(test_v, std_basis, coords_std)
    is_correct_new, _ = verify_basis_representation(test_v, new_basis, coords_new)
    print(f"  Both correct: {is_correct_std and is_correct_new}")
```

## Change of Basis

### Change of Basis Matrix

The change of basis transformation is a fundamental operation that allows us to convert between different coordinate representations of the same vector space. This is crucial in machine learning for feature transformations, dimensionality reduction, and understanding data from different perspectives.

**Mathematical Foundation:**
Given two bases B₁ and B₂ for vector space V, the change of basis matrix P satisfies:
[v]B₂ = P⁻¹[v]B₁

where [v]B₁ and [v]B₂ are the coordinate representations of vector v in bases B₁ and B₂ respectively.

**Key Properties:**
1. **Invertibility**: P is always invertible
2. **Composition**: P₁→₂ @ P₂→₃ = P₁→₃
3. **Identity**: P₁→₁ = I

**Geometric Interpretation:**
The change of basis matrix P tells us how to "rotate" or "transform" our coordinate system. Each column of P represents the coordinates of the new basis vectors in the old basis.

```python
def change_of_basis_matrix(old_basis, new_basis):
    """
    Find change of basis matrix from old_basis to new_basis
    
    Mathematical approach:
    P = [new_basis]_old_basis
    This matrix satisfies: [v]_new = P^(-1) [v]_old
    
    The matrix P has the property that:
    P @ [v]_old = [v]_new
    
    Parameters:
    old_basis: list of numpy arrays - original basis
    new_basis: list of numpy arrays - new basis
    
    Returns:
    numpy array - change of basis matrix P
    """
    # Construct matrices for both bases
    old_matrix = np.column_stack(old_basis)
    new_matrix = np.column_stack(new_basis)
    
    # P = new_matrix @ old_matrix^(-1)
    # This gives us the transformation matrix
    P = new_matrix @ np.linalg.inv(old_matrix)
    
    return P

def change_coordinates(vector, old_basis, new_basis):
    """
    Change vector coordinates from old_basis to new_basis
    
    This is the direct application of the change of basis transformation
    
    Parameters:
    vector: numpy array - vector in old basis coordinates
    old_basis: list of numpy arrays - original basis
    new_basis: list of numpy arrays - new basis
    
    Returns:
    numpy array - coordinates in new basis
    """
    P = change_of_basis_matrix(old_basis, new_basis)
    
    # Get coordinates in old basis
    coords_old = vector_in_basis(vector, old_basis)
    
    # Apply change of basis transformation
    coords_new = np.linalg.inv(P) @ coords_old
    
    return coords_new

def verify_change_of_basis(old_basis, new_basis, vector):
    """
    Verify change of basis transformation
    
    Parameters:
    old_basis: list of numpy arrays - original basis
    new_basis: list of numpy arrays - new basis
    vector: numpy array - test vector
    
    Returns:
    bool - True if transformation is correct
    """
    # Method 1: Direct computation
    coords_old = vector_in_basis(vector, old_basis)
    coords_new_direct = vector_in_basis(vector, new_basis)
    
    # Method 2: Using change of basis matrix
    P = change_of_basis_matrix(old_basis, new_basis)
    coords_new_via_P = np.linalg.inv(P) @ coords_old
    
    # Compare results
    error = np.linalg.norm(coords_new_direct - coords_new_via_P)
    return error < 1e-10, error

# Example: Change of basis
print("\n=== Change of Basis Transformation ===")

old_basis = [np.array([1, 0]), np.array([0, 1])]  # Standard basis
new_basis = [np.array([1, 1]), np.array([1, -1])]  # New basis

vector = np.array([3, 4])

print(f"Vector: {vector}")
print(f"Old basis: {[b.tolist() for b in old_basis]}")
print(f"New basis: {[b.tolist() for b in new_basis]}")

# Method 1: Direct computation
coords_old = vector_in_basis(vector, old_basis)
coords_new = vector_in_basis(vector, new_basis)

print(f"\nMethod 1 - Direct computation:")
print(f"Coordinates in old basis: {coords_old}")
print(f"Coordinates in new basis: {coords_new}")

# Method 2: Using change of basis matrix
P = change_of_basis_matrix(old_basis, new_basis)
coords_new_via_P = np.linalg.inv(P) @ coords_old

print(f"\nMethod 2 - Using change of basis matrix:")
print(f"Change of basis matrix P:")
print(P)
print(f"Coordinates via change of basis: {coords_new_via_P}")

# Verify the transformation
is_correct, error = verify_change_of_basis(old_basis, new_basis, vector)
print(f"\nTransformation verification:")
print(f"Methods agree: {is_correct} (error: {error:.2e})")

# Properties of change of basis matrix
print(f"\nProperties of change of basis matrix:")
print(f"P shape: {P.shape}")
print(f"P is invertible: {np.linalg.det(P) != 0}")
print(f"P^(-1) @ P = I (error: {np.linalg.norm(np.eye(2) - P @ np.linalg.inv(P)):.2e})")

# Test with multiple vectors
test_vectors = [np.array([1, 0]), np.array([0, 1]), np.array([2, 3])]
print(f"\n=== Testing Change of Basis with Multiple Vectors ===")

for i, test_v in enumerate(test_vectors):
    coords_old = vector_in_basis(test_v, old_basis)
    coords_new_direct = vector_in_basis(test_v, new_basis)
    coords_new_via_P = np.linalg.inv(P) @ coords_old
    
    print(f"Vector {test_v}:")
    print(f"  Old coords: {coords_old}")
    print(f"  New coords (direct): {coords_new_direct}")
    print(f"  New coords (via P): {coords_new_via_P}")
    
    error = np.linalg.norm(coords_new_direct - coords_new_via_P)
    print(f"  Agreement: {error < 1e-10} (error: {error:.2e})")
```

## Gram-Schmidt Orthogonalization

### Orthogonal Basis

The Gram-Schmidt process is a method for converting a set of linearly independent vectors into an orthogonal (or orthonormal) set that spans the same subspace. This is fundamental in machine learning for creating orthogonal features, implementing QR decomposition, and building orthogonal bases for numerical stability.

**Mathematical Foundation:**
Given linearly independent vectors v₁, v₂, ..., vₙ, the Gram-Schmidt process constructs orthogonal vectors u₁, u₂, ..., uₙ as follows:

u₁ = v₁
u₂ = v₂ - proj_u₁(v₂)
u₃ = v₃ - proj_u₁(v₃) - proj_u₂(v₃)
...

where proj_u(v) = (v·u)/(u·u) * u is the projection of v onto u.

**Key Properties:**
1. **Preserves span**: span{u₁, u₂, ..., uₙ} = span{v₁, v₂, ..., vₙ}
2. **Orthogonality**: uᵢ·uⱼ = 0 for i ≠ j
3. **Uniqueness**: The orthogonal set is unique up to scaling

**Geometric Interpretation:**
Gram-Schmidt is like "straightening" a set of vectors. We start with the first vector, then "subtract out" the component of each subsequent vector that lies in the direction of the previous orthogonal vectors, leaving only the "perpendicular" component.

```python
def gram_schmidt(vectors, normalize=True):
    """
    Apply Gram-Schmidt orthogonalization to vectors
    
    Mathematical approach:
    For each vector v_i, we subtract its projection onto all previous
    orthogonal vectors u_1, u_2, ..., u_{i-1}
    
    u_i = v_i - Σ_{j=1}^{i-1} proj_{u_j}(v_i)
    where proj_u(v) = (v·u)/(u·u) * u
    
    Parameters:
    vectors: list of numpy arrays - linearly independent vectors
    normalize: bool - whether to normalize the orthogonal vectors
    
    Returns:
    list of numpy arrays - orthogonal (or orthonormal) basis
    """
    if not vectors:
        return []
    
    orthogonal_basis = []
    
    for i, v in enumerate(vectors):
        # Start with original vector
        u = v.copy().astype(float)
        
        # Subtract projections onto previous orthogonal vectors
        for j in range(i):
            # Compute projection: proj_u_j(v_i) = (v_i · u_j) / (u_j · u_j) * u_j
            numerator = np.dot(v, orthogonal_basis[j])
            denominator = np.dot(orthogonal_basis[j], orthogonal_basis[j])
            
            if abs(denominator) > 1e-10:  # Avoid division by zero
                proj_coeff = numerator / denominator
                u = u - proj_coeff * orthogonal_basis[j]
        
        # Check if resulting vector is non-zero
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid zero vectors
            if normalize:
                u = u / norm  # Normalize to unit length
            orthogonal_basis.append(u)
        else:
            print(f"Warning: Vector {i+1} became zero after orthogonalization")
    
    return orthogonal_basis

def verify_gram_schmidt(original_vectors, orthogonal_vectors):
    """
    Verify Gram-Schmidt orthogonalization
    
    Parameters:
    original_vectors: list of numpy arrays - original vectors
    orthogonal_vectors: list of numpy arrays - orthogonal vectors
    
    Returns:
    dict - verification results
    """
    results = {}
    
    # Check orthogonality
    max_dot_product = 0
    for i in range(len(orthogonal_vectors)):
        for j in range(i+1, len(orthogonal_vectors)):
            dot_product = abs(np.dot(orthogonal_vectors[i], orthogonal_vectors[j]))
            max_dot_product = max(max_dot_product, dot_product)
    
    results['orthogonal'] = max_dot_product < 1e-10
    
    # Check normalization (if applicable)
    norms = [np.linalg.norm(u) for u in orthogonal_vectors]
    results['normalized'] = all(abs(norm - 1.0) < 1e-10 for norm in norms)
    
    # Check span preservation (simplified)
    if len(original_vectors) == len(orthogonal_vectors):
        results['span_preserved'] = True
    else:
        results['span_preserved'] = False
    
    return results

def gram_schmidt_with_projection_details(vectors):
    """
    Gram-Schmidt with detailed projection information
    
    Parameters:
    vectors: list of numpy arrays - linearly independent vectors
    
    Returns:
    tuple - (orthogonal_basis, projection_details)
    """
    if not vectors:
        return [], []
    
    orthogonal_basis = []
    projection_details = []
    
    for i, v in enumerate(vectors):
        u = v.copy().astype(float)
        projections = []
        
        print(f"\nProcessing vector v{i+1} = {v}")
        
        for j in range(i):
            numerator = np.dot(v, orthogonal_basis[j])
            denominator = np.dot(orthogonal_basis[j], orthogonal_basis[j])
            
            if abs(denominator) > 1e-10:
                proj_coeff = numerator / denominator
                proj_vector = proj_coeff * orthogonal_basis[j]
                u = u - proj_vector
                
                projections.append({
                    'onto': j,
                    'coefficient': proj_coeff,
                    'projection': proj_vector
                })
                
                print(f"  Subtract projection onto u{j+1}: {proj_coeff:.3f} * u{j+1}")
        
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            u_normalized = u / norm
            orthogonal_basis.append(u_normalized)
            projection_details.append(projections)
            
            print(f"  Result: u{i+1} = {u_normalized}")
        else:
            print(f"  Warning: Vector became zero!")
    
    return orthogonal_basis, projection_details

# Example: Gram-Schmidt orthogonalization
print("\n=== Gram-Schmidt Orthogonalization ===")

vectors = [
    np.array([1, 1, 0]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1])
]

print("Original vectors:")
for i, v in enumerate(vectors):
    print(f"v{i+1} = {v}")

# Apply Gram-Schmidt with detailed output
orthogonal_basis, details = gram_schmidt_with_projection_details(vectors)

print(f"\nOrthogonal basis:")
for i, u in enumerate(orthogonal_basis):
    print(f"u{i+1} = {u}")

# Verify orthogonality
print(f"\nOrthogonality check:")
max_dot_product = 0
for i in range(len(orthogonal_basis)):
    for j in range(i+1, len(orthogonal_basis)):
        dot_product = np.dot(orthogonal_basis[i], orthogonal_basis[j])
        max_dot_product = max(max_dot_product, abs(dot_product))
        print(f"u{i+1} · u{j+1} = {dot_product:.2e}")

print(f"Maximum off-diagonal dot product: {max_dot_product:.2e}")

# Verify normalization
print(f"\nNormalization check:")
for i, u in enumerate(orthogonal_basis):
    norm = np.linalg.norm(u)
    print(f"||u{i+1}|| = {norm:.6f}")

# Comprehensive verification
verification = verify_gram_schmidt(vectors, orthogonal_basis)
print(f"\nVerification results:")
for key, value in verification.items():
    print(f"  {key}: {value}")

# Test with different vector sets
print(f"\n=== Testing Different Vector Sets ===")

# Set 1: Already orthogonal vectors
orthogonal_vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1])
]

orth_basis1 = gram_schmidt(orthogonal_vectors)
print(f"Already orthogonal vectors:")
for i, u in enumerate(orth_basis1):
    print(f"  u{i+1} = {u}")

# Set 2: Linearly dependent vectors
dependent_vectors = [
    np.array([1, 0, 0]),
    np.array([2, 0, 0]),  # Multiple of first vector
    np.array([0, 1, 0])
]

orth_basis2 = gram_schmidt(dependent_vectors)
print(f"\nLinearly dependent vectors:")
print(f"Original count: {len(dependent_vectors)}")
print(f"Orthogonal count: {len(orth_basis2)}")
for i, u in enumerate(orth_basis2):
    print(f"  u{i+1} = {u}")
```

## Applications in Machine Learning

Linear independence and basis concepts are fundamental to many machine learning algorithms and techniques. Understanding these concepts helps us design better features, reduce dimensionality, and improve model performance.

### Feature Selection and Dimensionality Reduction

**Mathematical Foundation:**
In machine learning, we often work with feature matrices X ∈ ℝ^(n×d) where n is the number of samples and d is the number of features. Linear dependencies among features can cause:

1. **Multicollinearity**: Features that are linearly dependent can cause numerical instability in regression models
2. **Redundant Information**: Dependent features don't add new information
3. **Overfitting**: Models may fit to noise in dependent features

**Key Concepts:**
- **Feature Rank**: The rank of the feature matrix determines the maximum number of linearly independent features
- **Feature Selection**: Choosing a subset of linearly independent features that preserve the most information
- **Dimensionality Reduction**: Reducing the number of features while maintaining model performance

```python
def select_linearly_independent_features(feature_matrix, tol=1e-10, method='qr'):
    """
    Select linearly independent features from matrix
    
    Mathematical approaches:
    1. QR decomposition with pivoting: A = Q @ R @ P^T
       where P is a permutation matrix that identifies independent columns
    2. SVD decomposition: A = U @ Σ @ V^T
       where the rank is determined by non-zero singular values
    3. Gaussian elimination: Row reduction to identify pivot columns
    
    Parameters:
    feature_matrix: numpy array - feature matrix (samples × features)
    tol: float - tolerance for numerical rank determination
    method: str - 'qr', 'svd', or 'gaussian'
    
    Returns:
    tuple - (independent_features, independent_indices, rank)
    """
    n_samples, n_features = feature_matrix.shape
    
    if method == 'qr':
        # QR decomposition with pivoting
        Q, R, P = np.linalg.qr(feature_matrix, mode='full', pivoting=True)
        
        # Find rank by counting non-zero diagonal elements of R
        rank = np.sum(np.abs(np.diag(R)) > tol)
        
        # Select independent features
        independent_indices = P[:rank]
        independent_features = feature_matrix[:, independent_indices]
        
    elif method == 'svd':
        # SVD decomposition
        U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
        
        # Find rank by counting non-zero singular values
        rank = np.sum(S > tol)
        
        # Select independent features (first rank columns of U)
        independent_features = U[:, :rank] @ np.diag(S[:rank])
        independent_indices = np.arange(rank)
        
    elif method == 'gaussian':
        # Gaussian elimination approach
        A = feature_matrix.copy()
        n_rows, n_cols = A.shape
        rank = 0
        independent_indices = []
        
        for col in range(n_cols):
            # Find pivot
            pivot_row = rank
            for row in range(rank, n_rows):
                if abs(A[row, col]) > abs(A[pivot_row, col]):
                    pivot_row = row
            
            if abs(A[pivot_row, col]) > tol:
                # Swap rows if necessary
                if pivot_row != rank:
                    A[rank, :], A[pivot_row, :] = A[pivot_row, :].copy(), A[rank, :].copy()
                
                # Eliminate column
                for row in range(rank + 1, n_rows):
                    factor = A[row, col] / A[rank, col]
                    A[row, :] -= factor * A[rank, :]
                
                independent_indices.append(col)
                rank += 1
        
        independent_features = feature_matrix[:, independent_indices]
        independent_indices = np.array(independent_indices)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return independent_features, independent_indices, rank

def analyze_feature_dependencies(feature_matrix, feature_names=None):
    """
    Analyze linear dependencies in feature matrix
    
    Parameters:
    feature_matrix: numpy array - feature matrix
    feature_names: list - names of features (optional)
    
    Returns:
    dict - analysis results
    """
    n_samples, n_features = feature_matrix.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(feature_matrix.T)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > 0.95:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })
    
    # Compute rank and condition number
    rank = np.linalg.matrix_rank(feature_matrix)
    condition_number = np.linalg.cond(feature_matrix)
    
    # Find independent features
    independent_features, independent_indices, computed_rank = select_linearly_independent_features(feature_matrix)
    
    return {
        'original_shape': feature_matrix.shape,
        'rank': rank,
        'computed_rank': computed_rank,
        'condition_number': condition_number,
        'independent_features': independent_features,
        'independent_indices': independent_indices,
        'independent_feature_names': [feature_names[i] for i in independent_indices],
        'high_correlation_pairs': high_corr_pairs,
        'correlation_matrix': corr_matrix
    }

def verify_feature_selection(original_matrix, independent_matrix, independent_indices):
    """
    Verify that selected features preserve the original space
    
    Parameters:
    original_matrix: numpy array - original feature matrix
    independent_matrix: numpy array - selected independent features
    independent_indices: numpy array - indices of selected features
    
    Returns:
    dict - verification results
    """
    # Check if independent features can reconstruct original space
    # This is a simplified check - in practice, we'd need more sophisticated methods
    
    # Compute projection matrix
    P = independent_matrix @ np.linalg.pinv(independent_matrix)
    
    # Project original matrix onto independent subspace
    projected_matrix = P @ original_matrix
    
    # Compute reconstruction error
    reconstruction_error = np.linalg.norm(original_matrix - projected_matrix, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(original_matrix, 'fro')
    
    # Check rank preservation
    original_rank = np.linalg.matrix_rank(original_matrix)
    independent_rank = np.linalg.matrix_rank(independent_matrix)
    
    return {
        'reconstruction_error': reconstruction_error,
        'relative_error': relative_error,
        'original_rank': original_rank,
        'independent_rank': independent_rank,
        'rank_preserved': original_rank == independent_rank,
        'good_reconstruction': relative_error < 1e-10
    }

# Example: Feature selection
print("=== Feature Selection and Dimensionality Reduction ===")

np.random.seed(42)
# Create feature matrix with some linear dependencies
n_samples, n_features = 100, 5
features = np.random.randn(n_samples, n_features)
feature_names = [f"X{i+1}" for i in range(n_features)]

# Add linear dependencies
features[:, 2] = 2 * features[:, 0] + 3 * features[:, 1]  # Feature 2 is dependent
features[:, 4] = features[:, 0] - features[:, 1]          # Feature 4 is dependent

print(f"Original feature matrix shape: {features.shape}")
print(f"Original rank: {np.linalg.matrix_rank(features)}")

# Analyze feature dependencies
analysis = analyze_feature_dependencies(features, feature_names)
print(f"\nFeature dependency analysis:")
print(f"Rank: {analysis['rank']}")
print(f"Condition number: {analysis['condition_number']:.2e}")
print(f"High correlation pairs: {len(analysis['high_correlation_pairs'])}")

for pair in analysis['high_correlation_pairs']:
    print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")

# Select independent features using different methods
methods = ['qr', 'svd', 'gaussian']
for method in methods:
    print(f"\n--- Method: {method.upper()} ---")
    independent_features, indices, rank = select_linearly_independent_features(features, method=method)
    
    print(f"Selected features: {[feature_names[i] for i in indices]}")
    print(f"Independent features shape: {independent_features.shape}")
    print(f"Rank: {rank}")
    
    # Verify selection
    verification = verify_feature_selection(features, independent_features, indices)
    print(f"Reconstruction error: {verification['relative_error']:.2e}")
    print(f"Rank preserved: {verification['rank_preserved']}")

# Test with real-world scenario
print(f"\n=== Real-world Feature Selection Example ===")

# Simulate a dataset with known dependencies
np.random.seed(123)
n_samples = 200

# Generate base features
base_features = np.random.randn(n_samples, 3)

# Create derived features
derived_features = np.column_stack([
    base_features[:, 0] + 0.1 * np.random.randn(n_samples),  # Slightly noisy copy
    2 * base_features[:, 1] - base_features[:, 2],            # Linear combination
    base_features[:, 0] * base_features[:, 1],                # Non-linear feature
    base_features[:, 2] ** 2                                  # Quadratic feature
])

# Combine all features
all_features = np.column_stack([base_features, derived_features])
feature_names = ['Base1', 'Base2', 'Base3', 'Derived1', 'Derived2', 'Derived3', 'Derived4']

print(f"Combined feature matrix shape: {all_features.shape}")
print(f"Combined rank: {np.linalg.matrix_rank(all_features)}")

# Analyze and select features
analysis = analyze_feature_dependencies(all_features, feature_names)
independent_features, indices, rank = select_linearly_independent_features(all_features)

print(f"\nSelected independent features:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}")

print(f"Reduced from {all_features.shape[1]} to {len(indices)} features")
print(f"Rank: {rank}")
```

### Principal Component Analysis (PCA)

**Mathematical Foundation:**
PCA finds an orthogonal basis for the data that maximizes variance along each direction. The principal components are the eigenvectors of the covariance matrix, ordered by their corresponding eigenvalues.

**Key Properties:**
1. **Orthogonality**: Principal components are orthogonal to each other
2. **Variance Maximization**: Each component captures maximum variance in the remaining directions
3. **Dimensionality Reduction**: We can truncate to k components while preserving most variance

**Geometric Interpretation:**
PCA finds the "natural" coordinate system for the data. The first principal component points in the direction of maximum variance, the second in the direction of maximum variance perpendicular to the first, and so on.

```python
def pca_basis(data, n_components=None, center=True, scale=False):
    """
    Find PCA basis for data
    
    Mathematical approach:
    1. Center the data: X_centered = X - mean(X)
    2. Compute covariance matrix: C = (1/n) X_centered^T @ X_centered
    3. Find eigenvalues and eigenvectors: C @ v = λ @ v
    4. Sort by eigenvalues (descending order)
    5. Principal components are the eigenvectors
    
    Parameters:
    data: numpy array - data matrix (samples × features)
    n_components: int - number of components to return
    center: bool - whether to center the data
    scale: bool - whether to scale the data
    
    Returns:
    tuple - (basis_vectors, eigenvalues, explained_variance_ratio, mean_vector)
    """
    n_samples, n_features = data.shape
    
    # Center the data
    if center:
        mean_vector = np.mean(data, axis=0)
        data_centered = data - mean_vector
    else:
        mean_vector = np.zeros(n_features)
        data_centered = data
    
    # Scale the data (optional)
    if scale:
        std_vector = np.std(data_centered, axis=0)
        data_centered = data_centered / std_vector
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Determine number of components
    if n_components is None:
        n_components = len(eigenvalues)
    elif n_components > len(eigenvalues):
        n_components = len(eigenvalues)
    
    # Extract basis vectors and eigenvalues
    basis_vectors = [eigenvectors_sorted[:, i] for i in range(n_components)]
    eigenvalues_selected = eigenvalues_sorted[:n_components]
    
    # Compute explained variance ratio
    total_variance = np.sum(eigenvalues_sorted)
    explained_variance_ratio = eigenvalues_selected / total_variance
    
    return basis_vectors, eigenvalues_selected, explained_variance_ratio, mean_vector

def project_data_pca(data, basis_vectors, mean_vector=None):
    """
    Project data onto PCA basis
    
    Parameters:
    data: numpy array - data to project
    basis_vectors: list - PCA basis vectors
    mean_vector: numpy array - mean vector (if data was centered)
    
    Returns:
    numpy array - projected data
    """
    if mean_vector is not None:
        data_centered = data - mean_vector
    else:
        data_centered = data
    
    # Project onto basis
    basis_matrix = np.column_stack(basis_vectors)
    projected_data = data_centered @ basis_matrix
    
    return projected_data

def reconstruct_data_pca(projected_data, basis_vectors, mean_vector=None):
    """
    Reconstruct data from PCA projection
    
    Parameters:
    projected_data: numpy array - projected data
    basis_vectors: list - PCA basis vectors
    mean_vector: numpy array - mean vector (if data was centered)
    
    Returns:
    numpy array - reconstructed data
    """
    # Reconstruct from basis
    basis_matrix = np.column_stack(basis_vectors)
    reconstructed_centered = projected_data @ basis_matrix.T
    
    if mean_vector is not None:
        reconstructed_data = reconstructed_centered + mean_vector
    else:
        reconstructed_data = reconstructed_centered
    
    return reconstructed_data

def analyze_pca_quality(original_data, reconstructed_data):
    """
    Analyze quality of PCA reconstruction
    
    Parameters:
    original_data: numpy array - original data
    reconstructed_data: numpy array - reconstructed data
    
    Returns:
    dict - quality metrics
    """
    # Compute reconstruction error
    mse = np.mean((original_data - reconstructed_data) ** 2)
    rmse = np.sqrt(mse)
    
    # Compute relative error
    relative_error = np.linalg.norm(original_data - reconstructed_data, 'fro') / np.linalg.norm(original_data, 'fro')
    
    # Compute explained variance
    total_variance = np.var(original_data, axis=0).sum()
    explained_variance = total_variance - np.var(original_data - reconstructed_data, axis=0).sum()
    explained_variance_ratio = explained_variance / total_variance
    
    return {
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'explained_variance_ratio': explained_variance_ratio
    }

# Example: PCA basis
print("\n=== Principal Component Analysis ===")

np.random.seed(42)
# Create correlated data
n_samples = 100
data = np.random.randn(n_samples, 3)
data[:, 2] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n_samples)  # Add correlation

print(f"Data shape: {data.shape}")
print(f"Data correlation matrix:")
print(np.corrcoef(data.T))

# Find PCA basis
pca_basis_vectors, eigenvalues, explained_variance_ratio, mean_vector = pca_basis(data, n_components=2)

print(f"\nPCA basis vectors:")
for i, v in enumerate(pca_basis_vectors):
    print(f"PC{i+1} = {v}")

print(f"\nEigenvalues:")
for i, eigenval in enumerate(eigenvalues):
    print(f"λ{i+1} = {eigenval:.3f}")

print(f"\nExplained variance ratios:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")

print(f"Cumulative explained variance: {np.sum(explained_variance_ratio):.3f}")

# Project data onto PCA basis
projected_data = project_data_pca(data, pca_basis_vectors, mean_vector)
print(f"\nProjected data shape: {projected_data.shape}")

# Reconstruct data
reconstructed_data = reconstruct_data_pca(projected_data, pca_basis_vectors, mean_vector)

# Analyze reconstruction quality
quality = analyze_pca_quality(data, reconstructed_data)
print(f"\nReconstruction quality:")
for key, value in quality.items():
    print(f"  {key}: {value:.6f}")

# Test with different numbers of components
print(f"\n=== Testing Different Numbers of Components ===")

for n_comp in [1, 2, 3]:
    basis_vectors, eigenvalues, explained_variance, mean_vector = pca_basis(data, n_components=n_comp)
    
    projected = project_data_pca(data, basis_vectors, mean_vector)
    reconstructed = reconstruct_data_pca(projected, basis_vectors, mean_vector)
    
    quality = analyze_pca_quality(data, reconstructed)
    
    print(f"Components: {n_comp}")
    print(f"  Explained variance: {np.sum(explained_variance):.3f}")
    print(f"  Relative error: {quality['relative_error']:.6f}")

# Visualize PCA transformation
print(f"\n=== PCA Transformation Analysis ===")

# Original data statistics
print(f"Original data:")
print(f"  Mean: {np.mean(data, axis=0)}")
print(f"  Variance: {np.var(data, axis=0)}")
print(f"  Total variance: {np.sum(np.var(data, axis=0)):.3f}")

# Projected data statistics
print(f"\nProjected data (2 components):")
print(f"  Mean: {np.mean(projected_data, axis=0)}")
print(f"  Variance: {np.var(projected_data, axis=0)}")
print(f"  Total variance: {np.sum(np.var(projected_data, axis=0)):.3f}")

# Verify orthogonality of principal components
print(f"\nOrthogonality of principal components:")
for i in range(len(pca_basis_vectors)):
    for j in range(i+1, len(pca_basis_vectors)):
        dot_product = np.dot(pca_basis_vectors[i], pca_basis_vectors[j])
        print(f"  PC{i+1} · PC{j+1} = {dot_product:.2e}")

# Verify that principal components are unit vectors
print(f"\nNorm of principal components:")
for i, pc in enumerate(pca_basis_vectors):
    norm = np.linalg.norm(pc)
    print(f"  ||PC{i+1}|| = {norm:.6f}")
```

### Linear Independence in Neural Networks

**Mathematical Foundation:**
In neural networks, linear independence is crucial for:
1. **Weight Matrix Rank**: Ensures the network can learn diverse representations
2. **Feature Learning**: Independent neurons learn different features
3. **Gradient Flow**: Prevents vanishing/exploding gradients

**Key Concepts:**
- **Weight Matrix Conditioning**: Well-conditioned weight matrices have independent rows/columns
- **Feature Diversity**: Independent neurons capture different aspects of the data
- **Regularization**: Techniques like dropout promote independence

```python
def analyze_neural_network_weights(weight_matrices, layer_names=None):
    """
    Analyze linear independence in neural network weights
    
    Parameters:
    weight_matrices: list of numpy arrays - weight matrices for each layer
    layer_names: list - names of layers (optional)
    
    Returns:
    dict - analysis results
    """
    if layer_names is None:
        layer_names = [f"Layer_{i}" for i in range(len(weight_matrices))]
    
    results = {}
    
    for i, (weights, name) in enumerate(zip(weight_matrices, layer_names)):
        # Analyze rows (output neurons)
        row_rank = np.linalg.matrix_rank(weights)
        row_condition = np.linalg.cond(weights)
        
        # Analyze columns (input features)
        col_rank = np.linalg.matrix_rank(weights.T)
        col_condition = np.linalg.cond(weights.T)
        
        # Find independent rows and columns
        independent_rows, row_indices, _ = select_linearly_independent_features(weights)
        independent_cols, col_indices, _ = select_linearly_independent_features(weights.T)
        
        results[name] = {
            'shape': weights.shape,
            'row_rank': row_rank,
            'col_rank': col_rank,
            'row_condition': row_condition,
            'col_condition': col_condition,
            'independent_rows': independent_rows,
            'independent_cols': independent_cols,
            'row_indices': row_indices,
            'col_indices': col_indices,
            'row_independence_ratio': row_rank / weights.shape[0],
            'col_independence_ratio': col_rank / weights.shape[1]
        }
    
    return results

def promote_weight_independence(weight_matrix, method='orthogonal_init', strength=0.1):
    """
    Promote linear independence in weight matrix
    
    Parameters:
    weight_matrix: numpy array - weight matrix
    method: str - method to promote independence
    strength: float - strength of regularization
    
    Returns:
    numpy array - modified weight matrix
    """
    if method == 'orthogonal_init':
        # Initialize with orthogonal matrix
        U, _, Vt = np.linalg.svd(weight_matrix)
        orthogonal_matrix = U @ Vt
        return (1 - strength) * weight_matrix + strength * orthogonal_matrix
    
    elif method == 'rank_regularization':
        # Add regularization to promote full rank
        n_rows, n_cols = weight_matrix.shape
        min_dim = min(n_rows, n_cols)
        
        # Compute singular values
        U, S, Vt = np.linalg.svd(weight_matrix)
        
        # Promote larger singular values
        S_modified = S + strength * np.ones_like(S)
        
        return U @ np.diag(S_modified) @ Vt
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Example: Neural network weight analysis
print("\n=== Neural Network Weight Analysis ===")

# Simulate neural network weights
np.random.seed(42)
layer_sizes = [10, 8, 6, 4]
weight_matrices = []

for i in range(len(layer_sizes) - 1):
    # Create weight matrix with some dependencies
    weights = np.random.randn(layer_sizes[i+1], layer_sizes[i])
    
    # Add some linear dependencies
    if i > 0:
        weights[0, :] = 0.5 * weights[1, :] + 0.3 * weights[2, :]
    
    weight_matrices.append(weights)

layer_names = [f"Hidden_{i+1}" for i in range(len(weight_matrices)-1)] + ["Output"]

# Analyze weights
analysis = analyze_neural_network_weights(weight_matrices, layer_names)

print("Weight matrix analysis:")
for layer_name, layer_analysis in analysis.items():
    print(f"\n{layer_name}:")
    print(f"  Shape: {layer_analysis['shape']}")
    print(f"  Row rank: {layer_analysis['row_rank']}/{layer_analysis['shape'][0]} ({layer_analysis['row_independence_ratio']:.2f})")
    print(f"  Column rank: {layer_analysis['col_rank']}/{layer_analysis['shape'][1]} ({layer_analysis['col_independence_ratio']:.2f})")
    print(f"  Row condition number: {layer_analysis['row_condition']:.2e}")
    print(f"  Column condition number: {layer_analysis['col_condition']:.2e}")

# Promote independence in problematic layers
print(f"\n=== Promoting Weight Independence ===")

for i, (weights, layer_name) in enumerate(zip(weight_matrices, layer_names)):
    if analysis[layer_name]['row_independence_ratio'] < 0.9:
        print(f"Promoting independence in {layer_name}...")
        
        original_rank = analysis[layer_name]['row_rank']
        modified_weights = promote_weight_independence(weights, method='orthogonal_init', strength=0.1)
        
        # Analyze modified weights
        modified_analysis = analyze_neural_network_weights([modified_weights], [layer_name])
        modified_rank = modified_analysis[layer_name]['row_rank']
        
        print(f"  Original rank: {original_rank}")
        print(f"  Modified rank: {modified_rank}")
        print(f"  Improvement: {modified_rank - original_rank}")
        
        weight_matrices[i] = modified_weights
```

## Exercises

The following exercises will help you master the concepts of linear independence, basis, and their applications in machine learning. Each exercise builds upon the previous ones and includes both theoretical understanding and practical implementation.

### Exercise 1: Linear Independence Testing and Analysis

**Objective**: Develop a comprehensive understanding of linear independence testing and its geometric interpretation.

```python
# Test linear independence of different vector sets and analyze their properties
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

vectors_set3 = [
    np.array([1, 1, 1]),
    np.array([1, -1, 0]),
    np.array([0, 1, -1])
]

# Your tasks:
# 1. Test linear independence of all three sets using multiple methods
# 2. Find bases for each set and determine their dimensions
# 3. Analyze the geometric properties of each set
# 4. Compute the condition number of each set's matrix
# 5. Visualize the vectors in 3D space (if possible)
```

### Exercise 2: Advanced Change of Basis Transformations

**Objective**: Master change of basis transformations and understand their geometric interpretation.

```python
# Perform comprehensive change of basis analysis
old_basis = [np.array([1, 0]), np.array([0, 1])]  # Standard basis
new_basis = [np.array([2, 1]), np.array([1, 2])]  # New basis
vector = np.array([5, 3])

# Your tasks:
# 1. Find coordinates of vector in both bases
# 2. Compute the change of basis matrix P and verify its properties
# 3. Test the transformation with multiple vectors
# 4. Analyze the geometric interpretation of the transformation
# 5. Compute the determinant and condition number of P
# 6. Verify that P^(-1) @ P = I and P @ P^(-1) = I
```

### Exercise 3: Gram-Schmidt Process with Error Analysis

**Objective**: Implement and analyze the Gram-Schmidt process with comprehensive error checking.

```python
# Apply Gram-Schmidt to various vector sets with detailed analysis
vectors = [
    np.array([1, 1, 1]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1])
]

# Your tasks:
# 1. Apply Gram-Schmidt orthogonalization with detailed step-by-step output
# 2. Verify orthogonality of the resulting vectors
# 3. Check if the result forms a basis for the original space
# 4. Analyze numerical stability and error accumulation
# 5. Compare with different normalization strategies
# 6. Test with nearly dependent vectors to check robustness
```

### Exercise 4: Feature Selection and Dimensionality Reduction

**Objective**: Apply linear independence concepts to real-world feature selection problems.

```python
# Create a realistic dataset with known dependencies
np.random.seed(42)
n_samples = 500
n_features = 10

# Generate base features
base_features = np.random.randn(n_samples, 4)

# Create derived features with various dependencies
derived_features = np.column_stack([
    base_features[:, 0] + 0.1 * np.random.randn(n_samples),  # Nearly dependent
    2 * base_features[:, 1] - base_features[:, 2],            # Linear combination
    base_features[:, 0] * base_features[:, 1],                # Non-linear
    base_features[:, 2] ** 2,                                 # Quadratic
    base_features[:, 0] + base_features[:, 1] + base_features[:, 2],  # Sum
    base_features[:, 3] * 0.5                                 # Scaled copy
])

# Combine all features
all_features = np.column_stack([base_features, derived_features])
feature_names = ['Base1', 'Base2', 'Base3', 'Base4', 'Derived1', 'Derived2', 
                'Derived3', 'Derived4', 'Derived5', 'Derived6']

# Your tasks:
# 1. Analyze the rank and condition number of the feature matrix
# 2. Identify linear dependencies using correlation analysis
# 3. Apply different feature selection methods (QR, SVD, Gaussian)
# 4. Compare the results and explain differences
# 5. Verify that selected features preserve the original space
# 6. Analyze the impact on a simple regression model
```

### Exercise 5: PCA Implementation and Analysis

**Objective**: Implement PCA from scratch and analyze its properties.

```python
# Create a dataset with known structure for PCA analysis
np.random.seed(123)
n_samples = 300

# Generate data with known principal components
true_pc1 = np.array([0.8, 0.6, 0.0])
true_pc2 = np.array([0.0, 0.0, 1.0])
true_pc3 = np.array([-0.6, 0.8, 0.0])

# Create data along these directions with different variances
data = np.column_stack([
    3.0 * np.random.randn(n_samples) * true_pc1,
    1.5 * np.random.randn(n_samples) * true_pc2,
    0.5 * np.random.randn(n_samples) * true_pc3
])

# Add some noise
data += 0.1 * np.random.randn(n_samples, 3)

# Your tasks:
# 1. Implement PCA from scratch (without using sklearn)
# 2. Compare your implementation with the built-in functions
# 3. Analyze the eigenvalues and explained variance ratios
# 4. Project data onto different numbers of components
# 5. Reconstruct data and analyze reconstruction quality
# 6. Visualize the principal components and data projections
```

### Exercise 6: Neural Network Weight Analysis

**Objective**: Analyze and improve linear independence in neural network weights.

```python
# Simulate a neural network with known weight dependencies
np.random.seed(456)
layer_sizes = [20, 15, 10, 5]
weight_matrices = []

for i in range(len(layer_sizes) - 1):
    weights = np.random.randn(layer_sizes[i+1], layer_sizes[i])
    
    # Add some linear dependencies
    if i > 0:
        # Make some neurons nearly dependent
        weights[0, :] = 0.7 * weights[1, :] + 0.3 * weights[2, :] + 0.1 * np.random.randn(layer_sizes[i])
    
    weight_matrices.append(weights)

layer_names = [f"Hidden_{i+1}" for i in range(len(weight_matrices)-1)] + ["Output"]

# Your tasks:
# 1. Analyze the rank and condition number of each weight matrix
# 2. Identify dependent neurons in each layer
# 3. Apply techniques to promote weight independence
# 4. Compare different regularization methods
# 5. Analyze the impact on network capacity
# 6. Test the modified network on a simple classification task
```

## Solutions

### Solution 1: Linear Independence Testing and Analysis
```python
def comprehensive_linear_independence_analysis(vector_sets, set_names):
    """
    Comprehensive analysis of linear independence for multiple vector sets
    """
    results = {}
    
    for vectors, name in zip(vector_sets, set_names):
        print(f"\n=== Analysis of {name} ===")
        
        # Convert to matrix
        matrix = np.column_stack(vectors)
        
        # Basic properties
        shape = matrix.shape
        rank = np.linalg.matrix_rank(matrix)
        condition_number = np.linalg.cond(matrix)
        determinant = np.linalg.det(matrix) if shape[0] == shape[1] else None
        
        print(f"Matrix shape: {shape}")
        print(f"Rank: {rank}")
        print(f"Condition number: {condition_number:.2e}")
        if determinant is not None:
            print(f"Determinant: {determinant:.6f}")
        
        # Test linear independence
        is_independent = rank == len(vectors)
        print(f"Linearly independent: {is_independent}")
        
        # Find basis
        if is_independent:
            basis = vectors
            print("Basis: All vectors are independent")
        else:
            basis, indices, computed_rank = select_linearly_independent_features(matrix)
            print(f"Basis: {len(basis)} vectors (indices: {indices})")
        
        # Geometric analysis
        if shape[1] <= 3:  # Can visualize in 3D or less
            print("Geometric properties:")
            for i, v in enumerate(vectors):
                norm = np.linalg.norm(v)
                print(f"  Vector {i+1}: norm = {norm:.3f}")
            
            # Check angles between vectors
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    cos_angle = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    print(f"  Angle between v{i+1} and v{j+1}: {np.degrees(angle):.1f}°")
        
        results[name] = {
            'shape': shape,
            'rank': rank,
            'condition_number': condition_number,
            'determinant': determinant,
            'is_independent': is_independent,
            'basis': basis if is_independent else len(basis)
        }
    
    return results

# Apply comprehensive analysis
vector_sets = [vectors_set1, vectors_set2, vectors_set3]
set_names = ['Set 1', 'Set 2', 'Set 3']

analysis_results = comprehensive_linear_independence_analysis(vector_sets, set_names)

# Summary comparison
print(f"\n=== Summary Comparison ===")
for name, results in analysis_results.items():
    print(f"{name}:")
    print(f"  Rank: {results['rank']}/{results['shape'][1]}")
    print(f"  Independent: {results['is_independent']}")
    print(f"  Condition number: {results['condition_number']:.2e}")

# Additional geometric analysis for Set 1 (nearly dependent)
print(f"\n=== Detailed Analysis of Set 1 ===")
matrix1 = np.column_stack(vectors_set1)
print("Set 1 has nearly dependent vectors (notice the pattern in coordinates)")

# Check if vectors lie on a plane
# If three 3D vectors are linearly dependent, they lie on a plane
if len(vectors_set1) == 3:
    # Compute the volume of the parallelepiped
    volume = abs(np.linalg.det(matrix1))
    print(f"Volume of parallelepiped: {volume:.6f}")
    if volume < 1e-10:
        print("Vectors lie on a plane (volume ≈ 0)")
    else:
        print("Vectors span 3D space")

# For Set 2 (standard basis with one dependent vector)
print(f"\n=== Detailed Analysis of Set 2 ===")
print("Set 2 has the standard basis vectors plus a dependent vector")
print("This creates a 2D subspace in 3D space")

# For Set 3 (independent vectors)
print(f"\n=== Detailed Analysis of Set 3 ===")
print("Set 3 has three independent vectors")
print("These vectors span the entire 3D space")
```

### Solution 2: Advanced Change of Basis Transformations

```python
def comprehensive_change_of_basis_analysis(old_basis, new_basis, test_vectors):
    """
    Comprehensive analysis of change of basis transformation
    """
    print("=== Comprehensive Change of Basis Analysis ===")
    
    # 1. Find coordinates in both bases
    print(f"\n1. Coordinates in different bases:")
    for i, vector in enumerate(test_vectors):
        coords_old = vector_in_basis(vector, old_basis)
        coords_new = vector_in_basis(vector, new_basis)
        
        print(f"Vector {vector}:")
        print(f"  Old basis coords: {coords_old}")
        print(f"  New basis coords: {coords_new}")
    
    # 2. Compute change of basis matrix
    P = change_of_basis_matrix(old_basis, new_basis)
    print(f"\n2. Change of basis matrix P:")
    print(P)
    
    # 3. Verify properties of P
    print(f"\n3. Properties of change of basis matrix:")
    print(f"Shape: {P.shape}")
    print(f"Determinant: {np.linalg.det(P):.6f}")
    print(f"Condition number: {np.linalg.cond(P):.2e}")
    print(f"P is invertible: {np.linalg.det(P) != 0}")
    
    # 4. Verify P^(-1) @ P = I and P @ P^(-1) = I
    P_inv = np.linalg.inv(P)
    identity_error1 = np.linalg.norm(np.eye(P.shape[0]) - P_inv @ P)
    identity_error2 = np.linalg.norm(np.eye(P.shape[0]) - P @ P_inv)
    
    print(f"P^(-1) @ P = I (error: {identity_error1:.2e})")
    print(f"P @ P^(-1) = I (error: {identity_error2:.2e})")
    
    # 5. Test transformation with multiple vectors
    print(f"\n4. Testing transformation with multiple vectors:")
    for i, vector in enumerate(test_vectors):
        coords_old = vector_in_basis(vector, old_basis)
        coords_new_direct = vector_in_basis(vector, new_basis)
        coords_new_via_P = np.linalg.inv(P) @ coords_old
        
        error = np.linalg.norm(coords_new_direct - coords_new_via_P)
        print(f"Vector {vector}: transformation error = {error:.2e}")
    
    # 6. Geometric interpretation
    print(f"\n5. Geometric interpretation:")
    print(f"Old basis vectors:")
    for i, v in enumerate(old_basis):
        print(f"  b{i+1} = {v}")
    
    print(f"New basis vectors:")
    for i, v in enumerate(new_basis):
        print(f"  b'{i+1} = {v}")
    
    # Show how new basis vectors are expressed in old basis
    print(f"New basis vectors in old basis coordinates:")
    for i, v in enumerate(new_basis):
        coords = vector_in_basis(v, old_basis)
        print(f"  [b'{i+1}]_old = {coords}")
    
    return P, P_inv

# Apply comprehensive analysis
test_vectors = [
    np.array([5, 3]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([2, 2])
]

P, P_inv = comprehensive_change_of_basis_analysis(old_basis, new_basis, test_vectors)

# Additional analysis: eigenvalues and eigenvectors of P
print(f"\n6. Eigenvalue analysis of change of basis matrix:")
eigenvalues, eigenvectors = np.linalg.eig(P)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:")
for i, eigenvector in enumerate(eigenvectors.T):
    print(f"  λ{i+1} = {eigenvalues[i]:.3f}: {eigenvector}")

# Check if P is orthogonal
is_orthogonal = np.allclose(P @ P.T, np.eye(P.shape[0]))
print(f"P is orthogonal: {is_orthogonal}")

if not is_orthogonal:
    print("P is not orthogonal - this means the new basis is not orthonormal")
    print("The transformation involves both rotation and scaling")
```

### Solution 3: Gram-Schmidt Process with Error Analysis

```python
def gram_schmidt_with_error_analysis(vectors, normalize=True, verbose=True):
    """
    Gram-Schmidt process with comprehensive error analysis
    """
    if verbose:
        print("=== Gram-Schmidt Process with Error Analysis ===")
        print("Original vectors:")
        for i, v in enumerate(vectors):
            print(f"v{i+1} = {v}")
    
    n_vectors = len(vectors)
    orthogonal_basis = []
    error_metrics = []
    
    for i, v in enumerate(vectors):
        if verbose:
            print(f"\nProcessing vector v{i+1} = {v}")
        
        # Start with original vector
        u = v.copy().astype(float)
        step_errors = []
        
        # Subtract projections onto previous orthogonal vectors
        for j in range(i):
            # Compute projection coefficient
            numerator = np.dot(v, orthogonal_basis[j])
            denominator = np.dot(orthogonal_basis[j], orthogonal_basis[j])
            
            if abs(denominator) > 1e-10:
                proj_coeff = numerator / denominator
                proj_vector = proj_coeff * orthogonal_basis[j]
                u = u - proj_vector
                
                # Compute error in this step
                step_error = np.linalg.norm(u_old - u - proj_vector)
                step_errors.append(step_error)
                
                if verbose:
                    print(f"  Subtract projection onto u{j+1}: {proj_coeff:.6f} * u{j+1}")
                    print(f"  Step error: {step_error:.2e}")
            else:
                if verbose:
                    print(f"  Warning: Denominator too small for projection onto u{j+1}")
        
        # Check if resulting vector is non-zero
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            if normalize:
                u = u / norm  # Normalize to unit length
            orthogonal_basis.append(u)
            if verbose:
                print(f"  Result: u{i+1} = {u_normalized}")
            
            error_metrics.append({
                'vector_index': i,
                'step_errors': step_errors,
                'final_norm': norm,
                'normalized': normalize
            })
        else:
            if verbose:
                print(f"  Warning: Vector {i+1} became zero after orthogonalization")
    
    # Comprehensive verification
    verification_results = verify_gram_schmidt_comprehensive(vectors, orthogonal_basis)
    
    if verbose:
        print(f"\nVerification results:")
        for key, value in verification_results.items():
            print(f"  {key}: {value}")
    
    return orthogonal_basis, error_metrics, verification_results

def verify_gram_schmidt_comprehensive(original_vectors, orthogonal_vectors):
    """
    Comprehensive verification of Gram-Schmidt process
    """
    results = {}
    
    # Check orthogonality
    max_dot_product = 0
    orthogonality_errors = []
    for i in range(len(orthogonal_vectors)):
        for j in range(i+1, len(orthogonal_vectors)):
            dot_product = abs(np.dot(orthogonal_vectors[i], orthogonal_vectors[j]))
            max_dot_product = max(max_dot_product, dot_product)
            orthogonality_errors.append(dot_product)
    
    results['orthogonal'] = max_dot_product < 1e-10
    
    # Check normalization
    norms = [np.linalg.norm(u) for u in orthogonal_vectors]
    results['normalized'] = all(abs(norm - 1.0) < 1e-10 for norm in norms)
    results['norm_errors'] = [abs(norm - 1.0) for norm in norms]
    
    # Check span preservation (simplified)
    if len(original_vectors) == len(orthogonal_vectors):
        results['span_preserved'] = True
    else:
        results['span_preserved'] = False
    
    # Check condition number of orthogonal basis
    if len(orthogonal_vectors) > 0:
        basis_matrix = np.column_stack(orthogonal_vectors)
        results['condition_number'] = np.linalg.cond(basis_matrix)
    else:
        results['condition_number'] = float('inf')
    
    return results

# Apply Gram-Schmidt with error analysis
orthogonal_basis, error_metrics, verification = gram_schmidt_with_error_analysis(vectors, verbose=True)

# Test with nearly dependent vectors
print(f"\n=== Testing with Nearly Dependent Vectors ===")
nearly_dependent_vectors = [
    np.array([1, 0, 0]),
    np.array([1, 1e-8, 0]),  # Nearly parallel to first vector
    np.array([0, 1, 0])
]

orth_basis_nearly, error_metrics_nearly, verification_nearly = gram_schmidt_with_error_analysis(
    nearly_dependent_vectors, verbose=True
)

# Compare different normalization strategies
print(f"\n=== Comparing Normalization Strategies ===")
orth_basis_normalized, _, verification_normalized = gram_schmidt_with_error_analysis(
    vectors, normalize=True, verbose=False
)
orth_basis_not_normalized, _, verification_not_normalized = gram_schmidt_with_error_analysis(
    vectors, normalize=False, verbose=False
)

print("Normalized basis:")
for i, v in enumerate(orth_basis_normalized):
    norm = np.linalg.norm(v)
    print(f"  u{i+1}: norm = {norm:.6f}")

print("Non-normalized basis:")
for i, v in enumerate(orth_basis_not_normalized):
    norm = np.linalg.norm(v)
    print(f"  u{i+1}: norm = {norm:.6f}")
```

### Solution 4: Feature Selection and Dimensionality Reduction

```python
def comprehensive_feature_analysis(feature_matrix, feature_names):
    """
    Comprehensive feature analysis with multiple selection methods
    """
    print("=== Comprehensive Feature Analysis ===")
    
    # Basic properties
    n_samples, n_features = feature_matrix.shape
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    
    # Rank analysis
    rank = np.linalg.matrix_rank(feature_matrix)
    condition_number = np.linalg.cond(feature_matrix)
    print(f"Matrix rank: {rank}")
    print(f"Condition number: {condition_number:.2e}")
    
    # Correlation analysis
    corr_matrix = np.corrcoef(feature_matrix.T)
    print(f"\nCorrelation analysis:")
    
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = corr_matrix[i, j]
            if abs(corr) > 0.9:
                high_corr_pairs.append((i, j, corr))
                print(f"  High correlation: {feature_names[i]} ↔ {feature_names[j]}: {corr:.3f}")
    
    # Feature selection with different methods
    methods = ['qr', 'svd', 'gaussian']
    selection_results = {}
    
    print(f"\nFeature selection results:")
    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")
        
        independent_features, indices, computed_rank = select_linearly_independent_features(
            feature_matrix, method=method
        )
        
        print(f"Selected features: {[feature_names[i] for i in indices]}")
        print(f"Number of features: {len(indices)}")
        print(f"Rank: {computed_rank}")
        
        # Verify selection
        verification = verify_feature_selection(feature_matrix, independent_features, indices)
        print(f"Reconstruction error: {verification['relative_error']:.2e}")
        print(f"Rank preserved: {verification['rank_preserved']}")
        
        selection_results[method] = {
            'independent_features': independent_features,
            'indices': indices,
            'rank': computed_rank,
            'verification': verification
        }
    
    # Impact on simple regression model
    print(f"\n=== Impact on Regression Model ===")
    
    # Generate synthetic target variable
    np.random.seed(42)
    true_coefficients = np.random.randn(n_features)
    target = feature_matrix @ true_coefficients + 0.1 * np.random.randn(n_samples)
    
    # Fit model with original features
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    model_original = LinearRegression()
    model_original.fit(feature_matrix, target)
    score_original = r2_score(target, model_original.predict(feature_matrix))
    
    print(f"Original features R² score: {score_original:.4f}")
    
    # Fit models with selected features
    for method, results in selection_results.items():
        selected_features = results['independent_features']
        model_selected = LinearRegression()
        model_selected.fit(selected_features, target)
        score_selected = r2_score(target, model_selected.predict(selected_features))
        
        print(f"{method.upper()} selected features R² score: {score_selected:.4f}")
    
    return selection_results

# Apply comprehensive analysis
selection_results = comprehensive_feature_analysis(all_features, feature_names)

# Additional analysis: feature importance
print(f"\n=== Feature Importance Analysis ===")

# Compute feature importance using correlation with target
target = all_features @ np.random.randn(all_features.shape[1]) + 0.1 * np.random.randn(all_features.shape[0])
feature_importance = np.abs(np.corrcoef(all_features.T, target)[:-1, -1])

print("Feature importance (correlation with target):")
for i, importance in enumerate(feature_importance):
    print(f"  {feature_names[i]}: {importance:.3f}")

# Compare with selected features
for method, results in selection_results.items():
    selected_indices = results['indices']
    avg_importance = np.mean([feature_importance[i] for i in selected_indices])
    print(f"{method.upper()} selected features average importance: {avg_importance:.3f}")
```

### Solution 5: PCA Implementation and Analysis

```python
def pca_from_scratch(data, n_components=None, center=True):
    """
    PCA implementation from scratch
    """
    print("=== PCA Implementation from Scratch ===")
    
    n_samples, n_features = data.shape
    
    # Step 1: Center the data
    if center:
        mean_vector = np.mean(data, axis=0)
        data_centered = data - mean_vector
        print(f"Data centered: mean = {mean_vector}")
    else:
        mean_vector = np.zeros(n_features)
        data_centered = data
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Step 3: Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Step 5: Determine number of components
    if n_components is None:
        n_components = len(eigenvalues)
    elif n_components > len(eigenvalues):
        n_components = len(eigenvalues)
    
    # Extract components
    principal_components = eigenvectors_sorted[:, :n_components]
    eigenvalues_selected = eigenvalues_sorted[:n_components]
    
    # Compute explained variance
    total_variance = np.sum(eigenvalues_selected)
    explained_variance_ratio = eigenvalues_selected / total_variance
    
    print(f"Eigenvalues: {eigenvalues_selected}")
    print(f"Explained variance ratios: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {np.sum(explained_variance_ratio):.3f}")
    
    return principal_components, eigenvalues_selected, explained_variance_ratio, mean_vector

def compare_pca_implementations(data, n_components=2):
    """
    Compare our PCA implementation with built-in functions
    """
    print("=== Comparing PCA Implementations ===")
    
    # Our implementation
    pc_ours, eigenvals_ours, var_ratio_ours, mean_ours = pca_from_scratch(data, n_components)
    
    # Built-in implementation
    from sklearn.decomposition import PCA
    pca_sklearn = PCA(n_components=n_components)
    pca_sklearn.fit(data)
    
    pc_sklearn = pca_sklearn.components_.T
    eigenvals_sklearn = pca_sklearn.explained_variance_
    var_ratio_sklearn = pca_sklearn.explained_variance_ratio_
    
    print(f"\nComparison:")
    print(f"Our eigenvalues: {eigenvals_ours}")
    print(f"Sklearn eigenvalues: {eigenvals_sklearn}")
    
    print(f"Our explained variance: {var_ratio_ours}")
    print(f"Sklearn explained variance: {var_ratio_sklearn}")
    
    # Compare principal components (up to sign)
    for i in range(n_components):
        pc_diff = np.linalg.norm(pc_ours[:, i] - pc_sklearn[:, i])
        pc_diff_alt = np.linalg.norm(pc_ours[:, i] + pc_sklearn[:, i])  # Check opposite sign
        min_diff = min(pc_diff, pc_diff_alt)
        print(f"PC{i+1} difference: {min_diff:.2e}")
    
    return pc_ours, eigenvals_ours, var_ratio_ours

# Apply PCA analysis
pc_components, eigenvalues, explained_variance = compare_pca_implementations(data, n_components=2)

# Project and reconstruct data
print(f"\n=== Data Projection and Reconstruction ===")

# Project data
projected_data = project_data_pca(data, pc_components, mean_vector=None)
print(f"Projected data shape: {projected_data.shape}")

# Reconstruct data
reconstructed_data = reconstruct_data_pca(projected_data, pc_components, mean_vector=None)

# Analyze reconstruction quality
quality = analyze_pca_quality(data, reconstructed_data)
print(f"Reconstruction quality:")
for key, value in quality.items():
    print(f"  {key}: {value:.6f}")

# Test with different numbers of components
print(f"\n=== Testing Different Numbers of Components ===")
component_numbers = [1, 2, 3]

for n_comp in component_numbers:
    pc_comp, eigenvals_comp, var_ratio_comp = compare_pca_implementations(data, n_components=n_comp)
    
    projected_comp = project_data_pca(data, pc_comp, mean_vector=None)
    reconstructed_comp = reconstruct_data_pca(projected_comp, pc_comp, mean_vector=None)
    
    quality_comp = analyze_pca_quality(data, reconstructed_comp)
    
    print(f"\nComponents: {n_comp}")
    print(f"  Explained variance: {np.sum(var_ratio_comp):.3f}")
    print(f"  Relative error: {quality_comp['relative_error']:.6f}")

# Verify orthogonality and normalization
print(f"\n=== Verifying PCA Properties ===")

# Check orthogonality
print("Orthogonality of principal components:")
for i in range(len(pc_components)):
    for j in range(i+1, len(pc_components)):
        dot_product = np.dot(pc_components[:, i], pc_components[:, j])
        print(f"  PC{i+1} · PC{j+1} = {dot_product:.2e}")

# Check normalization
print("Norm of principal components:")
for i, pc in enumerate(pc_components.T):
    norm = np.linalg.norm(pc)
    print(f"  ||PC{i+1}|| = {norm:.6f}")
```

### Solution 6: Neural Network Weight Analysis

```python
def comprehensive_neural_network_analysis(weight_matrices, layer_names):
    """
    Comprehensive analysis of neural network weights
    """
    print("=== Comprehensive Neural Network Weight Analysis ===")
    
    analysis_results = analyze_neural_network_weights(weight_matrices, layer_names)
    
    # Display analysis results
    for layer_name, layer_analysis in analysis_results.items():
        print(f"\n{layer_name}:")
        print(f"  Shape: {layer_analysis['shape']}")
        print(f"  Row rank: {layer_analysis['row_rank']}/{layer_analysis['shape'][0]} ({layer_analysis['row_independence_ratio']:.2f})")
        print(f"  Column rank: {layer_analysis['col_rank']}/{layer_analysis['shape'][1]} ({layer_analysis['col_independence_ratio']:.2f})")
        print(f"  Row condition number: {layer_analysis['row_condition']:.2e}")
        print(f"  Column condition number: {layer_analysis['col_condition']:.2e}")

    return analysis_results

def improve_network_weights(weight_matrices, layer_names, analysis_results):
    """
    Improve neural network weights by promoting independence
    """
    print(f"\n=== Improving Network Weights ===")
    
    improved_matrices = []
    improvements = []
    
    for i, (weights, layer_name) in enumerate(zip(weight_matrices, layer_names)):
        original_analysis = analysis_results[layer_name]
        
        print(f"\nImproving {layer_name}...")
        print(f"  Original row independence: {original_analysis['row_independence_ratio']:.3f}")
        print(f"  Original column independence: {original_analysis['col_independence_ratio']:.3f}")
        
        # Apply different improvement methods
        methods = ['orthogonal_init', 'rank_regularization']
        best_improvement = 0
        best_weights = weights
        
        for method in methods:
            for strength in [0.1, 0.2, 0.3]:
                improved_weights = promote_weight_independence(weights, method=method, strength=strength)
                
                # Analyze improved weights
                improved_analysis = analyze_neural_network_weights([improved_weights], [layer_name])
                improved_metrics = improved_analysis[layer_name]
                
                row_improvement = improved_metrics['row_independence_ratio'] - original_analysis['row_independence_ratio']
                col_improvement = improved_metrics['col_independence_ratio'] - original_analysis['col_independence_ratio']
                
                total_improvement = row_improvement + col_improvement
                
                if total_improvement > best_improvement:
                    best_improvement = total_improvement
                    best_weights = improved_weights
                    best_method = method
                    best_strength = strength
        
        if best_improvement > 0:
            print(f"  Best improvement: {best_improvement:.3f} (method: {best_method}, strength: {best_strength})")
            
            # Analyze best improved weights
            best_analysis = analyze_neural_network_weights([best_weights], [layer_name])
            best_metrics = best_analysis[layer_name]
            
            print(f"  Improved row independence: {best_metrics['row_independence_ratio']:.3f}")
            print(f"  Improved column independence: {best_metrics['col_independence_ratio']:.3f}")
        else:
            print(f"  No improvement found")
            best_weights = weights
        
        improved_matrices.append(best_weights)
        improvements.append(best_improvement)
    
    return improved_matrices, improvements

def test_network_capacity(original_matrices, improved_matrices, layer_names):
    """
    Test the impact of weight improvements on network capacity
    """
    print(f"\n=== Testing Network Capacity ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    input_size = original_matrices[0].shape[1]
    output_size = original_matrices[-1].shape[0]
    
    X = np.random.randn(n_samples, input_size)
    y = np.random.randn(n_samples, output_size)
    
    def forward_pass(input_data, weight_matrices):
        """Simple forward pass through the network"""
        current = input_data
        for weights in weight_matrices:
            current = current @ weights.T
            # Apply ReLU activation
            current = np.maximum(0, current)
        return current
    
    # Test original network
    original_output = forward_pass(X, original_matrices)
    original_capacity = np.linalg.matrix_rank(original_output)
    
    # Test improved network
    improved_output = forward_pass(X, improved_matrices)
    improved_capacity = np.linalg.matrix_rank(improved_output)
    
    print(f"Original network output rank: {original_capacity}")
    print(f"Improved network output rank: {improved_capacity}")
    print(f"Capacity improvement: {improved_capacity - original_capacity}")
    
    # Test with different input sizes
    print(f"\nCapacity test with different input sizes:")
    for n_samples_test in [50, 100, 200]:
        X_test = np.random.randn(n_samples_test, input_size)
        
        original_output_test = forward_pass(X_test, original_matrices)
        improved_output_test = forward_pass(X_test, improved_matrices)
        
        original_rank = np.linalg.matrix_rank(original_output_test)
        improved_rank = np.linalg.matrix_rank(improved_output_test)
        
        print(f"  {n_samples_test} samples: Original rank {original_rank}, Improved rank {improved_rank}")
    
    return original_capacity, improved_capacity

# Apply comprehensive analysis
analysis_results = comprehensive_neural_network_analysis(weight_matrices, layer_names)

# Improve weights
improved_matrices, improvements = improve_network_weights(weight_matrices, layer_names, analysis_results)

# Test network capacity
original_capacity, improved_capacity = test_network_capacity(weight_matrices, improved_matrices, layer_names)

# Summary
print(f"\n=== Summary ===")
print(f"Total improvements across all layers: {sum(improvements):.3f}")
print(f"Network capacity improvement: {improved_capacity - original_capacity}")

# Visualize improvements
print(f"\nImprovements by layer:")
for i, (layer_name, improvement) in enumerate(zip(layer_names, improvements)):
    print(f"  {layer_name}: {improvement:.3f}")
```

## Summary

In this comprehensive chapter on linear independence and basis, we have covered:

### Key Concepts
- **Linear Independence**: Understanding when vectors are independent and how to test for it
- **Basis and Dimension**: Finding bases for vector spaces and understanding coordinate systems
- **Change of Basis**: Transforming between different coordinate representations
- **Gram-Schmidt Process**: Creating orthogonal bases from independent vectors

### Mathematical Foundations
- **Determinant and Rank**: Using matrix properties to test independence
- **Eigenvalues and Eigenvectors**: Understanding matrix structure
- **Orthogonality**: Creating perpendicular vector sets
- **Projection**: Decomposing vectors into orthogonal components

### Machine Learning Applications
- **Feature Selection**: Identifying and removing redundant features
- **Dimensionality Reduction**: Using PCA to find optimal representations
- **Neural Networks**: Ensuring weight matrices have good conditioning
- **Data Analysis**: Understanding data structure through linear algebra

### Practical Skills
- **Python Implementation**: Writing robust algorithms for independence testing
- **Numerical Stability**: Handling floating-point arithmetic carefully
- **Verification**: Testing implementations with multiple methods
- **Error Analysis**: Understanding and quantifying approximation errors

### Advanced Topics
- **QR Decomposition**: Using matrix factorizations for independence testing
- **SVD Analysis**: Understanding data structure through singular values
- **Condition Numbers**: Measuring numerical stability
- **Regularization**: Promoting independence in machine learning models

The concepts and techniques learned in this chapter are fundamental to understanding linear algebra in the context of machine learning and data science. They provide the mathematical foundation for many advanced algorithms and help us design better models and understand data structure.

## Next Steps

In the next chapter, we'll explore matrix decompositions, which are powerful tools for understanding matrix structure, solving systems of equations, and implementing efficient algorithms. Matrix decompositions build upon the concepts of linear independence and basis that we've developed here. 