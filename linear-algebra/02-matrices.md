# Matrices and Matrix Operations

[![Chapter](https://img.shields.io/badge/Chapter-2-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Matrices-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-brightgreen.svg)]()

## Introduction

Matrices are rectangular arrays of numbers that represent linear transformations and systems of linear equations. They are fundamental in machine learning for representing data, transformations, and computations. Understanding matrices is crucial for grasping linear algebra concepts and their applications in AI/ML.

### Why Matrices Matter in AI/ML

1. **Data Representation**: Datasets are often represented as matrices where rows are samples and columns are features
2. **Linear Transformations**: Matrices encode how vectors are transformed in space
3. **Neural Networks**: Weight matrices connect layers and transform activations
4. **Optimization**: Hessian matrices, covariance matrices, and other second-order information
5. **Dimensionality Reduction**: PCA, SVD, and other techniques work with matrices
6. **Systems of Equations**: Linear regression, least squares, and other ML problems

## What is a Matrix?

A matrix is a 2D array of numbers arranged in rows and columns. An m×n matrix has m rows and n columns, representing a linear transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$.

### Mathematical Notation

A matrix $A$ of size $m \times n$ is written as:
$$A = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

Where:
- $a_{ij}$ is the element in the $i$-th row and $j$-th column
- $i$ ranges from 1 to $m$ (rows)
- $j$ ranges from 1 to $n$ (columns)

### Matrix as Linear Transformation

A matrix $A$ represents a linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ such that:
$$T(\vec{x}) = A\vec{x}$$

This means:
- Input: Vector $\vec{x} \in \mathbb{R}^n$
- Output: Vector $\vec{y} = A\vec{x} \in \mathbb{R}^m$
- The transformation is linear: $T(c\vec{x} + \vec{y}) = cT(\vec{x}) + T(\vec{y})$

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

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
print("Number of rows:", A.shape[0])
print("Number of columns:", A.shape[1])
print("Total elements:", A.size)
print("Data type:", A.dtype)

print("\nMatrix B:")
print(B)

# Creating matrices with different methods
zeros_matrix = np.zeros((3, 4))  # 3×4 zero matrix
ones_matrix = np.ones((2, 3))    # 2×3 matrix of ones
random_matrix = np.random.randn(3, 3)  # 3×3 random matrix
identity_matrix = np.eye(4)      # 4×4 identity matrix

print("\nZero matrix (3×4):")
print(zeros_matrix)
print("\nOnes matrix (2×3):")
print(ones_matrix)
print("\nRandom matrix (3×3):")
print(random_matrix)
print("\nIdentity matrix (4×4):")
print(identity_matrix)

# Matrix indexing and slicing
print("\nFirst row of A:", A[0, :])
print("Second column of A:", A[:, 1])
print("Element at position (1, 2):", A[1, 2])
print("Submatrix (first 2 rows, first 2 columns):")
print(A[:2, :2])
```

## Matrix Operations

### Addition and Subtraction

**Mathematical Definition:**
Matrices are added/subtracted element-wise (must have same dimensions):
$$(A + B)_{ij} = A_{ij} + B_{ij}$$
$$(A - B)_{ij} = A_{ij} - B_{ij}$$

**Properties:**
- **Commutative**: $A + B = B + A$
- **Associative**: $(A + B) + C = A + (B + C)$
- **Identity**: $A + 0 = A$ (where 0 is the zero matrix)
- **Inverse**: $A + (-A) = 0$

**Geometric Interpretation:**
- Matrix addition corresponds to adding the corresponding linear transformations
- $(A + B)\vec{x} = A\vec{x} + B\vec{x}$

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

# Verification of properties
print("\nCommutative property (A + B == B + A):", np.array_equal(A + B, B + A))
print("Associative property check:", np.array_equal((A + B) + C, A + (B + C)))

# Element-wise operations
print("\nElement-wise addition (same as matrix addition):")
print(np.add(A, B))

# Adding matrices of different shapes (broadcasting)
row_vector = np.array([[1, 2, 3]])
column_vector = np.array([[1], [2], [3]])

print("\nAdding row vector to each row of A:")
print(A + row_vector)
print("\nAdding column vector to each column of A:")
print(A + column_vector)
```

### Scalar Multiplication

**Mathematical Definition:**
Multiplying a matrix by a scalar multiplies each element:
$$(cA)_{ij} = c \times A_{ij}$$

**Properties:**
- **Distributive over matrix addition**: $c(A + B) = cA + cB$
- **Distributive over scalar addition**: $(c + d)A = cA + dA$
- **Associative**: $(cd)A = c(dA)$
- **Identity**: $1A = A$

**Geometric Interpretation:**
- Scalar multiplication scales the linear transformation
- $(cA)\vec{x} = c(A\vec{x})$

```python
# Scalar multiplication
scaled_A = 2 * A
print("2 * A:")
print(scaled_A)

# Division
divided_A = A / 2
print("\nA / 2:")
print(divided_A)

# Multiple scalar operations
result_chain = 2 * 3 * A
print("\n2 * 3 * A:")
print(result_chain)

# Verification of properties
print("\nDistributive over matrix addition:", 
      np.array_equal(2 * (A + B), 2 * A + 2 * B))
print("Associative property:", 
      np.array_equal((2 * 3) * A, 2 * (3 * A)))

# Scaling effects on transformation
scales = [0.5, 1, 2, -1]
for scale in scales:
    scaled = scale * A
    print(f"\n{scale} * A:")
    print(scaled)
```

### Matrix Multiplication

**Mathematical Definition:**
Matrix multiplication is defined as:
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Where $A$ is $m \times n$ and $B$ is $n \times p$, resulting in an $m \times p$ matrix.

**Key Points:**
- The number of columns in $A$ must equal the number of rows in $B$
- Matrix multiplication is **not commutative**: $AB \neq BA$ in general
- Matrix multiplication is **associative**: $(AB)C = A(BC)$
- Matrix multiplication is **distributive**: $A(B + C) = AB + AC$

**Geometric Interpretation:**
- Matrix multiplication represents composition of linear transformations
- $(AB)\vec{x} = A(B\vec{x})$: Apply transformation $B$ first, then $A$

**Step-by-Step Process:**
1. Take the $i$-th row of matrix $A$
2. Take the $j$-th column of matrix $B$
3. Compute the dot product of these vectors
4. Place the result in position $(i, j)$ of the product matrix

```python
# Matrix multiplication
AB = np.dot(A, B)
print("A × B:")
print(AB)

# Alternative syntax
AB_alt = A @ B
print("\nA @ B (same result):")
print(AB_alt)

# Manual calculation for first element
manual_first = sum(A[0, k] * B[k, 0] for k in range(A.shape[1]))
print(f"\nManual calculation of (AB)₁₁: {manual_first}")
print(f"NumPy result (AB)₁₁: {AB[0, 0]}")

# Verification of properties
print("\nAssociative property check:")
C = np.random.randn(3, 3)
print("(AB)C == A(BC):", np.allclose((A @ B) @ C, A @ (B @ C)))

print("\nDistributive property check:")
print("A(B + C) == AB + AC:", np.allclose(A @ (B + C), A @ B + A @ C))

# Non-commutativity demonstration
BA = B @ A
print("\nB × A:")
print(BA)
print("A × B == B × A?", np.array_equal(AB, BA))

# Matrix-vector multiplication
vector = np.array([1, 2, 3])
result = A @ vector
print(f"\nA × vector {vector}:")
print(result)

# Element-wise multiplication (Hadamard product)
element_wise = A * B
print("\nElement-wise multiplication (A * B):")
print(element_wise)
print("This is NOT matrix multiplication!")

# Matrix powers
A_squared = A @ A
A_cubed = A @ A @ A
print("\nA²:")
print(A_squared)
print("\nA³:")
print(A_cubed)
```

## Special Matrices

### Identity Matrix

**Mathematical Definition:**
The identity matrix $I_n$ is an $n \times n$ matrix with 1s on the diagonal and 0s elsewhere:
$$I_n = \begin{bmatrix} 
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}$$

**Properties:**
- $AI = A$ and $IA = A$ for any matrix $A$ of appropriate size
- $I$ represents the identity transformation: $I\vec{x} = \vec{x}$
- $I$ is the multiplicative identity for matrices

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

# Verify: IA = A
result2 = I @ A
print("\nI × A = A:")
print(result2)
print("\nIs I × A equal to A?", np.array_equal(result2, A))

# Identity matrix as linear transformation
test_vector = np.array([1, 2, 3])
transformed = I @ test_vector
print(f"\nI × {test_vector} = {transformed}")
print("Identity transformation preserves the vector:", np.array_equal(test_vector, transformed))

# Creating identity matrices of different sizes
I_2 = np.eye(2)
I_4 = np.eye(4)
print("\n2×2 Identity matrix:")
print(I_2)
print("\n4×4 Identity matrix:")
print(I_4)
```

### Zero Matrix

**Mathematical Definition:**
A zero matrix $0$ has all elements equal to zero.

**Properties:**
- $A + 0 = A$ for any matrix $A$
- $0A = 0$ and $A0 = 0$ for any matrix $A$ of appropriate size
- $0$ represents the zero transformation: $0\vec{x} = \vec{0}$

```python
# Zero matrix
Z = np.zeros((3, 3))
print("3×3 Zero matrix:")
print(Z)

# Properties verification
print("\nA + 0 = A:", np.array_equal(A + Z, A))
print("0 × A = 0:", np.array_equal(Z @ A, Z))
print("A × 0 = 0:", np.array_equal(A @ Z, Z))

# Zero transformation
test_vector = np.array([1, 2, 3])
zero_result = Z @ test_vector
print(f"\n0 × {test_vector} = {zero_result}")
print("Zero transformation produces zero vector:", np.array_equal(zero_result, np.zeros(3)))
```

### Diagonal Matrix

**Mathematical Definition:**
A diagonal matrix has non-zero elements only on the main diagonal:
$$D = \begin{bmatrix} 
d_1 & 0 & \cdots & 0 \\
0 & d_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & d_n
\end{bmatrix}$$

**Properties:**
- Diagonal matrices commute: $D_1D_2 = D_2D_1$
- Powers of diagonal matrices are easy to compute: $D^k = \text{diag}(d_1^k, d_2^k, \ldots, d_n^k)$
- Diagonal matrices represent scaling transformations

```python
# Diagonal matrix
D = np.diag([1, 2, 3])
print("Diagonal matrix:")
print(D)

# Extract diagonal from matrix
diagonal = np.diag(A)
print("\nDiagonal of A:")
print(diagonal)

# Create diagonal matrix from vector
diag_from_vector = np.diag([5, 10, 15])
print("\nDiagonal matrix from vector [5, 10, 15]:")
print(diag_from_vector)

# Properties of diagonal matrices
D1 = np.diag([1, 2, 3])
D2 = np.diag([4, 5, 6])
print("\nD1:")
print(D1)
print("\nD2:")
print(D2)
print("\nD1 × D2:")
print(D1 @ D2)
print("\nD2 × D1:")
print(D2 @ D1)
print("Diagonal matrices commute:", np.array_equal(D1 @ D2, D2 @ D1))

# Powers of diagonal matrices
D_squared = D @ D
D_cubed = D @ D @ D
print("\nD²:")
print(D_squared)
print("\nD³:")
print(D_cubed)
print("Manual calculation D² = diag([1², 2², 3²]):", np.array_equal(D_squared, np.diag([1, 4, 9])))
```

## Matrix Properties

### Transpose

**Mathematical Definition:**
The transpose of a matrix flips rows and columns:
$$(A^T)_{ij} = A_{ji}$$

**Properties:**
- $(A^T)^T = A$ (transpose of transpose is original)
- $(A + B)^T = A^T + B^T$
- $(cA)^T = cA^T$
- $(AB)^T = B^T A^T$ (important!)

**Geometric Interpretation:**
- Transpose relates to the adjoint transformation
- For real matrices, transpose corresponds to reflecting across the main diagonal

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
print("(cA)ᵀ = cAᵀ:", np.array_equal((2*A).T, 2*A.T))
print("(AB)ᵀ = BᵀAᵀ:", np.array_equal((A @ B).T, B.T @ A.T))

# Transpose of different matrix types
row_vector = np.array([[1, 2, 3]])
column_vector = row_vector.T
print("\nRow vector:", row_vector)
print("Column vector (transpose):", column_vector)
print("Shape of row vector:", row_vector.shape)
print("Shape of column vector:", column_vector.shape)

# Transpose and matrix-vector multiplication
vector = np.array([1, 2, 3])
result1 = A @ vector
result2 = (vector @ A.T).T
print(f"\nA × {vector}:")
print(result1)
print(f"({vector} × Aᵀ)ᵀ:")
print(result2)
print("Results are equal:", np.array_equal(result1, result2))
```

### Trace

**Mathematical Definition:**
The trace is the sum of diagonal elements:
$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

**Properties:**
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(cA) = c \cdot \text{tr}(A)$
- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
- $\text{tr}(A^T) = \text{tr}(A)$

**Applications:**
- Trace is used in optimization (e.g., matrix calculus)
- Trace appears in many ML algorithms (e.g., covariance matrices)
- Trace is invariant under similarity transformations

```python
# Matrix trace
trace_A = np.trace(A)
print("Trace of A:", trace_A)

# Manual calculation
manual_trace = sum(A[i, i] for i in range(min(A.shape)))
print("Manual trace calculation:", manual_trace)

# Verification: tr(A + B) = tr(A) + tr(B)
trace_sum = np.trace(A + B)
trace_A_plus_trace_B = np.trace(A) + np.trace(B)
print("tr(A + B) = tr(A) + tr(B):", trace_sum == trace_A_plus_trace_B)

# Verification: tr(cA) = c·tr(A)
c = 3
trace_scaled = np.trace(c * A)
trace_c_times_trace_A = c * np.trace(A)
print(f"tr({c}A) = {c}·tr(A):", trace_scaled == trace_c_times_trace_A)

# Cyclic property: tr(AB) = tr(BA)
trace_AB = np.trace(A @ B)
trace_BA = np.trace(B @ A)
print("tr(AB) = tr(BA):", trace_AB == trace_BA)

# Trace of transpose
trace_A_transpose = np.trace(A.T)
print("tr(Aᵀ) = tr(A):", trace_A == trace_A_transpose)

# Trace in optimization context
# Example: Trace of covariance matrix
data = np.random.randn(100, 3)  # 100 samples, 3 features
cov_matrix = np.cov(data.T)
trace_cov = np.trace(cov_matrix)
print(f"\nTrace of covariance matrix: {trace_cov:.4f}")
print("This represents the total variance in the data")
```

## Matrix Types

### Symmetric Matrix

**Mathematical Definition:**
A symmetric matrix satisfies $A = A^T$.

**Properties:**
- Eigenvalues are real
- Eigenvectors can be chosen to be orthogonal
- Symmetric matrices are diagonalizable
- Common in statistics (covariance matrices, correlation matrices)

```python
# Create symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

print("Symmetric matrix S:")
print(S)
print("\nS = Sᵀ:", np.array_equal(S, S.T))

# Verify symmetry property
print("Is S symmetric?", np.array_equal(S, S.T))

# Symmetric matrix from any matrix
any_matrix = np.random.randn(3, 3)
symmetric_from_any = (any_matrix + any_matrix.T) / 2
print("\nOriginal matrix:")
print(any_matrix)
print("\nSymmetric matrix created from it:")
print(symmetric_from_any)
print("Is it symmetric?", np.array_equal(symmetric_from_any, symmetric_from_any.T))

# Covariance matrix example (always symmetric)
data = np.random.randn(50, 3)
cov_matrix = np.cov(data.T)
print("\nCovariance matrix (symmetric):")
print(cov_matrix)
print("Is covariance matrix symmetric?", np.array_equal(cov_matrix, cov_matrix.T))
```

### Skew-Symmetric Matrix

**Mathematical Definition:**
A skew-symmetric matrix satisfies $A = -A^T$.

**Properties:**
- Diagonal elements are zero
- Eigenvalues are purely imaginary or zero
- Used in cross product representations and rotations

```python
# Create skew-symmetric matrix
K = np.array([[0, 2, -3],
              [-2, 0, 4],
              [3, -4, 0]])

print("Skew-symmetric matrix K:")
print(K)
print("\nK = -Kᵀ:", np.array_equal(K, -K.T))

# Verify skew-symmetry
print("Is K skew-symmetric?", np.array_equal(K, -K.T))
print("Diagonal elements are zero:", np.allclose(np.diag(K), 0))

# Skew-symmetric matrix from any matrix
any_matrix = np.random.randn(3, 3)
skew_from_any = (any_matrix - any_matrix.T) / 2
print("\nOriginal matrix:")
print(any_matrix)
print("\nSkew-symmetric matrix created from it:")
print(skew_from_any)
print("Is it skew-symmetric?", np.array_equal(skew_from_any, -skew_from_any.T))

# Cross product matrix (skew-symmetric)
def cross_product_matrix(v):
    """Create skew-symmetric matrix for cross product v × x"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

vector = np.array([1, 2, 3])
cross_matrix = cross_product_matrix(vector)
print(f"\nCross product matrix for vector {vector}:")
print(cross_matrix)
print("Is it skew-symmetric?", np.array_equal(cross_matrix, -cross_matrix.T))

# Verify cross product property
test_vector = np.array([4, 5, 6])
cross_result1 = np.cross(vector, test_vector)
cross_result2 = cross_matrix @ test_vector
print(f"\nCross product {vector} × {test_vector}:")
print("Using np.cross:", cross_result1)
print("Using matrix multiplication:", cross_result2)
print("Results are equal:", np.array_equal(cross_result1, cross_result2))
```

### Orthogonal Matrix

**Mathematical Definition:**
An orthogonal matrix satisfies $Q^T Q = Q Q^T = I$.

**Properties:**
- Columns are orthonormal (orthogonal and unit length)
- Rows are orthonormal
- $Q^T = Q^{-1}$
- Preserves lengths and angles: $|Q\vec{x}| = |\vec{x}|$
- Used in QR decomposition, rotations, and reflections

```python
# Create orthogonal matrix (rotation matrix)
def rotation_matrix_2d(angle):
    """Create 2D rotation matrix"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

Q = rotation_matrix_2d(np.pi/4)  # 45-degree rotation
print("Orthogonal matrix (rotation):")
print(Q)

# Verify orthogonality
print("\nQᵀQ = I:")
print(Q.T @ Q)
print("Is Q orthogonal?", np.allclose(Q.T @ Q, np.eye(2)))

# Verify Qᵀ = Q⁻¹
Q_inverse = np.linalg.inv(Q)
print("\nQ⁻¹:")
print(Q_inverse)
print("Qᵀ:")
print(Q.T)
print("Qᵀ = Q⁻¹:", np.allclose(Q.T, Q_inverse))

# Preserve lengths
test_vector = np.array([3, 4])
original_length = np.linalg.norm(test_vector)
transformed_vector = Q @ test_vector
transformed_length = np.linalg.norm(transformed_vector)
print(f"\nOriginal vector: {test_vector}, length: {original_length:.4f}")
print(f"Transformed vector: {transformed_vector}, length: {transformed_length:.4f}")
print("Length preserved:", np.isclose(original_length, transformed_length))

# Create random orthogonal matrix using QR decomposition
random_matrix = np.random.randn(3, 3)
Q_random, R = np.linalg.qr(random_matrix)
print("\nRandom orthogonal matrix (from QR decomposition):")
print(Q_random)
print("Is it orthogonal?", np.allclose(Q_random.T @ Q_random, np.eye(3)))
```

## Matrix Visualization

```python
# Heatmap visualization of matrices
def plot_matrix_heatmap(matrix, title, ax):
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_title(title)
    
    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="white")
    
    plt.colorbar(im, ax=ax)

# Create subplots for different matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_matrix_heatmap(A, 'Matrix A', axes[0, 0])
plot_matrix_heatmap(A.T, 'Matrix Aᵀ (Transpose)', axes[0, 1])
plot_matrix_heatmap(A @ B, 'Matrix A × B', axes[1, 0])
plot_matrix_heatmap(A * B, 'Element-wise A * B', axes[1, 1])

plt.tight_layout()
plt.show()

# 3D visualization of matrix transformation
def plot_matrix_transformation():
    fig = plt.figure(figsize=(15, 5))
    
    # Original unit vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transformed vectors
    Ae1 = A[:2, :2] @ e1
    Ae2 = A[:2, :2] @ e2
    
    # Plot 1: Original unit vectors
    ax1 = fig.add_subplot(131)
    ax1.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, color='red', label='e₁')
    ax1.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='e₂')
    ax1.set_xlim(-1, 2)
    ax1.set_ylim(-1, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Unit Vectors')
    ax1.legend()
    
    # Plot 2: Transformed vectors
    ax2 = fig.add_subplot(132)
    ax2.quiver(0, 0, Ae1[0], Ae1[1], angles='xy', scale_units='xy', scale=1, color='red', label='Ae₁')
    ax2.quiver(0, 0, Ae2[0], Ae2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Ae₂')
    ax2.set_xlim(-1, 6)
    ax2.set_ylim(-1, 6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Transformed Vectors')
    ax2.legend()
    
    # Plot 3: Unit square transformation
    ax3 = fig.add_subplot(133)
    # Original unit square
    square_original = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    ax3.plot(square_original[0], square_original[1], 'b-', label='Original')
    
    # Transformed unit square
    square_transformed = A[:2, :2] @ square_original
    ax3.plot(square_transformed[0], square_transformed[1], 'r-', label='Transformed')
    
    ax3.set_xlim(-1, 6)
    ax3.set_ylim(-1, 6)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Unit Square Transformation')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

plot_matrix_transformation()
```

## Applications in AI/ML

### 1. Data Representation
```python
# Dataset as matrix
# Rows: samples, Columns: features
dataset = np.random.randn(100, 5)  # 100 samples, 5 features
print("Dataset shape:", dataset.shape)
print("First 5 samples:")
print(dataset[:5])

# Feature matrix for machine learning
X = dataset[:, :-1]  # Features (first 4 columns)
y = dataset[:, -1]   # Target (last column)
print("\nFeature matrix X shape:", X.shape)
print("Target vector y shape:", y.shape)
```

### 2. Linear Transformations in Neural Networks
```python
# Simple neural network layer
def neural_network_layer(input_data, weights, bias):
    """Forward pass through a neural network layer"""
    return input_data @ weights + bias

# Example: Single layer with 3 inputs, 2 outputs
input_size = 3
output_size = 2
batch_size = 4

# Initialize weights and bias
W = np.random.randn(input_size, output_size) * 0.1
b = np.zeros(output_size)

# Input data
X = np.random.randn(batch_size, input_size)

# Forward pass
output = neural_network_layer(X, W, b)
print("Input shape:", X.shape)
print("Weight matrix shape:", W.shape)
print("Bias vector shape:", b.shape)
print("Output shape:", output.shape)
print("\nOutput:")
print(output)
```

### 3. Covariance Matrix
```python
# Calculate covariance matrix
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
cov_matrix = np.cov(data.T)

print("Covariance matrix:")
print(cov_matrix)
print("\nVariance of first feature:", cov_matrix[0, 0])
print("Variance of second feature:", cov_matrix[1, 1])
print("Covariance between features:", cov_matrix[0, 1])

# Eigendecomposition of covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

## Exercises

### Exercise 1: Matrix Operations
```python
# Given matrices A and B, compute various operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Compute: 2A + 3B, A², A×B, B×A
result1 = 2*A + 3*B
result2 = A @ A
result3 = A @ B
result4 = B @ A

print("2A + 3B:")
print(result1)
print("\nA²:")
print(result2)
print("\nA×B:")
print(result3)
print("\nB×A:")
print(result4)
print("\nA×B == B×A?", np.array_equal(result3, result4))
```

### Exercise 2: Matrix Properties
```python
# Verify matrix properties
A = np.random.randn(3, 3)
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)

# Associative property: (AB)C = A(BC)
left_side = (A @ B) @ C
right_side = A @ (B @ C)
print("(AB)C == A(BC):", np.allclose(left_side, right_side))

# Distributive property: A(B + C) = AB + AC
left_side = A @ (B + C)
right_side = A @ B + A @ C
print("A(B + C) == AB + AC:", np.allclose(left_side, right_side))

# Transpose property: (AB)ᵀ = BᵀAᵀ
left_side = (A @ B).T
right_side = B.T @ A.T
print("(AB)ᵀ == BᵀAᵀ:", np.allclose(left_side, right_side))
```

### Exercise 3: Special Matrices
```python
# Create and verify special matrices
n = 4

# Identity matrix
I = np.eye(n)
print("Identity matrix:")
print(I)
print("I² = I:", np.array_equal(I @ I, I))

# Diagonal matrix
D = np.diag([1, 2, 3, 4])
print("\nDiagonal matrix:")
print(D)
print("D²:")
print(D @ D)

# Symmetric matrix
S = np.random.randn(n, n)
S = (S + S.T) / 2  # Make symmetric
print("\nSymmetric matrix:")
print(S)
print("S = Sᵀ:", np.array_equal(S, S.T))
```

## Summary

In this chapter, we've covered:

1. **Matrix Fundamentals**: Definition, notation, and representation as linear transformations
2. **Basic Operations**: Addition, scalar multiplication, and matrix multiplication with detailed properties
3. **Special Matrices**: Identity, zero, diagonal, symmetric, skew-symmetric, and orthogonal matrices
4. **Matrix Properties**: Transpose, trace, and their mathematical properties
5. **Geometric Interpretation**: How matrices represent linear transformations
6. **Visualization**: Heatmaps and transformation plots
7. **AI/ML Applications**: Data representation, neural networks, and covariance matrices

### Key Takeaways:
- Matrices represent linear transformations between vector spaces
- Matrix multiplication is not commutative but is associative and distributive
- Special matrices have important properties and applications
- Understanding matrix operations is crucial for linear algebra and machine learning
- Matrices are fundamental for representing data and transformations in AI/ML

### Next Steps:
In the next chapter, we'll explore linear transformations in detail, understanding how matrices transform vectors and spaces. 