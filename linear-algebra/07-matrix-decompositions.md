# Matrix Decompositions

## Introduction

Matrix decompositions are fundamental tools in linear algebra that break down complex matrices into simpler, more manageable components. These decompositions reveal the underlying structure of matrices and enable efficient algorithms for solving systems of equations, understanding data patterns, and implementing machine learning algorithms.

**Mathematical Foundation:**
Matrix decompositions express a matrix $`A`$ as a product of simpler matrices:
```math
A = B_1 \times B_2 \times \cdots \times B_k
```

where each $`B_i`$ has a special structure (triangular, orthogonal, diagonal, etc.) that makes certain operations computationally efficient.

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

LU decomposition factors a square matrix $`A`$ into $`A = LU`$, where $`L`$ is lower triangular and $`U`$ is upper triangular. This decomposition is fundamental for solving systems of linear equations efficiently.

### Mathematical Foundation

**Definition:**
For a square matrix $`A \in \mathbb{R}^{n \times n}`$, the LU decomposition is:
```math
A = LU
```

where:
- $`L \in \mathbb{R}^{n \times n}`$ is lower triangular ($`L_{ij} = 0`$ for $`i < j`$)
- $`U \in \mathbb{R}^{n \times n}`$ is upper triangular ($`U_{ij} = 0`$ for $`i > j`$)

**Existence and Uniqueness:**
- **Existence**: LU decomposition exists if and only if all leading principal minors of $`A`$ are non-zero
- **Uniqueness**: If $`A`$ is invertible, the LU decomposition is unique when $`L`$ has ones on the diagonal
- **Generalization**: For any matrix, we can find $`A = PLU`$ where $`P`$ is a permutation matrix

**Geometric Interpretation:**
LU decomposition represents $`A`$ as a sequence of elementary row operations:
1. **L matrix**: Represents the row operations needed to eliminate entries below the diagonal
2. **U matrix**: The resulting upper triangular form after elimination
3. **P matrix**: Represents row exchanges needed for numerical stability

**Key Properties:**
1. **Determinant**: $`\det(A) = \det(L) \times \det(U) = \prod_i L_{ii} \times \prod_i U_{ii}`$
2. **Inverse**: $`A^{-1} = U^{-1}L^{-1}`$ (if $`A`$ is invertible)
3. **Linear Systems**: $`Ax = b`$ becomes $`LUx = b`$, solved by forward/backward substitution

### Algorithm: LU Decomposition with Partial Pivoting

**Mathematical Foundation:**
The LU decomposition with partial pivoting algorithm:

1. **Initialize**: $`L = I`$, $`U = A`$, $`P = I`$
2. **For each column** $`k = 1, 2, \ldots, n-1`$:
   - Find pivot: $`p = \arg\max_{i \geq k} |U_{ik}|`$
   - Exchange rows: $`U_{k,:} \leftrightarrow U_{p,:}`$, $`L_{k,1:k-1} \leftrightarrow L_{p,1:k-1}`$, $`P_{k,:} \leftrightarrow P_{p,:}`$
   - Eliminate: For $`i = k+1, \ldots, n`$:
     - $`L_{ik} = U_{ik}/U_{kk}`$
     - $`U_{i,k:n} = U_{i,k:n} - L_{ik}U_{k,k:n}`$

**Computational Complexity:**
- **Time Complexity**: $`O(n^3)`$ operations
- **Space Complexity**: $`O(n^2)`$ storage (in-place possible)
- **Numerical Stability**: Partial pivoting ensures $`|L_{ij}| \leq 1`$

### Solving Systems with LU Decomposition

**Mathematical Foundation:**
The LU decomposition enables efficient solution of linear systems $`Ax = b`$ through forward and backward substitution:

1. **Decomposition**: $`A = LU`$
2. **Forward Substitution**: Solve $`Ly = b`$ for $`y`$
3. **Backward Substitution**: Solve $`Ux = y`$ for $`x`$

**Forward Substitution Algorithm:**
For $`i = 1, 2, \ldots, n`$:
```math
y_i = \frac{1}{L_{ii}}\left(b_i - \sum_{j=1}^{i-1} L_{ij} y_j\right)
```

**Backward Substitution Algorithm:**
For $`i = n, n-1, \ldots, 1`$:
```math
x_i = \frac{1}{U_{ii}}\left(y_i - \sum_{j=i+1}^n U_{ij} x_j\right)
```

**Computational Complexity:**
- **Decomposition**: $`O(n^3)`$ operations (one-time cost)
- **Forward/Backward Substitution**: $`O(n^2)`$ operations per right-hand side
- **Multiple Right-hand Sides**: Very efficient after initial decomposition

**Numerical Stability:**
- **Partial Pivoting**: Exchanging rows to avoid small pivots
- **Growth Factor**: Measure of numerical stability
- **Condition Number**: Relationship between input and output perturbations

## QR Decomposition

QR decomposition factors a matrix $`A`$ into $`A = QR`$, where $`Q`$ is orthogonal and $`R`$ is upper triangular. This decomposition is fundamental for least squares problems, eigenvalue computation, and numerical stability.

### Mathematical Foundation

**Definition:**
For a matrix $`A \in \mathbb{R}^{m \times n}`$ with $`m \geq n`$, the QR decomposition is:
```math
A = QR
```

where:
- $`Q \in \mathbb{R}^{m \times m}`$ is orthogonal ($`Q^T Q = I`$)
- $`R \in \mathbb{R}^{m \times n}`$ is upper triangular ($`R_{ij} = 0`$ for $`i > j`$)

**Existence and Uniqueness:**
- **Existence**: QR decomposition always exists for any matrix $`A`$
- **Uniqueness**: If $`A`$ has full column rank, the decomposition is unique when $`R`$ has positive diagonal entries
- **Reduced Form**: For $`m > n`$, we can write $`A = Q_1 R_1`$ where $`Q_1 \in \mathbb{R}^{m \times n}`$ has orthonormal columns

**Geometric Interpretation:**
QR decomposition represents $`A`$ as:
1. **Q matrix**: Orthonormal basis for the column space of $`A`$
2. **R matrix**: Coordinates of $`A`$'s columns in the $`Q`$ basis
3. **Gram-Schmidt Connection**: QR decomposition is essentially Gram-Schmidt orthogonalization applied to $`A`$'s columns

**Key Properties:**
1. **Orthogonality**: $`Q^T Q = I`$ ($`Q`$ preserves lengths and angles)
2. **Upper Triangular**: $`R`$ is upper triangular, enabling efficient back-substitution
3. **Rank Preservation**: $`\text{rank}(A) = \text{rank}(R) =`$ number of non-zero diagonal elements of $`R`$
4. **Least Squares**: QR decomposition provides numerically stable solution to least squares problems

### Algorithm: Householder QR Decomposition

**Mathematical Foundation:**
Householder reflections provide a numerically stable method for QR decomposition:

1. **For each column** $`k = 1, 2, \ldots, n`$:
   - Define $`v_k = a_k - \text{sign}(a_{kk})\|a_k\|_2 e_k`$
   - Householder matrix: $`H_k = I - 2\frac{v_k v_k^T}{v_k^T v_k}`$
   - Apply: $`A = H_k A`$, $`Q = Q H_k`$

**Properties:**
- **Numerical Stability**: Householder method is more stable than Gram-Schmidt
- **Orthogonality**: $`Q`$ is exactly orthogonal (up to machine precision)
- **Sparsity**: $`R`$ is upper triangular with zeros below diagonal

### Least Squares Applications

**Mathematical Foundation:**
QR decomposition provides a numerically stable method for solving least squares problems:

```math
\min \|Ax - b\|_2
```

The solution is obtained by:
1. **Decomposition**: $`A = QR`$
2. **Transformation**: $`\|Ax - b\|_2 = \|QRx - b\|_2 = \|Rx - Q^T b\|_2`$
3. **Solution**: Solve $`Rx = Q^T b`$ by back-substitution

**Key Advantages:**
1. **Numerical Stability**: QR decomposition is more stable than normal equations
2. **Rank Deficiency**: Handles rank-deficient matrices gracefully
3. **Multiple Right-hand Sides**: Efficient for solving multiple least squares problems
4. **Conditioning**: Preserves conditioning better than other methods

**Algorithm:**
1. Compute $`A = QR`$ using Householder method
2. Compute $`c = Q^T b`$
3. Solve $`Rx = c`$ by back-substitution
4. Solution: $`x = R^{-1} c`$

## Singular Value Decomposition (SVD)

SVD decomposes a matrix $`A`$ into $`A = U\Sigma V^T`$, where $`U`$ and $`V`$ are orthogonal and $`\Sigma`$ is diagonal. This is one of the most powerful matrix decompositions, revealing the fundamental structure of any matrix.

### Mathematical Definition

**Full SVD:**
For a matrix $`A \in \mathbb{R}^{m \times n}`$, the SVD is:
```math
A = U \Sigma V^T
```

where:
- $`U \in \mathbb{R}^{m \times m}`$ is orthogonal (left singular vectors)
- $`\Sigma \in \mathbb{R}^{m \times n}`$ is diagonal (singular values)
- $`V \in \mathbb{R}^{n \times n}`$ is orthogonal (right singular vectors)

**Reduced SVD:**
For $`m \geq n`$, we can write:
```math
A = U_1 \Sigma_1 V^T
```

where $`U_1 \in \mathbb{R}^{m \times n}`$ has orthonormal columns and $`\Sigma_1 \in \mathbb{R}^{n \times n}`$ is diagonal.

**Properties:**
1. **Singular Values**: $`\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0`$ where $`r = \text{rank}(A)`$
2. **Rank**: $`\text{rank}(A) =`$ number of non-zero singular values
3. **Condition Number**: $`\kappa(A) = \sigma_1 / \sigma_r`$
4. **Frobenius Norm**: $`\|A\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2}`$

### Geometric Interpretation

**SVD as Coordinate Transformation:**
1. **Right Singular Vectors** ($`V`$): Orthonormal basis for the input space
2. **Left Singular Vectors** ($`U`$): Orthonormal basis for the output space
3. **Singular Values** ($`\Sigma`$): Scaling factors along principal axes

**Data Analysis Perspective:**
- **Principal Components**: Right singular vectors are principal components
- **Variance**: Singular values squared represent variance explained
- **Dimensionality**: Number of significant singular values indicates intrinsic dimension

### Low-Rank Approximation

**Mathematical Foundation:**
The best rank-$`k`$ approximation of $`A`$ is:
```math
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T
```

where $`u_i`$ and $`v_i`$ are the $`i`$-th columns of $`U`$ and $`V`$ respectively.

**Optimality:**
- **Eckart-Young Theorem**: $`A_k`$ is the best rank-$`k`$ approximation in both Frobenius and spectral norms
- **Error**: $`\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}`$

**Applications:**
- **Data Compression**: Store only top-$`k`$ singular values and vectors
- **Noise Reduction**: Remove components with small singular values
- **Dimensionality Reduction**: Project data onto top singular vectors

## Cholesky Decomposition

Cholesky decomposition factors a positive definite matrix $`A`$ into $`A = LL^T`$, where $`L`$ is lower triangular. This is the most efficient decomposition for positive definite matrices.

### Mathematical Definition

**Definition:**
For a symmetric positive definite matrix $`A \in \mathbb{R}^{n \times n}`$:
```math
A = LL^T
```

where $`L \in \mathbb{R}^{n \times n}`$ is lower triangular with positive diagonal entries.

**Existence and Uniqueness:**
- **Existence**: Cholesky decomposition exists if and only if $`A`$ is symmetric positive definite
- **Uniqueness**: The decomposition is unique when $`L`$ has positive diagonal entries

**Algorithm:**
For $`j = 1, 2, \ldots, n`$:
```math
\begin{align}
L_{jj} &= \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2} \\
L_{ij} &= \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk}\right) \quad \text{for } i = j+1, \ldots, n
\end{align}
```

**Properties:**
1. **Stability**: Cholesky is numerically stable for positive definite matrices
2. **Efficiency**: Requires $`O(n^3/3)`$ operations (half of LU)
3. **Storage**: Can overwrite $`A`$ with $`L`$ (saves memory)
4. **Determinant**: $`\det(A) = \prod_{i=1}^n L_{ii}^2`$

### Solving Systems with Cholesky

**Mathematical Foundation:**
For solving $`Ax = b`$ where $`A`$ is positive definite:

1. **Decomposition**: $`A = LL^T`$
2. **Forward Substitution**: Solve $`Ly = b`$ for $`y`$
3. **Backward Substitution**: Solve $`L^T x = y`$ for $`x`$

**Advantages over LU:**
- **Efficiency**: Requires half the operations of LU
- **Stability**: No pivoting needed for positive definite matrices
- **Memory**: Can overwrite original matrix

**Applications:**
- **Covariance Matrices**: Natural for multivariate statistics
- **Finite Element Methods**: Stiffness matrices are positive definite
- **Optimization**: Hessian matrices in Newton's method

## Eigenvalue Decomposition

Eigenvalue decomposition factors a diagonalizable matrix $`A`$ into $`A = PDP^{-1}`$, where $`P`$ contains eigenvectors and $`D`$ is diagonal.

### Mathematical Definition

**Definition:**
For a diagonalizable matrix $`A \in \mathbb{R}^{n \times n}`$:
```math
A = PDP^{-1}
```

where:
- $`P \in \mathbb{R}^{n \times n}`$ contains eigenvectors as columns
- $`D \in \mathbb{R}^{n \times n}`$ is diagonal with eigenvalues
- $`A`$ must be diagonalizable (have $`n`$ linearly independent eigenvectors)

**Properties:**
1. **Eigenvalues**: $`D_{ii} = \lambda_i`$ are the eigenvalues of $`A`$
2. **Eigenvectors**: $`P_{:,i}`$ is the eigenvector corresponding to $`\lambda_i`$
3. **Powers**: $`A^k = PD^k P^{-1}`$ for any integer $`k`$
4. **Functions**: $`f(A) = Pf(D) P^{-1}`$ for analytic functions $`f`$

**Existence:**
- **Sufficient Condition**: If $`A`$ has $`n`$ distinct eigenvalues, it is diagonalizable
- **Necessary Condition**: $`A`$ must have $`n`$ linearly independent eigenvectors
- **Symmetric Matrices**: Always diagonalizable with orthogonal eigenvectors

### Algorithm: Power Iteration

**Mathematical Foundation:**
For finding the dominant eigenvalue and eigenvector:

1. **Initialize**: $`x_0`$ (random vector)
2. **Iterate**: $`x_{k+1} = \frac{A x_k}{\|A x_k\|_2}`$
3. **Converge**: $`x_k \to v_1`$ (dominant eigenvector)
4. **Eigenvalue**: $`\lambda_1 = \frac{x_k^T A x_k}{x_k^T x_k}`$

**Properties:**
- **Convergence Rate**: Linear with ratio $`|\lambda_2/\lambda_1|`$
- **Dominant Eigenvalue**: Finds largest eigenvalue in magnitude
- **Extensions**: Can find multiple eigenvalues with deflation

## Applications in Machine Learning

### Principal Component Analysis (PCA)

**Mathematical Foundation:**
PCA finds the principal components of data matrix $`X \in \mathbb{R}^{n \times d}`$:

1. **Center Data**: $`X' = X - \bar{X}`$
2. **Covariance Matrix**: $`C = \frac{1}{n-1} X'^T X'`$
3. **Eigenvalue Decomposition**: $`C = V \Lambda V^T`$
4. **Projection**: $`Y = X' V`$

**SVD Connection:**
PCA can also be computed via SVD of $`X'`$:
```math
X' = U \Sigma V^T
```

The principal components are the right singular vectors $`V`$, and the explained variance is proportional to $`\sigma_i^2`$.

**Applications:**
- **Dimensionality Reduction**: Keep top-$`k`$ components
- **Data Visualization**: Project to 2D/3D for plotting
- **Feature Engineering**: Create new features from principal components
- **Noise Reduction**: Remove components with low variance

### Matrix Factorization for Recommender Systems

**Mathematical Foundation:**
Matrix factorization models user-item ratings as:
```math
R \approx U V^T
```

where:
- $`R \in \mathbb{R}^{m \times n}`$ is the rating matrix
- $`U \in \mathbb{R}^{m \times k}`$ contains user embeddings
- $`V \in \mathbb{R}^{n \times k}`$ contains item embeddings

**Optimization Problem:**
```math
\min_{U,V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i^T V_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
```

where $`\Omega`$ is the set of observed ratings.

**Algorithms:**
1. **Alternating Least Squares**: Fix $`U`$, solve for $`V`$, then vice versa
2. **Stochastic Gradient Descent**: Update embeddings for each rating
3. **SVD-based**: Use truncated SVD for initialization

### Linear Regression and Regularization

**Mathematical Foundation:**
Linear regression solves:
```math
\min_w \|Xw - y\|_2^2 + \lambda \|w\|_2^2
```

**QR Solution:**
1. Compute $`X = QR`$
2. Solve $`Rw = Q^T y`$ by back-substitution

**Ridge Regression:**
The solution is:
```math
w = (X^T X + \lambda I)^{-1} X^T y
```

**Cholesky Solution:**
1. Compute $`X^T X + \lambda I = LL^T`$
2. Solve $`LL^T w = X^T y`$ by forward/backward substitution

### Neural Network Initialization

**Mathematical Foundation:**
Orthogonal initialization for neural networks:

1. **Generate**: Random matrix $`W \in \mathbb{R}^{m \times n}`$
2. **SVD**: $`W = U \Sigma V^T`$
3. **Initialize**: $`W = UV^T`$ (orthogonal matrix)

**Properties:**
- **Orthogonality**: $`W^T W = I`$ (for $`m \geq n`$)
- **Gradient Flow**: Prevents vanishing/exploding gradients
- **Feature Diversity**: Ensures neurons learn different features

## Exercises

### Exercise 1: LU Decomposition

**Objective**: Implement and analyze LU decomposition with partial pivoting.

**Tasks:**
1. Implement LU decomposition with partial pivoting
2. Test on matrix $`A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & -6 & 0 \\ -2 & 7 & 2 \end{bmatrix}`$
3. Verify $`PA = LU`$ where $`P`$ is the permutation matrix
4. Solve the system $`Ax = [1, -2, 7]^T`$

### Exercise 2: QR Decomposition

**Objective**: Implement QR decomposition and apply to least squares.

**Tasks:**
1. Implement Householder QR decomposition
2. Test on matrix $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}`$
3. Verify $`A = QR`$ and $`Q^T Q = I`$
4. Solve least squares problem $`\min \|Ax - b\|_2`$ where $`b = [1, 2, 3]^T`$

### Exercise 3: SVD and Low-Rank Approximation

**Objective**: Implement SVD and analyze low-rank approximations.

**Tasks:**
1. Implement SVD using numpy.linalg.svd
2. Test on matrix $`A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}`$
3. Compute rank-1 and rank-2 approximations
4. Compare reconstruction error for different ranks

### Exercise 4: Cholesky Decomposition

**Objective**: Implement Cholesky decomposition for positive definite matrices.

**Tasks:**
1. Implement Cholesky decomposition
2. Test on matrix $`A = \begin{bmatrix} 4 & 12 & -16 \\ 12 & 37 & -43 \\ -16 & -43 & 98 \end{bmatrix}`$
3. Verify $`A = LL^T`$
4. Solve system $`Ax = [1, 2, 3]^T`$ using Cholesky

### Exercise 5: Eigenvalue Decomposition

**Objective**: Implement power iteration and analyze eigenvalue decomposition.

**Tasks:**
1. Implement power iteration for dominant eigenvalue
2. Test on symmetric matrix $`A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}`$
3. Compare with numpy.linalg.eig
4. Analyze convergence rate

### Exercise 6: PCA Implementation

**Objective**: Implement PCA from scratch using SVD.

**Tasks:**
1. Generate synthetic data with known structure
2. Implement PCA using SVD
3. Analyze explained variance ratio
4. Visualize data in principal component space

## Solutions

### Solution 1: LU Decomposition

**Step 1: Implementation**
```python
def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)
    
    for k in range(n-1):
        # Find pivot
        p = k + np.argmax(np.abs(U[k:, k]))
        
        # Exchange rows
        U[k], U[p] = U[p].copy(), U[k].copy()
        L[k, :k], L[p, :k] = L[p, :k].copy(), L[k, :k].copy()
        P[k], P[p] = P[p].copy(), P[k].copy()
        
        # Eliminate
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return P, L, U
```

**Step 2: Verification**
```python
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
P, L, U = lu_decomposition(A)
print("PA = LU:", np.allclose(P @ A, L @ U))
```

**Step 3: System Solution**
```python
b = np.array([1, -2, 7])
# Forward substitution: Ly = Pb
y = np.linalg.solve(L, P @ b)
# Backward substitution: Ux = y
x = np.linalg.solve(U, y)
print("Solution:", x)
```

### Solution 2: QR Decomposition

**Step 1: Householder Implementation**
```python
def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for k in range(n):
        # Householder vector
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x - np.sign(x[0]) * np.linalg.norm(x) * e1
        
        # Householder matrix
        H = np.eye(m)
        H[k:, k:] -= 2 * np.outer(v, v) / (v @ v)
        
        # Apply transformation
        R = H @ R
        Q = Q @ H
    
    return Q, R
```

**Step 2: Least Squares Solution**
```python
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])
Q, R = householder_qr(A)

# Solve Rx = Q^T b
c = Q.T @ b
x = np.linalg.solve(R[:2, :2], c[:2])
print("Least squares solution:", x)
```

### Solution 3: SVD and Low-Rank Approximation

**Step 1: SVD Implementation**
```python
def low_rank_approximation(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Truncate to rank k
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct
    A_k = U_k @ np.diag(s_k) @ Vt_k
    return A_k, s
```

**Step 2: Analysis**
```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
A_1, s_1 = low_rank_approximation(A, 1)
A_2, s_2 = low_rank_approximation(A, 2)

print("Rank-1 error:", np.linalg.norm(A - A_1, 'fro'))
print("Rank-2 error:", np.linalg.norm(A - A_2, 'fro'))
print("Singular values:", s_1)
```

### Solution 4: Cholesky Decomposition

**Step 1: Implementation**
```python
def cholesky(A):
    n = len(A)
    L = np.zeros_like(A)
    
    for j in range(n):
        # Diagonal element
        L[j, j] = np.sqrt(A[j, j] - np.sum(L[j, :j]**2))
        
        # Off-diagonal elements
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    
    return L
```

**Step 2: System Solution**
```python
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
L = cholesky(A)
b = np.array([1, 2, 3])

# Solve LL^T x = b
y = np.linalg.solve(L, b)
x = np.linalg.solve(L.T, y)
print("Cholesky solution:", x)
```

### Solution 5: Eigenvalue Decomposition

**Step 1: Power Iteration**
```python
def power_iteration(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for i in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    lambda_1 = (x.T @ A @ x) / (x.T @ x)
    return lambda_1, x
```

**Step 2: Comparison**
```python
A = np.array([[2, 1], [1, 3]])
lambda_1, v_1 = power_iteration(A)
eigenvals, eigenvecs = np.linalg.eig(A)

print("Power iteration eigenvalue:", lambda_1)
print("Exact eigenvalues:", eigenvals)
```

### Solution 6: PCA Implementation

**Step 1: PCA Algorithm**
```python
def pca(X, n_components):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Project to principal components
    X_pca = X_centered @ Vt.T[:, :n_components]
    
    return X_pca, s, Vt
```

**Step 2: Analysis**
```python
# Generate data
np.random.seed(42)
X = np.random.randn(100, 3)
X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1]  # Linear dependency

X_pca, s, Vt = pca(X, 2)
explained_variance = s**2 / (len(X) - 1)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Explained variance ratio:", explained_variance_ratio[:2])
```

## Summary

In this comprehensive chapter on matrix decompositions, we have covered:

### Key Decompositions
- **LU Decomposition**: For solving linear systems efficiently
- **QR Decomposition**: For least squares problems and numerical stability
- **SVD**: For dimensionality reduction and matrix approximation
- **Cholesky Decomposition**: For positive definite matrices
- **Eigenvalue Decomposition**: For diagonalizable matrices

### Mathematical Foundations
- **Existence and Uniqueness**: Conditions for each decomposition
- **Algorithms**: Efficient computational methods
- **Numerical Stability**: Considerations for floating-point arithmetic
- **Geometric Interpretation**: Understanding decompositions as coordinate transformations

### Machine Learning Applications
- **PCA**: Dimensionality reduction using SVD
- **Recommender Systems**: Matrix factorization for collaborative filtering
- **Linear Regression**: QR decomposition for least squares
- **Neural Networks**: Orthogonal initialization and weight analysis

### Practical Skills
- **Implementation**: Writing robust decomposition algorithms
- **Numerical Analysis**: Understanding stability and conditioning
- **Problem Solving**: Choosing appropriate decompositions for different problems
- **Performance Analysis**: Computational complexity and efficiency

### Advanced Topics
- **Condition Numbers**: Measuring numerical stability
- **Low-Rank Approximations**: Optimal matrix approximations
- **Regularization**: Incorporating prior knowledge in decompositions
- **Parallel Algorithms**: Scalable implementations for large matrices

Matrix decompositions are essential tools for understanding matrix structure and implementing efficient algorithms in machine learning and data science. They provide the mathematical foundation for many advanced techniques and help us design better models and understand data structure.

## Next Steps

In the next chapter, we'll explore applications of linear algebra in machine learning, including linear regression, neural networks, and optimization. 