# Numerical Linear Algebra

## Introduction

Numerical linear algebra is the study of algorithms for performing linear algebra computations on computers with finite precision arithmetic. This field bridges the gap between theoretical linear algebra and practical computational methods, addressing issues of numerical stability, computational efficiency, and algorithmic complexity.

**Mathematical Foundation:**
Numerical linear algebra deals with the practical implementation of mathematical concepts:
- **Finite Precision**: Computers represent real numbers with limited precision (typically 64 bits for double precision)
- **Rounding Errors**: Accumulation of small errors in arithmetic operations
- **Conditioning**: Sensitivity of problems to small perturbations in input data
- **Stability**: Ability of algorithms to produce accurate results despite rounding errors

**Key Challenges:**
1. **Numerical Stability**: Ensuring algorithms don't amplify rounding errors
2. **Computational Complexity**: Balancing accuracy with computational cost
3. **Memory Efficiency**: Handling large-scale problems within memory constraints
4. **Parallelization**: Exploiting modern hardware for performance

**Geometric Interpretation:**
Numerical issues can be understood geometrically:
- **Ill-conditioned problems**: Small changes in input cause large changes in output
- **Stable algorithms**: Preserve geometric relationships despite numerical errors
- **Convergence**: Iterative methods approach solutions through geometric optimization

**Applications in Modern Computing:**
- **Machine Learning**: Large-scale matrix operations in neural networks
- **Scientific Computing**: Solving partial differential equations
- **Data Science**: Principal component analysis and dimensionality reduction
- **Computer Graphics**: 3D transformations and rendering
- **Signal Processing**: Filtering and spectral analysis

## 1. Numerical Stability and Conditioning

### Mathematical Foundation

**Condition Number:**
The condition number measures how sensitive a problem is to perturbations in the input data. For a matrix $`A`$, the condition number is:
```math
\kappa(A) = \|A\| \cdot \|A^{-1}\|
```

where $`\|\cdot\|`$ is a matrix norm (typically the 2-norm).

**Error Analysis:**
For the linear system $`Ax = b`$, if we perturb $`b`$ by $`\delta b`$, the solution changes by $`\delta x`$:
```math
(A + \delta A)(x + \delta x) = b + \delta b
```

The relative error bound is:
```math
\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \cdot \left(\frac{\|\delta A\|}{\|A\|} + \frac{\|\delta b\|}{\|b\|}\right)
```

**Backward Error Analysis:**
Instead of asking "how accurate is the computed solution?", we ask "for what perturbed problem is our computed solution exact?"

**Stability Definitions:**
1. **Forward Stability**: Computed solution is close to exact solution
2. **Backward Stability**: Computed solution is exact for slightly perturbed problem
3. **Mixed Stability**: Combination of forward and backward stability

### Condition Number Analysis

**Mathematical Foundation:**
The condition number can be computed using singular values:
```math
\kappa_2(A) = \frac{\sigma_1}{\sigma_n}
```

where $`\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n`$ are the singular values of $`A`$.

**Properties:**
- $`\kappa(A) \geq 1`$ for any matrix $`A`$
- $`\kappa(A) = 1`$ if and only if $`A`$ is a multiple of an orthogonal matrix
- $`\kappa(AB) \leq \kappa(A) \cdot \kappa(B)`$
- $`\kappa(A^T A) = \kappa(A)^2`$

**Interpretation:**
- $`\kappa \approx 1`$: Well-conditioned problem
- $`\kappa \approx 10^6`$: Moderately ill-conditioned
- $`\kappa \approx 10^{12}`$: Severely ill-conditioned (may lose all precision)

**Examples:**
1. **Hilbert Matrix**: $`H_{ij} = \frac{1}{i+j-1}`$ has condition number growing exponentially with size
2. **Identity Matrix**: $`\kappa(I) = 1`$ (perfectly conditioned)
3. **Random Matrix**: Typically well-conditioned for large matrices

### Numerical Stability Examples

**Example 1: Linear System Solution**
Consider the system:
```math
\begin{bmatrix} 1 & 1 \\ 1 & 1.0001 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 \\ 2.0001 \end{bmatrix}
```

The exact solution is $`x_1 = 1`$, $`x_2 = 1`$. However, small perturbations can cause large changes in the solution due to the high condition number.

**Example 2: Matrix Inversion**
For a nearly singular matrix:
```math
A = \begin{bmatrix} 1 & 1 \\ 1 & 1 + \epsilon \end{bmatrix}
```

The condition number is $`\kappa(A) \approx \frac{2}{\epsilon}`$, which becomes very large as $`\epsilon \to 0`$.

**Example 3: Eigenvalue Computation**
The eigenvalues of a matrix can be very sensitive to perturbations. For example, the matrix:
```math
A = \begin{bmatrix} 1 & 1 \\ 0 & 1 + \epsilon \end{bmatrix}
```

has eigenvalues $`1`$ and $`1 + \epsilon`$, but small perturbations can cause complex eigenvalues.

## 2. Iterative Methods for Linear Systems

### Jacobi Method

**Mathematical Foundation:**
The Jacobi method decomposes matrix $`A`$ as $`A = D + L + U`$, where:
- $`D`$ is the diagonal of $`A`$
- $`L`$ is the strictly lower triangular part
- $`U`$ is the strictly upper triangular part

**Iteration Formula:**
```math
x^{(k+1)} = D^{-1}(b - (L + U)x^{(k)})
```

**Component-wise Form:**
```math
x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right)
```

**Convergence:**
- **Sufficient Condition**: $`A`$ is strictly diagonally dominant
- **Necessary Condition**: $`\rho(D^{-1}(L + U)) < 1`$ where $`\rho`$ is the spectral radius

**Algorithm:**
1. **Initialize**: $`x^{(0)} = 0`$
2. **For each iteration** $`k = 0, 1, \ldots`$:
   - $`x^{(k+1)} = D^{-1}(b - (L + U)x^{(k)})`$
   - Check convergence: $`\|x^{(k+1)} - x^{(k)}\| < \epsilon`$
3. **Return**: $`x^{(k+1)}`$

### Gauss-Seidel Method

**Mathematical Foundation:**
Gauss-Seidel uses the most recent values available:
```math
x^{(k+1)} = (D + L)^{-1}(b - U x^{(k)})
```

**Component-wise Form:**
```math
x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)}\right)
```

**Advantages over Jacobi:**
- Generally faster convergence
- Uses updated values immediately
- Better for parallel implementation

**Convergence:**
- **Sufficient Condition**: $`A`$ is symmetric positive definite
- **Rate**: Typically faster than Jacobi for well-conditioned systems

### Conjugate Gradient Method

**Mathematical Foundation:**
Conjugate gradient is an iterative method for symmetric positive definite systems that minimizes the quadratic form:
```math
\phi(x) = \frac{1}{2} x^T A x - x^T b
```

**Algorithm:**
1. **Initialize**: $`x^{(0)} = 0`$, $`r^{(0)} = b`$, $`p^{(0)} = r^{(0)}`$
2. **For each iteration** $`k = 0, 1, \ldots, n-1`$:
   - $`\alpha_k = \frac{(r^{(k)})^T r^{(k)}}{(p^{(k)})^T A p^{(k)}}`$
   - $`x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)}`$
   - $`r^{(k+1)} = r^{(k)} - \alpha_k A p^{(k)}`$
   - $`\beta_k = \frac{(r^{(k+1)})^T r^{(k+1)}}{(r^{(k)})^T r^{(k)}}`$
   - $`p^{(k+1)} = r^{(k+1)} + \beta_k p^{(k)}`$

**Properties:**
- **Exact Solution**: Converges in at most $`n`$ iterations (in exact arithmetic)
- **Optimality**: Minimizes error in $`A`$-norm at each step
- **Efficiency**: Only requires matrix-vector products

**Convergence Rate:**
```math
\|x^{(k)} - x^*\|_A \leq 2\left(\frac{\sqrt{\kappa(A)} - 1}{\sqrt{\kappa(A)} + 1}\right)^k \|x^{(0)} - x^*\|_A
```

## 3. Sparse Matrix Operations

### Sparse Matrix Formats

**Mathematical Foundation:**
Sparse matrices have most entries zero, allowing efficient storage and computation.

**Common Formats:**

**1. Coordinate Format (COO):**
- Store triplets $`(i, j, value)`$ for non-zero entries
- Memory: $`O(nnz)`$ where $`nnz`$ is number of non-zeros
- Good for construction, poor for operations

**2. Compressed Sparse Row (CSR):**
- Row pointers, column indices, values
- Memory: $`O(nnz + n)`$
- Efficient for matrix-vector multiplication

**3. Compressed Sparse Column (CSC):**
- Column pointers, row indices, values
- Memory: $`O(nnz + n)`$
- Efficient for column operations

**4. Diagonal Format (DIA):**
- Store diagonals with offsets
- Memory: $`O(nd)`$ where $`d`$ is number of diagonals
- Efficient for banded matrices

### Sparse Matrix-Vector Multiplication

**Mathematical Foundation:**
For sparse matrix $`A`$ and vector $`x`$:
```math
y_i = \sum_{j: a_{ij} \neq 0} a_{ij} x_j
```

**CSR Implementation:**
```python
def csr_matvec(A_data, A_indices, A_indptr, x):
    n = len(A_indptr) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(A_indptr[i], A_indptr[i+1]):
            y[i] += A_data[j] * x[A_indices[j]]
    
    return y
```

**Performance Considerations:**
- **Memory Access Pattern**: Cache-friendly access improves performance
- **Load Balancing**: Parallel implementation requires careful load balancing
- **Compression**: Further compression possible for structured sparsity

### Sparse Direct Solvers

**Mathematical Foundation:**
Sparse direct solvers use matrix factorizations adapted for sparse matrices.

**LU Decomposition for Sparse Matrices:**
1. **Symbolic Factorization**: Determine sparsity pattern of factors
2. **Numeric Factorization**: Compute actual values
3. **Forward/Backward Substitution**: Solve triangular systems

**Fill-in:**
The process of creating new non-zeros during factorization. Minimizing fill-in is crucial for efficiency.

**Ordering Strategies:**
- **Minimum Degree**: Minimize fill-in locally
- **Nested Dissection**: Recursive graph partitioning
- **Approximate Minimum Degree (AMD)**: Heuristic for large matrices

## 4. Eigenvalue Problems

### Power Iteration

**Mathematical Foundation:**
Power iteration finds the dominant eigenvalue and eigenvector:
```math
x^{(k+1)} = \frac{A x^{(k)}}{\|A x^{(k)}\|_2}
```

**Convergence:**
```math
\|x^{(k)} - v_1\|_2 \leq C \left|\frac{\lambda_2}{\lambda_1}\right|^k
```

where $`v_1`$ is the dominant eigenvector and $`\lambda_1, \lambda_2`$ are the largest eigenvalues.

**Algorithm:**
1. **Initialize**: $`x^{(0)}`$ (random vector)
2. **For each iteration** $`k = 0, 1, \ldots`$:
   - $`y^{(k+1)} = A x^{(k)}`$
   - $`x^{(k+1)} = \frac{y^{(k+1)}}{\|y^{(k+1)}\|_2}`$
   - Check convergence: $`\|x^{(k+1)} - x^{(k)}\|_2 < \epsilon`$
3. **Eigenvalue**: $`\lambda_1 = \frac{(x^{(k)})^T A x^{(k)}}{(x^{(k)})^T x^{(k)}}`$

**Properties:**
- **Convergence Rate**: Linear with ratio $`|\lambda_2/\lambda_1|`$
- **Dominant Eigenvalue**: Finds largest eigenvalue in magnitude
- **Extensions**: Can find multiple eigenvalues with deflation

### Inverse Iteration

**Mathematical Foundation:**
Inverse iteration finds eigenvalues closest to a shift $`\mu`$:
```math
x^{(k+1)} = \frac{(A - \mu I)^{-1} x^{(k)}}{\|(A - \mu I)^{-1} x^{(k)}\|_2}
```

**Convergence:**
```math
\|x^{(k)} - v_i\|_2 \leq C \left|\frac{\lambda_i - \mu}{\lambda_j - \mu}\right|^k
```

where $`\lambda_j`$ is the eigenvalue closest to $`\mu`$.

**Algorithm:**
1. **Initialize**: $`x^{(0)}`$ (random vector), choose shift $`\mu`$
2. **For each iteration** $`k = 0, 1, \ldots`$:
   - Solve: $`(A - \mu I) y^{(k+1)} = x^{(k)}`$
   - Normalize: $`x^{(k+1)} = \frac{y^{(k+1)}}{\|y^{(k+1)}\|_2}`$
3. **Eigenvalue**: $`\lambda_i = \mu + \frac{1}{(x^{(k)})^T y^{(k+1)}}`$

**Applications:**
- **Shift-and-Invert**: Find eigenvalues near specific values
- **Rayleigh Quotient Iteration**: Adaptive shift selection
- **Deflation**: Remove found eigenvalues to find others

## 5. QR Algorithm for Eigenvalues

### Mathematical Foundation

**QR Decomposition:**
Every matrix $`A`$ can be written as:
```math
A = QR
```

where $`Q`$ is orthogonal and $`R`$ is upper triangular.

**QR Algorithm:**
1. **Initialize**: $`A_0 = A`$
2. **For each iteration** $`k = 0, 1, \ldots`$:
   - Compute QR decomposition: $`A_k = Q_k R_k`$
   - Update: $`A_{k+1} = R_k Q_k`$
3. **Convergence**: $`A_k`$ converges to upper triangular form with eigenvalues on diagonal

**Shifted QR Algorithm:**
```math
A_k - \mu_k I = Q_k R_k
A_{k+1} = R_k Q_k + \mu_k I
```

**Wilkinson Shift:**
```math
\mu_k = a_{nn}^{(k)} - \frac{(a_{n,n-1}^{(k)})^2}{a_{n-1,n-1}^{(k)} - a_{nn}^{(k)}}
```

**Convergence Properties:**
- **Global Convergence**: Always converges for symmetric matrices
- **Local Convergence**: Quadratic convergence for simple eigenvalues
- **Deflation**: Can deflate converged eigenvalues

### Implementation Details

**Hessenberg Form:**
Reduce matrix to upper Hessenberg form before QR iterations:
```math
A = U^T H U
```

where $`H`$ is upper Hessenberg (zeros below first subdiagonal).

**Givens Rotations:**
Use Givens rotations to zero subdiagonal elements:
```math
G_{i,i+1} = \begin{bmatrix} c & -s \\ s & c \end{bmatrix}
```

where $`c = \frac{h_{i,i}}{\sqrt{h_{i,i}^2 + h_{i+1,i}^2}}`$ and $`s = \frac{h_{i+1,i}}{\sqrt{h_{i,i}^2 + h_{i+1,i}^2}}`$.

**Implicit QR:**
Avoid explicit QR decomposition by applying Givens rotations directly to the matrix.

## 6. Singular Value Decomposition (SVD) for Large Matrices

### Mathematical Foundation

**SVD for Large Matrices:**
For large matrices, computing full SVD is expensive. We use iterative methods to find partial SVD.

**Power Method for SVD:**
```math
u^{(k+1)} = \frac{A v^{(k)}}{\|A v^{(k)}\|_2}
v^{(k+1)} = \frac{A^T u^{(k+1)}}{\|A^T u^{(k+1)}\|_2}
```

**Lanczos Method:**
Build Krylov subspace for $`A^T A`$:
```math
\mathcal{K}_k(A^T A, v_1) = \text{span}\{v_1, A^T A v_1, \ldots, (A^T A)^{k-1} v_1\}
```

**Randomized SVD:**
1. **Generate**: Random matrix $`\Omega \in \mathbb{R}^{n \times k}`$
2. **Compute**: $`Y = A \Omega`$
3. **QR**: $`Y = QR`$
4. **Project**: $`B = Q^T A`$
5. **SVD**: $`B = \hat{U} \Sigma V^T`$
6. **Result**: $`A \approx Q \hat{U} \Sigma V^T`$

**Advantages:**
- **Efficiency**: $`O(mnk)`$ instead of $`O(mn^2)`$
- **Accuracy**: Good approximation for top-$`k`$ singular values
- **Parallelization**: Easy to parallelize

### Applications in Data Science

**Low-Rank Approximation:**
```math
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T
```

**Error Bound:**
```math
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
```

**Randomized PCA:**
1. **Project**: $`Y = A \Omega`$ where $`\Omega`$ is random
2. **QR**: $`Y = QR`$
3. **SVD**: $`Q^T A = U \Sigma V^T`$
4. **Result**: Principal components are columns of $`QU`$

**Performance Comparison:**
- **Classical SVD**: $`O(mn^2)`$ time, $`O(mn)`$ memory
- **Randomized SVD**: $`O(mnk)`$ time, $`O(mk + nk)`$ memory
- **Accuracy**: Randomized methods provide good approximations

## 7. Performance Comparison

### Computational Complexity

**Matrix Operations:**
| Operation | Dense | Sparse |
|-----------|-------|--------|
| Matrix-Vector | $`O(n^2)`$ | $`O(nnz)`$ |
| Matrix-Matrix | $`O(n^3)`$ | $`O(nnz \cdot n)`$ |
| LU Decomposition | $`O(n^3)`$ | $`O(nnz^{1.5})`$ |
| Eigenvalues | $`O(n^3)`$ | $`O(n^2)`$ |

**Memory Requirements:**
| Format | Memory |
|--------|--------|
| Dense | $`O(n^2)`$ |
| CSR | $`O(nnz + n)`$ |
| CSC | $`O(nnz + n)`$ |
| Diagonal | $`O(nd)`$ |

### Numerical Stability Comparison

**Direct Methods:**
- **Gaussian Elimination**: $`O(n^3)`$, can be unstable
- **LU with Pivoting**: $`O(n^3)`$, stable
- **QR Decomposition**: $`O(n^3)`$, very stable
- **Cholesky**: $`O(n^3/3)`$, stable for positive definite

**Iterative Methods:**
- **Jacobi**: $`O(n^2)`$ per iteration, slow convergence
- **Gauss-Seidel**: $`O(n^2)`$ per iteration, faster than Jacobi
- **Conjugate Gradient**: $`O(n^2)`$ per iteration, optimal for SPD

**Eigenvalue Methods:**
- **Power Iteration**: $`O(n^2)`$ per iteration, finds dominant eigenvalue
- **QR Algorithm**: $`O(n^3)`$, finds all eigenvalues
- **Lanczos**: $`O(n^2)`$ per iteration, finds extreme eigenvalues

### Practical Considerations

**Matrix Size Guidelines:**
- **Small** ($`n < 100`$): Use direct methods
- **Medium** ($`100 \leq n < 1000`$): Consider iterative methods
- **Large** ($`n \geq 1000`$): Use iterative or randomized methods

**Sparsity Guidelines:**
- **Dense**: Use dense algorithms
- **Sparse** ($`nnz = O(n)`$): Use sparse formats
- **Very Sparse** ($`nnz \ll n^2`$): Use specialized algorithms

**Hardware Considerations:**
- **Cache**: Optimize memory access patterns
- **Parallelism**: Use parallel algorithms for large matrices
- **GPU**: Accelerate matrix operations on GPU

## Exercises

### Exercise 1: Condition Number Analysis

**Objective**: Analyze condition numbers and their impact on numerical stability.

**Tasks:**
1. Create matrices with different condition numbers:
   - Identity matrix: $`I_n`$
   - Hilbert matrix: $`H_{ij} = \frac{1}{i+j-1}`$
   - Random matrix with controlled condition number
2. Solve linear systems $`Ax = b`$ with perturbed right-hand sides
3. Analyze relative errors and their relationship to condition numbers
4. Compare exact vs computed solutions

### Exercise 2: Iterative Methods Comparison

**Objective**: Implement and compare different iterative methods for linear systems.

**Tasks:**
1. Generate test matrices with known properties:
   - Symmetric positive definite
   - Diagonally dominant
   - Random sparse matrix
2. Implement Jacobi, Gauss-Seidel, and Conjugate Gradient methods
3. Compare convergence rates and final accuracy
4. Analyze performance vs matrix size and condition number

### Exercise 3: Sparse Matrix Operations

**Objective**: Implement sparse matrix formats and operations.

**Tasks:**
1. Create sparse matrices in different formats (COO, CSR, CSC)
2. Implement matrix-vector multiplication for each format
3. Compare memory usage and performance
4. Analyze sparsity patterns and their impact on performance

### Exercise 4: Eigenvalue Computation

**Objective**: Implement eigenvalue algorithms and analyze their properties.

**Tasks:**
1. Implement power iteration and inverse iteration
2. Test on matrices with known eigenvalues
3. Analyze convergence rates and accuracy
4. Compare with built-in eigenvalue solvers

### Exercise 5: QR Algorithm Implementation

**Objective**: Implement the QR algorithm for eigenvalue computation.

**Tasks:**
1. Implement basic QR algorithm
2. Add Wilkinson shift for faster convergence
3. Test on symmetric and non-symmetric matrices
4. Compare with other eigenvalue methods

### Exercise 6: Randomized SVD

**Objective**: Implement randomized SVD and compare with classical methods.

**Tasks:**
1. Implement randomized SVD algorithm
2. Test on large matrices with known structure
3. Compare accuracy and performance with classical SVD
4. Analyze the effect of oversampling parameter

## Solutions

### Solution 1: Condition Number Analysis

**Step 1: Matrix Generation**
```python
import numpy as np
from scipy.linalg import hilbert

def create_test_matrices(n=10):
    # Identity matrix (well-conditioned)
    I = np.eye(n)
    
    # Hilbert matrix (ill-conditioned)
    H = hilbert(n)
    
    # Random matrix with controlled condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.logspace(0, 6, n))  # Condition number ~10^6
    V, _ = np.linalg.qr(np.random.randn(n, n))
    A = U @ S @ V.T
    
    return I, H, A

# Compute condition numbers
I, H, A = create_test_matrices()
print(f"Identity matrix condition number: {np.linalg.cond(I):.2e}")
print(f"Hilbert matrix condition number: {np.linalg.cond(H):.2e}")
print(f"Random matrix condition number: {np.linalg.cond(A):.2e}")
```

**Step 2: Error Analysis**
```python
def solve_with_perturbation(A, b, perturbation_level=1e-10):
    # Exact solution
    x_exact = np.linalg.solve(A, b)
    
    # Perturbed right-hand side
    b_perturbed = b + perturbation_level * np.random.randn(len(b))
    x_perturbed = np.linalg.solve(A, b_perturbed)
    
    # Relative errors
    rel_error_b = np.linalg.norm(b_perturbed - b) / np.linalg.norm(b)
    rel_error_x = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)
    
    return rel_error_b, rel_error_x

# Test on different matrices
b = np.random.randn(10)
for name, matrix in [("Identity", I), ("Hilbert", H), ("Random", A)]:
    rel_error_b, rel_error_x = solve_with_perturbation(matrix, b)
    condition_number = np.linalg.cond(matrix)
    amplification = rel_error_x / rel_error_b
    print(f"{name}: Amplification factor = {amplification:.2e}, Condition number = {condition_number:.2e}")
```

### Solution 2: Iterative Methods Implementation

**Step 1: Jacobi Method**
```python
def jacobi_method(A, b, max_iter=1000, tol=1e-10):
    n = len(A)
    x = np.zeros(n)
    
    # Extract diagonal and off-diagonal parts
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    
    for k in range(max_iter):
        x_new = np.linalg.solve(D, b - L_plus_U @ x)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return x, k + 1

def gauss_seidel_method(A, b, max_iter=1000, tol=1e-10):
    n = len(A)
    x = np.zeros(n)
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
        
        if np.linalg.norm(x - x_old) < tol:
            break
    
    return x, k + 1

def conjugate_gradient_method(A, b, max_iter=None, tol=1e-10):
    n = len(A)
    if max_iter is None:
        max_iter = n
    
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    
    for k in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if np.linalg.norm(r_new) < tol:
            break
        
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    
    return x, k + 1
```

**Step 2: Performance Comparison**
```python
def create_test_systems(n=100):
    # Symmetric positive definite
    A_spd = np.random.randn(n, n)
    A_spd = A_spd.T @ A_spd + n * np.eye(n)
    
    # Diagonally dominant
    A_dd = np.random.randn(n, n)
    A_dd = A_dd + n * np.eye(n)
    
    # Random sparse
    A_sparse = np.random.randn(n, n)
    mask = np.random.random((n, n)) < 0.1  # 10% sparsity
    A_sparse[~mask] = 0
    A_sparse = A_sparse + n * np.eye(n)
    
    b = np.random.randn(n)
    return A_spd, A_dd, A_sparse, b

# Test all methods
A_spd, A_dd, A_sparse, b = create_test_systems()

for name, A in [("SPD", A_spd), ("Diagonally Dominant", A_dd), ("Sparse", A_sparse)]:
    print(f"\n{name} Matrix:")
    
    x_jacobi, iter_jacobi = jacobi_method(A, b)
    x_gs, iter_gs = gauss_seidel_method(A, b)
    x_cg, iter_cg = conjugate_gradient_method(A, b)
    
    x_exact = np.linalg.solve(A, b)
    
    print(f"Jacobi: {iter_jacobi} iterations, error: {np.linalg.norm(x_jacobi - x_exact):.2e}")
    print(f"Gauss-Seidel: {iter_gs} iterations, error: {np.linalg.norm(x_gs - x_exact):.2e}")
    print(f"Conjugate Gradient: {iter_cg} iterations, error: {np.linalg.norm(x_cg - x_exact):.2e}")
```

### Solution 3: Sparse Matrix Implementation

**Step 1: CSR Format**
```python
def dense_to_csr(A):
    """Convert dense matrix to CSR format"""
    n, m = A.shape
    data = []
    indices = []
    indptr = [0]
    
    for i in range(n):
        for j in range(m):
            if A[i, j] != 0:
                data.append(A[i, j])
                indices.append(j)
        indptr.append(len(data))
    
    return np.array(data), np.array(indices), np.array(indptr)

def csr_matvec(data, indices, indptr, x):
    """Matrix-vector multiplication for CSR format"""
    n = len(indptr) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(indptr[i], indptr[i+1]):
            y[i] += data[j] * x[indices[j]]
    
    return y
```

**Step 2: Performance Comparison**
```python
def create_sparse_matrix(n=1000, density=0.01):
    """Create random sparse matrix"""
    nnz = int(n * n * density)
    data = np.random.randn(nnz)
    row_indices = np.random.randint(0, n, nnz)
    col_indices = np.random.randint(0, n, nnz)
    
    # Create dense matrix
    A_dense = np.zeros((n, n))
    for i in range(nnz):
        A_dense[row_indices[i], col_indices[i]] = data[i]
    
    # Convert to CSR
    data_csr, indices_csr, indptr_csr = dense_to_csr(A_dense)
    
    return A_dense, data_csr, indices_csr, indptr_csr

# Performance test
n = 1000
A_dense, data_csr, indices_csr, indptr_csr = create_sparse_matrix(n, 0.01)
x = np.random.randn(n)

# Time dense multiplication
import time
start = time.time()
y_dense = A_dense @ x
time_dense = time.time() - start

# Time sparse multiplication
start = time.time()
y_sparse = csr_matvec(data_csr, indices_csr, indptr_csr, x)
time_sparse = time.time() - start

print(f"Dense multiplication: {time_dense:.4f}s")
print(f"Sparse multiplication: {time_sparse:.4f}s")
print(f"Speedup: {time_dense/time_sparse:.2f}x")
print(f"Memory dense: {A_dense.nbytes} bytes")
print(f"Memory sparse: {(data_csr.nbytes + indices_csr.nbytes + indptr_csr.nbytes)} bytes")
```

### Solution 4: Eigenvalue Methods

**Step 1: Power Iteration**
```python
def power_iteration(A, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for k in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    lambda_1 = (x.T @ A @ x) / (x.T @ x)
    return lambda_1, x, k + 1

def inverse_iteration(A, mu, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for k in range(max_iter):
        # Solve (A - mu*I)y = x
        y = np.linalg.solve(A - mu * np.eye(n), x)
        x_new = y / np.linalg.norm(y)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    lambda_i = mu + 1 / (x.T @ y)
    return lambda_i, x, k + 1
```

**Step 2: Testing**
```python
# Create test matrix
n = 100
A = np.random.randn(n, n)
A = A + A.T  # Make symmetric

# Find eigenvalues using different methods
eigenvals_exact, eigenvecs_exact = np.linalg.eig(A)
lambda_max_exact = np.max(np.abs(eigenvals_exact))
lambda_min_exact = np.min(np.abs(eigenvals_exact))

# Power iteration for largest eigenvalue
lambda_max_power, v_max, iter_max = power_iteration(A)
print(f"Largest eigenvalue:")
print(f"  Exact: {lambda_max_exact:.6f}")
print(f"  Power iteration: {lambda_max_power:.6f}")
print(f"  Error: {abs(lambda_max_power - lambda_max_exact):.2e}")
print(f"  Iterations: {iter_max}")

# Inverse iteration for smallest eigenvalue
lambda_min_inv, v_min, iter_min = inverse_iteration(A, 0)
print(f"\nSmallest eigenvalue:")
print(f"  Exact: {lambda_min_exact:.6f}")
print(f"  Inverse iteration: {lambda_min_inv:.6f}")
print(f"  Error: {abs(lambda_min_inv - lambda_min_exact):.2e}")
print(f"  Iterations: {iter_min}")
```

### Solution 5: QR Algorithm

**Step 1: Basic QR Algorithm**
```python
def qr_algorithm(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    A_k = A.copy()
    
    for k in range(max_iter):
        # QR decomposition
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        
        # Check convergence (off-diagonal elements)
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
    
    return np.diag(A_k), A_k, k + 1

def qr_algorithm_with_shift(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    A_k = A.copy()
    
    for k in range(max_iter):
        # Wilkinson shift
        if n > 1:
            a = A_k[n-2, n-2]
            b = A_k[n-2, n-1]
            c = A_k[n-1, n-1]
            delta = (a - c) / 2
            mu = c - b**2 / (delta + np.sign(delta) * np.sqrt(delta**2 + b**2))
        else:
            mu = A_k[0, 0]
        
        # Shifted QR
        Q, R = np.linalg.qr(A_k - mu * np.eye(n))
        A_k = R @ Q + mu * np.eye(n)
        
        # Check convergence
        off_diag = np.abs(A_k - np.diag(np.diag(A_k)))
        if np.max(off_diag) < tol:
            break
    
    return np.diag(A_k), A_k, k + 1
```

**Step 2: Comparison**
```python
# Create symmetric test matrix
n = 50
A = np.random.randn(n, n)
A = A + A.T

# Exact eigenvalues
eigenvals_exact = np.linalg.eigvals(A)
eigenvals_exact = np.sort(eigenvals_exact)

# QR algorithm without shift
eigenvals_qr, _, iter_qr = qr_algorithm(A)
eigenvals_qr = np.sort(eigenvals_qr)

# QR algorithm with shift
eigenvals_qr_shift, _, iter_qr_shift = qr_algorithm_with_shift(A)
eigenvals_qr_shift = np.sort(eigenvals_qr_shift)

print("Eigenvalue Comparison:")
print(f"Exact eigenvalues (first 5): {eigenvals_exact[:5]}")
print(f"QR without shift (first 5): {eigenvals_qr[:5]}")
print(f"QR with shift (first 5): {eigenvals_qr_shift[:5]}")
print(f"\nIterations:")
print(f"QR without shift: {iter_qr}")
print(f"QR with shift: {iter_qr_shift}")
print(f"\nErrors:")
print(f"QR without shift: {np.linalg.norm(eigenvals_qr - eigenvals_exact):.2e}")
print(f"QR with shift: {np.linalg.norm(eigenvals_qr_shift - eigenvals_exact):.2e}")
```

### Solution 6: Randomized SVD

**Step 1: Implementation**
```python
def randomized_svd(A, k, oversample=10):
    """Randomized SVD for large matrices"""
    m, n = A.shape
    l = k + oversample
    
    # Generate random matrix
    Omega = np.random.randn(n, l)
    
    # Compute Y = A * Omega
    Y = A @ Omega
    
    # QR decomposition of Y
    Q, _ = np.linalg.qr(Y)
    
    # Project A onto Q
    B = Q.T @ A
    
    # SVD of B
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Reconstruct U
    U = Q @ U_tilde
    
    return U[:, :k], S[:k], Vt[:k, :]

def randomized_svd_power_iteration(A, k, oversample=10, power_iter=2):
    """Randomized SVD with power iteration for better accuracy"""
    m, n = A.shape
    l = k + oversample
    
    # Generate random matrix
    Omega = np.random.randn(n, l)
    
    # Power iteration
    Y = A @ Omega
    for _ in range(power_iter):
        Y = A.T @ Y
        Y = A @ Y
    
    # QR decomposition
    Q, _ = np.linalg.qr(Y)
    
    # Project and compute SVD
    B = Q.T @ A
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    
    return U[:, :k], S[:k], Vt[:k, :]
```

**Step 2: Comparison**
```python
# Create large test matrix
m, n = 1000, 500
A = np.random.randn(m, n)
k = 10

# Exact SVD
U_exact, s_exact, Vt_exact = np.linalg.svd(A, full_matrices=False)
U_exact = U_exact[:, :k]
s_exact = s_exact[:k]
Vt_exact = Vt_exact[:k, :]

# Randomized SVD
U_rand, s_rand, Vt_rand = randomized_svd(A, k)

# Randomized SVD with power iteration
U_rand_power, s_rand_power, Vt_rand_power = randomized_svd_power_iteration(A, k)

# Compare accuracy
A_exact = U_exact @ np.diag(s_exact) @ Vt_exact
A_rand = U_rand @ np.diag(s_rand) @ Vt_rand
A_rand_power = U_rand_power @ np.diag(s_rand_power) @ Vt_rand_power

error_exact = np.linalg.norm(A - A_exact, 'fro')
error_rand = np.linalg.norm(A - A_rand, 'fro')
error_rand_power = np.linalg.norm(A - A_rand_power, 'fro')

print("SVD Comparison:")
print(f"Exact SVD error: {error_exact:.2e}")
print(f"Randomized SVD error: {error_rand:.2e}")
print(f"Randomized SVD with power iteration error: {error_rand_power:.2e}")
print(f"Error ratio (rand/exact): {error_rand/error_exact:.2f}")
print(f"Error ratio (rand_power/exact): {error_rand_power/error_exact:.2f}")
```

## Key Takeaways

- **Numerical stability is crucial** for reliable computations
- **Condition number measures problem sensitivity** to perturbations
- **Iterative methods are efficient** for large, sparse systems
- **Sparse matrix formats save memory** and computation time
- **Different eigenvalue algorithms** have different strengths
- **Performance varies significantly** with matrix size and structure
- **Randomized methods** provide good approximations for large matrices
- **Hardware considerations** impact algorithm choice and performance

## Next Chapter

In the final chapter, we'll provide a comprehensive summary of all concepts covered and present practice problems to reinforce understanding. 