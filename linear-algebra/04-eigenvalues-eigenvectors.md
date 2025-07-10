# Eigenvalues and Eigenvectors

[![Chapter](https://img.shields.io/badge/Chapter-4-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Eigenvalues_Eigenvectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Eigenvalues and eigenvectors are fundamental concepts in linear algebra that reveal the intrinsic properties of matrices. They are crucial for understanding matrix behavior, diagonalization, and applications in machine learning like Principal Component Analysis (PCA), spectral clustering, and dimensionality reduction. Eigenvalues and eigenvectors provide a "natural coordinate system" for understanding how matrices transform space.

### Why Eigenvalues and Eigenvectors Matter in AI/ML

1. **Dimensionality Reduction**: PCA uses eigenvectors to find the most important directions in data.
2. **Spectral Clustering**: Uses eigenvectors of similarity matrices for clustering.
3. **PageRank Algorithm**: Uses eigenvectors to rank web pages.
4. **Neural Networks**: Weight matrices have eigenvalues that affect training dynamics.
5. **Optimization**: Hessian eigenvalues determine convergence properties.
6. **Signal Processing**: Fourier transforms and other spectral methods use eigenvectors.

## What are Eigenvalues and Eigenvectors?

### Mathematical Definition

For a square matrix $`A`$ of size $`n \times n`$, a non-zero vector $`\vec{v} \in \mathbb{R}^n`$ is an **eigenvector** if:
```math
A\vec{v} = \lambda\vec{v}
```
where $`\lambda`$ is a scalar called the **eigenvalue** corresponding to $`\vec{v}`$.

### Geometric Interpretation

Eigenvectors are special vectors that don't change direction when transformed by the matrix $`A`$. They only get scaled by their corresponding eigenvalue:

- If $`\lambda > 0`$: The eigenvector is stretched by factor $`\lambda`$.
- If $`\lambda < 0`$: The eigenvector is stretched by factor $`|\lambda|`$ and flipped.
- If $`\lambda = 0`$: The eigenvector is mapped to the zero vector.
- If $`|\lambda| > 1`$: The eigenvector is expanded.
- If $`|\lambda| < 1`$: The eigenvector is contracted.

### Fundamental Properties

1. **Eigenvectors are not unique**: If $`\vec{v}`$ is an eigenvector, then $`c\vec{v}`$ is also an eigenvector for any scalar $`c \neq 0`$.
2. **Eigenvalues are unique**: Each eigenvector corresponds to exactly one eigenvalue.
3. **Eigenvectors can be complex**: Even for real matrices, eigenvalues and eigenvectors can be complex numbers.
4. **Number of eigenvalues**: An $`n \times n`$ matrix has exactly $`n`$ eigenvalues (counting multiplicities).

## Finding Eigenvalues and Eigenvectors

### Characteristic Equation

The eigenvalues are solutions to the characteristic equation:
```math
\det(A - \lambda I) = 0
```
This is a polynomial equation of degree $`n`$ in $`\lambda`$, called the characteristic polynomial.

### Step-by-Step Process

1. **Form the matrix** $`A - \lambda I`$
2. **Compute the determinant** $`\det(A - \lambda I)`$
3. **Set equal to zero** and solve for $`\lambda`$
4. **For each eigenvalue** $`\lambda_i`$, solve $`(A - \lambda_i I)\vec{v} = \vec{0}`$ for $`\vec{v}`$

## Properties of Eigenvalues and Eigenvectors

### Basic Properties

1. **Trace and Sum**: $`\text{tr}(A) = \sum_{i=1}^{n} \lambda_i`$
2. **Determinant and Product**: $`\det(A) = \prod_{i=1}^{n} \lambda_i`$
3. **Powers**: If $`\lambda`$ is an eigenvalue of $`A`$, then $`\lambda^k`$ is an eigenvalue of $`A^k`$
4. **Inverse**: If $`\lambda`$ is an eigenvalue of $`A`$, then $`1/\lambda`$ is an eigenvalue of $`A^{-1}`$ (if $`A`$ is invertible)

### Eigenvalues of Special Matrices

- **Diagonal Matrix**: The eigenvalues are the diagonal entries.
- **Triangular Matrix**: The eigenvalues are the diagonal entries.
- **Symmetric Matrix**: All eigenvalues are real.
- **Orthogonal Matrix**: All eigenvalues have absolute value 1.
- **Skew-Symmetric Matrix**: Eigenvalues are purely imaginary or zero.

## Diagonalization

### Mathematical Definition

A matrix $`A`$ is **diagonalizable** if it can be written as:
```math
A = PDP^{-1}
```
where:
- $`P`$ is an invertible matrix whose columns are eigenvectors of $`A`$
- $`D`$ is a diagonal matrix whose diagonal elements are the corresponding eigenvalues
- $`P^{-1}`$ is the inverse of $`P`$

### Conditions for Diagonalization

A matrix is diagonalizable if and only if:
1. It has $`n`$ linearly independent eigenvectors (where $`n`$ is the size of the matrix)
2. The geometric multiplicity equals the algebraic multiplicity for each eigenvalue

### Geometric Interpretation

Diagonalization represents a change of basis to the "eigenvector basis" where the matrix becomes diagonal. This makes many operations much simpler:

- **Powers**: $`A^k = PD^kP^{-1}`$
- **Exponential**: $`e^A = Pe^DP^{-1}`$
- **Functions**: $`f(A) = Pf(D)P^{-1}`$

## Power Method

The power method is an iterative algorithm to find the dominant eigenvalue and eigenvector of a matrix.

### Algorithm

1. Start with a random vector $`\vec{v}_0`$
2. Iterate: $`\vec{v}_{k+1} = \frac{A\vec{v}_k}{\|A\vec{v}_k\|}`$
3. The eigenvalue is approximated by: $`\lambda \approx \frac{\vec{v}_k^T A \vec{v}_k}{\vec{v}_k^T \vec{v}_k}`$

### Convergence

The power method converges to the eigenvalue with the largest magnitude, provided:
- The matrix has a unique dominant eigenvalue.
- The initial vector has a non-zero component in the direction of the dominant eigenvector.

## Applications in Machine Learning

### Principal Component Analysis (PCA)

PCA uses eigenvectors of the covariance matrix to find the principal components (directions of maximum variance).

### Spectral Clustering

Spectral clustering uses eigenvectors of the Laplacian matrix to perform clustering.

### PageRank Algorithm

PageRank uses the dominant eigenvector of the transition matrix to rank web pages.

## Visualization of Eigenvalues and Eigenvectors

*Visualizing eigenvectors as directions that remain unchanged under a matrix transformation, and eigenvalues as the scaling factors, can help build intuition. Try sketching how a matrix transforms a set of basis vectors and its eigenvectors.*

## Exercises

### Exercise 1: Eigenvalue Properties

Verify the trace, determinant, and reality properties of eigenvalues for different types of matrices including symmetric, skew-symmetric, triangular, and random matrices.

### Exercise 2: Power Method Implementation

Implement and test variations of the power method to find dominant and smallest eigenvalues, comparing results with exact calculations.

### Exercise 3: Diagonalization

Test diagonalization with different types of matrices including diagonalizable, non-diagonalizable, and symmetric matrices, verifying reconstruction accuracy.

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