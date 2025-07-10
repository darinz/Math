# Linear Transformations

[![Chapter](https://img.shields.io/badge/Chapter-3-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Linear_Transformations-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Linear transformations are functions that preserve vector addition and scalar multiplication. They are fundamental in understanding how matrices transform vectors and spaces, which is crucial for machine learning algorithms. Linear transformations form the mathematical foundation for neural networks, dimensionality reduction, and many other AI/ML techniques.

### Why Linear Transformations Matter in AI/ML

1. **Neural Networks**: Each layer applies a linear transformation followed by a nonlinear activation.
2. **Dimensionality Reduction**: PCA, SVD, and other techniques use linear transformations.
3. **Feature Engineering**: Linear combinations of features are linear transformations.
4. **Optimization**: Gradient descent and other algorithms work with linear approximations.
5. **Computer Vision**: Image transformations, rotations, scaling are linear operations.
6. **Signal Processing**: Filters and transforms are often linear.

## What is a Linear Transformation?

A function $`T: \mathbb{R}^n \rightarrow \mathbb{R}^m`$ is linear if it satisfies two fundamental properties:

1. **Additivity**: 

```math
T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})
```

      for all vectors
```math
\vec{u}, \vec{v} \in \mathbb{R}^n
```

2. **Homogeneity**: 
```math
T(c\vec{u}) = cT(\vec{u})
```

      for all scalars
```math
c \text{ and vectors } \vec{u} \in \mathbb{R}^n
```
### Mathematical Foundation

These properties can be combined into a single condition:
```math
T(a\vec{u} + b\vec{v}) = aT(\vec{u}) + bT(\vec{v})
```
This means linear transformations preserve linear combinations, which is why they're called "linear."

### Matrix Representation Theorem

**Fundamental Theorem**: Every linear transformation $`T: \mathbb{R}^n \rightarrow \mathbb{R}^m`$ can be represented by a unique $`m \times n`$ matrix $`A`$ such that:
```math
T(\vec{x}) = A\vec{x}
```
**Proof Sketch**:
1. Let $\{\vec{e}_1, \vec{e}_2, \ldots, \vec{e}_n\}$ be the standard basis for $\mathbb{R}^n$
2. Define $A$ as the matrix whose columns are 
```math
T(\vec{e}_1), T(\vec{e}_2), \ldots, T(\vec{e}_n)
```
3. For any vector 
```math
\vec{x} = x_1\vec{e}_1 + x_2\vec{e}_2 + \cdots + x_n\vec{e}_n
```

```math
T(\vec{x}) = T(x_1\vec{e}_1 + x_2\vec{e}_2 + \cdots + x_n\vec{e}_n)
= x_1T(\vec{e}_1) + x_2T(\vec{e}_2) + \cdots + x_nT(\vec{e}_n)
= A\vec{x}
```

### Geometric Interpretation

Linear transformations have several important geometric properties:
- **Preserve lines**: Lines remain lines (though they may be rotated, scaled, or sheared).
- **Preserve origin**: 
```math
T(\vec{0}) = \vec{0}.
```
- **Preserve parallelism**: Parallel lines remain parallel.
- **Preserve linear combinations**: The image of a linear combination is the linear combination of the images.

## Common Linear Transformations

### Scaling (Dilation/Contraction)

**Mathematical Definition**: Scaling transforms vectors by multiplying each component by a scalar factor.

**Matrix Form**: For uniform scaling by factor $`k`$:
```math
S = \begin{bmatrix} k & 0 \\ 0 & k \end{bmatrix}
```
**Properties**:
- Preserves angles between vectors.
- Changes lengths by factor $`|k|`$.
- If $`|k| > 1`$: dilation (expansion).
- If $`|k| < 1`$: contraction (compression).
- If $`k < 0`$: reflection through origin.

### Rotation

**Mathematical Definition**: Rotation transforms vectors by rotating them around the origin by a specified angle.

**Matrix Form**: For rotation by angle $`\theta`$ (counterclockwise):
```math
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
```
**Properties**:
- Preserves lengths:
```math
|R(\theta)\vec{v}| = |\vec{v}|
```
- Preserves angles between vectors.
- Determinant is 1: $`\det(R(\theta)) = 1`$
- Inverse is transpose: $`R(\theta)^{-1} = R(\theta)^T = R(-\theta)`$

### Reflection

**Mathematical Definition**: Reflection flips vectors across a specified line or plane.

**Matrix Forms**:
- Reflection across x-axis: $`R_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`$
- Reflection across y-axis: $`R_y = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}`$
- Reflection across line  $`y = x`$: $`R_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`$

**Properties**:
- Determinant is -1: $`\det(R) = -1`$
- $`R^2 = I`$ (applying reflection twice returns to original).
- Preserves lengths but changes orientation.

### Shear

**Mathematical Definition**: Shear transforms vectors by adding a multiple of one component to another, creating a "sliding" effect.

**Matrix Forms**:
- Horizontal shear: $`H_x = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}`$
- Vertical shear: $`H_y = \begin{bmatrix} 1 & 0 \\ k & 1 \end{bmatrix}`$

**Properties**:
- Determinant is 1: $`\det(H) = 1`$ (preserves area).
- Preserves lines parallel to the shear direction.
- Changes angles between vectors.

## Composition of Transformations

**Mathematical Principle**: Linear transformations can be combined by matrix multiplication. The composition of transformations $`T_1`$ and $`T_2`$ is given by:
```math
(T_2 \circ T_1)(\vec{x}) = T_2(T_1(\vec{x})) = A_2(A_1\vec{x}) = (A_2A_1)\vec{x}
```
**Important Note**: Matrix multiplication is not commutative, so the order of transformations matters!

## Properties of Linear Transformations

### Determinant and Area/Volume Scaling

**Key Property**: The determinant of a transformation matrix tells us how it affects area (in 2D) or volume (in 3D).

- $`|\det(A)| = 1`$: Preserves area/volume.
- $`|\det(A)| > 1`$: Expands area/volume.
- $`|\det(A)| < 1`$: Contracts area/volume.
- $`\det(A) < 0`$: Changes orientation (reflection).

### Eigenvalues and Eigenvectors

**Definition**: For a linear transformation $`T`$ represented by matrix $`A`$, a non-zero vector $`\vec{v}`$ is an eigenvector with eigenvalue $`\lambda`$ if:
```math
A\vec{v} = \lambda\vec{v}
```
**Geometric Interpretation**:
- Eigenvectors are vectors that don't change direction under the transformation.
- Eigenvalues tell us how much these vectors are scaled.
- The eigenvectors form a basis that diagonalizes the transformation.

## Applications in AI/ML

### Neural Network Layers

Neural network layers apply linear transformations to input data, followed by nonlinear activation functions.

### Principal Component Analysis (PCA)

PCA uses linear transformations to project data onto directions of maximum variance, which are found using eigenvectors of the covariance matrix.

## Visualization of Transformations

*Visualizing the effect of linear transformations on grids or shapes can help build intuition. Try sketching how a square or circle is transformed by scaling, rotation, or shear matrices.*

## Exercises

### Exercise 1: Linear Transformation Properties

Verify the additivity, homogeneity, and zero preservation properties for the matrix $`A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}`$.

### Exercise 2: Transformation Composition

Study the effect of transformation order by comparing different compositions of scaling, rotation, and shear transformations.

### Exercise 3: Eigenvalue Analysis

Analyze the eigenvalues, determinants, and traces of different transformation matrices including scaling, rotation, reflection, and shear.

## Summary

In this chapter, we've covered:

1. **Linear Transformation Fundamentals**: Definition, properties, and matrix representation
2. **Common Transformations**: Scaling, rotation, reflection, and shear with detailed mathematical foundations
3. **Transformation Composition**: How to combine transformations and why order matters
4. **Geometric Properties**: Determinant interpretation, eigenvalue analysis, and area/volume scaling
5. **AI/ML Applications**: Neural networks, PCA, and dimensionality reduction
6. **Visualization**: Comprehensive plotting of transformation effects

### Key Takeaways:
- Linear transformations preserve linear combinations and have important geometric properties
- Every linear transformation can be represented by a matrix
- The determinant tells us about area/volume scaling and orientation changes
- Eigenvalues and eigenvectors reveal the fundamental behavior of transformations
- Understanding linear transformations is crucial for neural networks and dimensionality reduction

### Next Steps:
In the next chapter, we'll explore eigenvalues and eigenvectors in detail, understanding how they reveal the fundamental structure of matrices and transformations. 