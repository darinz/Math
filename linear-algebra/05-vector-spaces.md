# Vector Spaces and Subspaces

## Introduction

Vector spaces provide the mathematical foundation for linear algebra. They are sets of vectors that satisfy certain axioms and are fundamental for understanding linear transformations, subspaces, and the structure of linear systems. The concept of a vector space generalizes familiar spaces like $`\mathbb{R}^n`$ and allows us to work with functions, polynomials, matrices, and more in a unified way.

### Why Vector Spaces Matter in AI/ML

1. **Feature Spaces**: Data points in ML are vectors in high-dimensional spaces
2. **Parameter Spaces**: Model parameters (weights) form vector spaces
3. **Function Spaces**: Solutions to differential equations, regression functions, and neural network outputs live in function spaces
4. **Subspaces**: Principal components, null spaces, and column spaces are all subspaces
5. **Basis and Dimension**: Feature selection, dimensionality reduction, and embeddings rely on these concepts

## What is a Vector Space?

A vector space $`V`$ over a field $`F`$ (usually $`\mathbb{R}`$ or $`\mathbb{C}`$) is a set of objects (called vectors) with two operations:
1. **Vector addition**: $`u + v \in V`$ for all $`u, v \in V`$
2. **Scalar multiplication**: $`cu \in V`$ for all $`c \in F, u \in V`$

### Axioms of a Vector Space
A set $`V`$ is a vector space if, for all $`u, v, w \in V`$ and $`c, d \in F`$:
- **Commutativity**: $`u + v = v + u`$
- **Associativity (addition)**: $`(u + v) + w = u + (v + w)`$
- **Additive identity**: There exists $`0 \in V`$ such that $`v + 0 = v`$
- **Additive inverse**: For each $`v \in V`$, there exists $`-v`$ such that $`v + (-v) = 0`$
- **Distributivity (vector)**: $`c(u + v) = cu + cv`$
- **Distributivity (scalar)**: $`(c + d)u = cu + du`$
- **Associativity (scalar)**: $`c(du) = (cd)u`$
- **Scalar identity**: $`1u = u`$

#### Examples and Counterexamples
- $`\mathbb{R}^n`$ is a vector space (all axioms hold)
- The set of all $`2 \times 2`$ matrices is a vector space
- The set of polynomials of degree $`\leq n`$ is a vector space
- The set of vectors in $`\mathbb{R}^2`$ with positive entries is **not** a vector space (not closed under scalar multiplication by negative numbers)
- The set of solutions to a homogeneous linear system is a vector space

## Common Vector Spaces

### $`\mathbb{R}^n`$ (Real Vector Space)
$`\mathbb{R}^n`$ is the set of all $`n`$-tuples of real numbers. It is the prototypical example of a vector space and forms the basis for most data representations in ML.

**Properties:**
- Dimension: $`n`$
- Standard basis: $`\{e_1, e_2, \ldots, e_n\}`$ where $`e_i`$ has 1 in position $`i`$ and 0 elsewhere
- Inner product: $`\langle u, v \rangle = u^T v = \sum_{i=1}^n u_i v_i`$

**Applications in ML:**
- Feature vectors: Each data point is a vector in $`\mathbb{R}^n`$ where $`n`$ is the number of features
- Parameter vectors: Model weights form vectors in $`\mathbb{R}^p`$ where $`p`$ is the number of parameters
- Embeddings: Word embeddings, image embeddings, etc. are vectors in high-dimensional spaces

### Matrix Spaces
The set of all $`m \times n`$ matrices with real entries, denoted $`M_{m \times n}(\mathbb{R})`$, is a vector space.

**Properties:**
- Dimension: $`mn`$
- Addition: $`(A + B)_{ij} = A_{ij} + B_{ij}`$
- Scalar multiplication: $`(cA)_{ij} = cA_{ij}`$

**Applications in ML:**
- Weight matrices in neural networks
- Covariance matrices in statistics
- Image representations (each pixel as matrix entry)

### Function Spaces
Function spaces are vector spaces where the elements are functions. For example, the set of all polynomials of degree $`\leq 2`$ forms a vector space.

**Properties:**
- Addition: $`(f + g)(x) = f(x) + g(x)`$
- Scalar multiplication: $`(cf)(x) = cf(x)`$
- Dimension: $`n+1`$ for polynomials of degree $`\leq n`$

**Applications in ML:**
- Kernel functions in support vector machines
- Activation functions in neural networks
- Regression functions

### Counterexample: Not a Vector Space
The set of vectors in $`\mathbb{R}^2`$ with positive $`x`$-coordinates is **not** a vector space because:
- It's not closed under scalar multiplication (multiply by -1)
- It doesn't contain the zero vector

## Subspaces

A subspace $`W`$ of a vector space $`V`$ is a subset that is itself a vector space under the same operations. Subspaces are important because they represent solution sets to homogeneous equations, null spaces, column spaces, and more.

### Subspace Criteria
For $`W`$ to be a subspace of $`V`$:
1. **Contains zero vector**: $`0 \in W`$
2. **Closure under addition**: $`u, v \in W \implies u + v \in W`$
3. **Closure under scalar multiplication**: $`u \in W, c \in F \implies cu \in W`$

If all three hold, $`W`$ automatically satisfies all vector space axioms (since it inherits them from $`V`$).

#### Examples
- The set of all vectors on a line through the origin in $`\mathbb{R}^2`$ is a subspace
- The set of all $`n \times n`$ symmetric matrices is a subspace of all $`n \times n`$ matrices
- The set of all solutions to $`Ax = 0`$ (the null space) is a subspace
- The set of all polynomials of degree $`\leq 2`$ is a subspace of all polynomials

#### Counterexamples
- The set of vectors in $`\mathbb{R}^2`$ with $`x \geq 0`$ is **not** a subspace (not closed under scalar multiplication by negative numbers)
- The set of vectors in $`\mathbb{R}^2`$ with $`x + y = 1`$ is **not** a subspace (does not contain the zero vector)

### Important Subspaces in ML

**Null Space (Kernel):**
- Set of vectors $`x`$ such that $`Ax = 0`$
- Represents redundant or dependent features
- Dimension is the number of free variables

**Column Space (Range):**
- Span of the columns of matrix $`A`$
- Represents all possible outputs $`Ax`$
- Dimension is the rank of $`A`$

**Row Space:**
- Span of the rows of matrix $`A`$
- Important for understanding the structure of linear systems

## Span

The span of a set of vectors is the set of all linear combinations of those vectors. The span is always a subspace.

**Mathematical Definition:**
```math
\text{span}\{v_1, \ldots, v_k\} = \{c_1 v_1 + \cdots + c_k v_k : c_i \in F\}
```

**Geometric Interpretation:**
- The span of one nonzero vector in $`\mathbb{R}^2`$ is a line through the origin
- The span of two linearly independent vectors in $`\mathbb{R}^2`$ is the whole plane
- The span of $`k`$ vectors in $`\mathbb{R}^n`$ is a $`\leq k`$-dimensional subspace

**Properties:**
- $`\text{span}\{v_1, \ldots, v_k\}`$ is the smallest subspace containing $`v_1, \ldots, v_k`$
- If $`W = \text{span}\{v_1, \ldots, v_k\}`$, then $`\{v_1, \ldots, v_k\}`$ is a spanning set for $`W`$

## Linear Independence

A set of vectors is linearly independent if no vector can be written as a linear combination of the others. Otherwise, they are linearly dependent.

**Mathematical Definition:**
Vectors $`v_1, \ldots, v_k`$ are linearly independent if the only solution to $`c_1 v_1 + \cdots + c_k v_k = 0`$ is $`c_1 = \cdots = c_k = 0`$.

**Geometric Interpretation:**
- In $`\mathbb{R}^2`$, two vectors are independent if they are not collinear
- In $`\mathbb{R}^3`$, three vectors are independent if they do not all lie in the same plane
- In $`\mathbb{R}^n`$, $`k`$ vectors are independent if they span a $`k`$-dimensional subspace

**Why It Matters:**
- The maximum number of linearly independent vectors in a space is its dimension
- Basis vectors must be linearly independent
- In ML, independent features provide unique information

**Testing Linear Independence:**
1. **Row reduction**: Put vectors as columns in a matrix and row reduce
2. **Determinant**: For $`n`$ vectors in $`\mathbb{R}^n`$, check if determinant is nonzero
3. **Rank**: The rank of the matrix formed by the vectors equals the number of independent vectors

## Basis and Dimension

A **basis** for a vector space is a linearly independent set that spans the space. Every vector in the space can be written uniquely as a linear combination of basis vectors.

**Mathematical Definition:**
A set $`\{v_1, \ldots, v_k\}`$ is a basis for $`V`$ if:
- The vectors are linearly independent
- $`\text{span}\{v_1, \ldots, v_k\} = V`$

**Dimension:**
The number of vectors in any basis for $`V`$ is called the **dimension** of $`V`$, denoted $`\dim(V)`$.

**Geometric Interpretation:**
- In $`\mathbb{R}^2`$, any two non-collinear vectors form a basis
- In $`\mathbb{R}^3`$, any three non-coplanar vectors form a basis
- The standard basis for $`\mathbb{R}^n`$ is $`\{e_1, \ldots, e_n\}`$ where $`e_i`$ has a 1 in the $`i`$-th position and 0 elsewhere

**Why It Matters:**
- The basis provides a coordinate system for the space
- Dimensionality reduction (PCA) finds a new basis for the data
- The number of features in ML is the dimension of the feature space
- Change of basis is fundamental in many ML algorithms

**Properties:**
- Every vector space has a basis
- All bases have the same number of vectors
- Any linearly independent set can be extended to a basis
- Any spanning set can be reduced to a basis

## Null Space and Column Space

### Null Space (Kernel)
The null space (or kernel) of a matrix $`A`$ is the set of vectors $`x`$ such that $`Ax = 0`$. It is a subspace of $`\mathbb{R}^n`$ (where $`n`$ is the number of columns of $`A`$).

**Mathematical Definition:**
```math
\text{null}(A) = \{x \in \mathbb{R}^n : Ax = 0\}
```

**Properties:**
- Always contains the zero vector
- Dimension is called the **nullity** of $`A`$
- Related to the number of free variables in the system $`Ax = 0`$

**Why It Matters:**
- The null space describes all solutions to the homogeneous system $`Ax = 0`$
- In ML, the null space can indicate redundant features or dependencies
- The null space of $`A^T`$ is orthogonal to the column space of $`A`$

### Column Space (Range)
The column space (or range) of a matrix $`A`$ is the span of its columns. It is a subspace of $`\mathbb{R}^m`$ (where $`m`$ is the number of rows of $`A`$).

**Mathematical Definition:**
```math
\text{col}(A) = \text{span}\{a_1, a_2, \ldots, a_n\}
```
where $`a_1, a_2, \ldots, a_n`$ are the columns of $`A`$.

**Properties:**
- Dimension is called the **rank** of $`A`$
- The rank equals the number of linearly independent columns
- The rank equals the number of linearly independent rows

**Why It Matters:**
- The column space represents all possible outputs $`Ax`$
- The rank of $`A`$ is the dimension of the column space
- In ML, the column space relates to the set of all possible predictions
- The column space determines whether $`Ax = b`$ has a solution

## Rank-Nullity Theorem

For any matrix $`A`$ with $`n`$ columns:
```math
\text{rank}(A) + \text{nullity}(A) = n
```

- **Rank**: Dimension of the column space
- **Nullity**: Dimension of the null space

**Why It Matters:**
- The theorem relates the number of independent columns to the number of free variables in $`Ax = 0`$
- In ML, it helps diagnose redundancy and feature selection
- Provides insight into the structure of linear systems
- Helps understand the relationship between input and output dimensions

**Applications:**
- **Feature Selection**: If nullity is high, many features are redundant
- **Dimensionality Reduction**: Rank determines the effective dimension
- **System Analysis**: Rank determines if a system has unique solutions

## Applications in Machine Learning

### Feature Space
In machine learning, data points are represented as vectors in a feature space. The dimension of this space equals the number of features.

**Examples:**
- **Image Classification**: Each pixel is a feature, so a $`28 \times 28`$ grayscale image is a vector in $`\mathbb{R}^{784}`$
- **Text Classification**: Each word or token is a feature, creating high-dimensional sparse vectors
- **Time Series**: Each time point is a feature, creating vectors in $`\mathbb{R}^T`$ where $`T`$ is the sequence length

**Properties:**
- High-dimensional spaces suffer from the "curse of dimensionality"
- Feature selection reduces dimension while preserving information
- Dimensionality reduction techniques (PCA, t-SNE) find new bases

### Kernel Methods
Kernel methods implicitly work in high-dimensional feature spaces without explicitly computing the coordinates.

**Mathematical Foundation:**
- Input space: $`\mathcal{X} \subseteq \mathbb{R}^n`$
- Feature space: $`\mathcal{F}`$ (often infinite-dimensional)
- Kernel function: $`k(x, y) = \langle \phi(x), \phi(y) \rangle`$ where $`\phi: \mathcal{X} \to \mathcal{F}`$

**Examples:**
- **Polynomial Kernel**: $`k(x, y) = (x^T y + c)^d`$
- **RBF Kernel**: $`k(x, y) = \exp(-\gamma \|x - y\|^2)`$
- **Linear Kernel**: $`k(x, y) = x^T y`$

### Neural Networks
Neural networks can be viewed as compositions of linear transformations and nonlinear activations.

**Mathematical Structure:**
- Each layer performs: $`y = \sigma(Wx + b)`$
- The weight matrix $`W`$ defines a linear transformation
- The activation function $`\sigma`$ introduces nonlinearity
- The entire network maps from input space to output space

### Principal Component Analysis (PCA)
PCA finds a new basis for the data that maximizes variance along each direction.

**Mathematical Process:**
1. Center the data: $`X' = X - \bar{X}`$
2. Compute covariance matrix: $`C = \frac{1}{n-1} X'^T X'`$
3. Find eigenvectors: $`Cv_i = \lambda_i v_i`$
4. Project data: $`Y = X'V`$ where $`V`$ contains the top eigenvectors

**Properties:**
- The new basis vectors are orthogonal
- The eigenvalues indicate the variance explained by each component
- Dimensionality reduction by keeping only top components

## Exercises

### Exercise 1: Subspace Verification
Show that the set $`W = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\}`$ is a subspace of $`\mathbb{R}^3`$. Find its dimension and a basis.

### Exercise 2: Linear Independence
Determine whether the following sets of vectors are linearly independent:
1. $`\{(1, 2), (3, 4)\}`$ in $`\mathbb{R}^2`$
2. $`\{(1, 0, 1), (0, 1, 1), (1, 1, 0)\}`$ in $`\mathbb{R}^3`$
3. $`\{1, x, x^2\}`$ in the space of polynomials of degree $`\leq 2`$

### Exercise 3: Matrix Spaces
Consider the vector space of all $`2 \times 2`$ matrices. Find a basis for the subspace of symmetric matrices. What is its dimension?

### Exercise 4: Null Space and Column Space
For the matrix $`A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 1 & 3 & 5 \end{bmatrix}`$:
1. Find a basis for the null space
2. Find a basis for the column space
3. Verify the rank-nullity theorem

### Exercise 5: Function Spaces
Show that the set $`\{1, \sin(x), \cos(x)\}`$ is linearly independent in the space of continuous functions on $`[0, 2\pi]`$.

## Solutions

### Solution 1: Subspace Verification
To show $`W`$ is a subspace, we verify the three criteria:

1. **Zero vector**: $`(0, 0, 0) \in W`$ because $`0 + 0 + 0 = 0`$
2. **Closure under addition**: If $`(x_1, y_1, z_1), (x_2, y_2, z_2) \in W`$, then $`x_1 + y_1 + z_1 = 0`$ and $`x_2 + y_2 + z_2 = 0`$. So $`(x_1 + x_2) + (y_1 + y_2) + (z_1 + z_2) = 0`$, hence $`(x_1 + x_2, y_1 + y_2, z_1 + z_2) \in W`$
3. **Closure under scalar multiplication**: If $`(x, y, z) \in W`$ and $`c \in \mathbb{R}`$, then $`cx + cy + cz = c(x + y + z) = c \cdot 0 = 0`$, so $`(cx, cy, cz) \in W`$

**Dimension and Basis:**
The dimension is 2. A basis is $`\{(1, -1, 0), (1, 0, -1)\}`$.

### Solution 2: Linear Independence
1. **Independent**: The determinant $`\begin{vmatrix} 1 & 3 \\ 2 & 4 \end{vmatrix} = 4 - 6 = -2 \neq 0`$
2. **Independent**: The determinant $`\begin{vmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 0 \end{vmatrix} = 1(0-1) - 0(0-1) + 1(0-1) = -1 - 1 = -2 \neq 0`$
3. **Independent**: These are the standard basis for polynomials of degree $`\leq 2`$

### Solution 3: Matrix Spaces
A symmetric $`2 \times 2`$ matrix has the form $`\begin{bmatrix} a & b \\ b & c \end{bmatrix}`$.

A basis is:
$`\left\{\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}\right\}`$

The dimension is 3.

### Solution 4: Null Space and Column Space
1. **Null space basis**: $`\{(-1, -2, 1)\}`$
2. **Column space basis**: $`\{(1, 0, 1), (2, 1, 3)\}`$
3. **Verification**: Rank = 2, nullity = 1, so $`2 + 1 = 3`$ ✓

### Solution 5: Function Spaces
To show independence, assume $`c_1 \cdot 1 + c_2 \cdot \sin(x) + c_3 \cdot \cos(x) = 0`$ for all $`x \in [0, 2\pi]`$.

Evaluating at specific points:
- $`x = 0`$: $`c_1 + c_3 = 0`$
- $`x = \pi/2`$: $`c_1 + c_2 = 0`$
- $`x = \pi`$: $`c_1 - c_3 = 0`$

Solving gives $`c_1 = c_2 = c_3 = 0`$, so the set is independent.

## Summary

In this chapter, we covered:
- Definition and axioms of vector spaces
- Subspaces and their properties
- Span and linear independence
- Basis and dimension
- Null space and column space
- Rank-nullity theorem
- Applications in machine learning

Vector spaces provide the theoretical foundation for understanding linear algebra concepts and their applications in data science and machine learning. The concepts of subspaces, basis, and dimension are fundamental to understanding feature spaces, dimensionality reduction, and the structure of linear models.

## Next Steps

In the next chapter, we'll explore linear independence and basis in more detail, focusing on coordinate systems and change of basis. 