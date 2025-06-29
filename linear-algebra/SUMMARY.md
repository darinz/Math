# Summary and Practice Problems

[![Chapter](https://img.shields.io/badge/Chapter-10-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Summary_and_Practice_Problems-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Review-brightgreen.svg)]()

## Introduction

This final chapter provides a comprehensive summary of all linear algebra concepts covered in this guide, along with practice problems to reinforce understanding and prepare for real-world applications in AI/ML and data science.

## Summary of Key Concepts

### 1. Vectors and Vector Operations
- **Vectors**: Ordered lists of numbers representing magnitude and direction
- **Operations**: Addition, scalar multiplication, dot product, cross product
- **Properties**: Magnitude, unit vectors, angle between vectors
- **Applications**: Feature vectors, similarity calculations, data normalization

### 2. Matrices and Matrix Operations
- **Matrices**: 2D arrays representing linear transformations and data
- **Operations**: Addition, multiplication, transpose, inverse
- **Properties**: Determinant, trace, rank, eigenvalues
- **Applications**: Data matrices, covariance matrices, neural network weights

### 3. Linear Transformations
- **Definition**: Functions preserving vector addition and scalar multiplication
- **Types**: Scaling, rotation, reflection, shear
- **Composition**: Matrix multiplication for combining transformations
- **Applications**: Computer graphics, feature transformations, data augmentation

### 4. Eigenvalues and Eigenvectors
- **Definition**: Vectors that don't change direction under transformation
- **Properties**: Sum equals trace, product equals determinant
- **Diagonalization**: $A = PDP^{-1}$ for diagonalizable matrices
- **Applications**: PCA, spectral clustering, PageRank, neural network analysis

### 5. Vector Spaces and Subspaces
- **Vector Spaces**: Sets with addition and scalar multiplication operations
- **Subspaces**: Vector spaces contained within larger spaces
- **Basis**: Linearly independent spanning sets
- **Applications**: Feature spaces, dimensionality reduction, clustering

### 6. Linear Independence and Basis
- **Linear Independence**: No non-trivial linear combination equals zero
- **Basis**: Minimal spanning set for a vector space
- **Applications**: Feature selection, removing redundant data

### 7. Matrix Decompositions
- **LU**: $A = LU$ for solving systems
- **QR**: $A = QR$ for least squares and eigenvalues
- **SVD**: $A = U\Sigma V^T$ for dimensionality reduction
- **Cholesky**: $A = LL^T$ for positive definite matrices

### 8. Machine Learning Applications
- **Linear Regression**: Normal equations and gradient descent
- **PCA**: Dimensionality reduction using eigenvectors
- **Neural Networks**: Matrix operations in forward/backward passes
- **Recommender Systems**: Matrix factorization
- **SVM**: Linear classification with margins

### 9. Numerical Linear Algebra
- **Conditioning**: Sensitivity to perturbations
- **Stability**: Accuracy of numerical computations
- **Iterative Methods**: Jacobi, Gauss-Seidel, Conjugate Gradient
- **Sparse Matrices**: Efficient storage and computation

## Comprehensive Practice Problems

### Problem Set 1: Fundamentals

**Problem 1.1: Vector Operations**
```python
# Given vectors v1 = [1, 2, 3], v2 = [4, 5, 6], v3 = [7, 8, 9]
# Compute: v1 + v2, 2*v1, v1 · v2, |v1|, unit vector in direction of v1
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])

# Your code here
```

**Problem 1.2: Matrix Operations**
```python
# Given matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]
# Compute: A + B, A × B, A^T, det(A), A^(-1)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Your code here
```

**Problem 1.3: Linear Independence**
```python
# Check if the vectors [1, 2, 3], [4, 5, 6], [2, 4, 6] are linearly independent
vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([2, 4, 6])]

# Your code here
```

### Problem Set 2: Linear Transformations

**Problem 2.1: Rotation Matrix**
```python
# Create a rotation matrix that rotates by 45 degrees
# Apply it to the vector [1, 0] and verify the result

# Your code here
```

**Problem 2.2: Composition of Transformations**
```python
# Create matrices for: scale by (2, 1), rotate by 30 degrees
# Find the matrix that represents scale then rotate
# Apply this to the unit circle and visualize

# Your code here
```

**Problem 2.3: Eigenvalues and Eigenvectors**
```python
# Find eigenvalues and eigenvectors of A = [[3, 1], [1, 3]]
# Verify that Av = λv for each eigenvector
# Diagonalize the matrix

# Your code here
```

### Problem Set 3: Matrix Decompositions

**Problem 3.1: LU Decomposition**
```python
# Perform LU decomposition on A = [[2, 3, 1], [4, 7, 7], [6, 18, 22]]
# Use it to solve Ax = b where b = [1, 5, 0]

# Your code here
```

**Problem 3.2: QR Decomposition**
```python
# Perform QR decomposition on a random 3×3 matrix
# Verify that Q is orthogonal and R is upper triangular
# Use it to find the least squares solution

# Your code here
```

**Problem 3.3: SVD for Dimensionality Reduction**
```python
# Create a 10×5 data matrix with some structure
# Perform SVD and keep only the top 2 singular values
# Compare the original and reconstructed matrices

# Your code here
```

### Problem Set 4: Machine Learning Applications

**Problem 4.1: Linear Regression from Scratch**
```python
# Generate synthetic data: y = 2*x1 + 3*x2 + noise
# Implement linear regression using normal equations
# Compare with gradient descent implementation

# Your code here
```

**Problem 4.2: PCA Implementation**
```python
# Load the iris dataset
# Implement PCA from scratch using eigenvalue decomposition
# Compare with sklearn's PCA
# Visualize the first two principal components

# Your code here
```

**Problem 4.3: Neural Network for XOR**
```python
# Implement a neural network with one hidden layer
# Train it to solve the XOR problem
# Visualize the decision boundary

# Your code here
```

### Problem Set 5: Numerical Methods

**Problem 5.1: Condition Number Analysis**
```python
# Create Hilbert matrices of different sizes
# Compute their condition numbers
# Solve Hx = b and analyze sensitivity to perturbations

# Your code here
```

**Problem 5.2: Iterative Methods Comparison**
```python
# Implement Jacobi, Gauss-Seidel, and Conjugate Gradient
# Compare convergence rates for a tridiagonal system
# Plot convergence history

# Your code here
```

**Problem 5.3: Sparse Matrix Operations**
```python
# Create a large sparse matrix (e.g., tridiagonal)
# Compare dense vs sparse storage and computation
# Measure performance differences

# Your code here
```

## Advanced Problems

### Problem A.1: Recommender System
```python
# Create a user-item rating matrix
# Implement matrix factorization using gradient descent
# Evaluate the model using cross-validation

def matrix_factorization_recommender():
    # Generate synthetic rating data
    n_users, n_items = 100, 50
    R = np.random.randint(1, 6, (n_users, n_items))
    R[R < 3] = 0  # Some missing ratings
    
    # Your implementation here
    pass
```

### Problem A.2: Image Compression with SVD
```python
# Load a grayscale image
# Compress it using different numbers of singular values
# Compare compression ratios and reconstruction quality

def image_compression_svd():
    # Your implementation here
    pass
```

### Problem A.3: Spectral Clustering
```python
# Implement spectral clustering using Laplacian matrix
# Apply to a dataset with multiple clusters
# Compare with k-means clustering

def spectral_clustering():
    # Your implementation here
    pass
```

### Problem A.4: PageRank Algorithm
```python
# Create a web graph as adjacency matrix
# Implement PageRank using power iteration
# Rank the pages by importance

def pagerank_algorithm():
    # Your implementation here
    pass
```

### Problem A.5: Support Vector Machine
```python
# Implement linear SVM using gradient descent
# Create a dataset with two classes
# Find the optimal hyperplane and margin

def linear_svm():
    # Your implementation here
    pass
```

## Solutions to Selected Problems

### Solution 1.1: Vector Operations
```python
# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition
v_sum = v1 + v2
print("v1 + v2 =", v_sum)

# Scalar multiplication
v_scaled = 2 * v1
print("2 * v1 =", v_scaled)

# Dot product
dot_product = np.dot(v1, v2)
print("v1 · v2 =", dot_product)

# Magnitude
magnitude = np.linalg.norm(v1)
print("|v1| =", magnitude)

# Unit vector
unit_v1 = v1 / magnitude
print("Unit vector =", unit_v1)
```

### Solution 2.1: Rotation Matrix
```python
def rotation_matrix(angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return np.array([[cos_theta, -sin_theta], 
                     [sin_theta, cos_theta]])

# Create rotation matrix
R = rotation_matrix(45)
print("Rotation matrix (45°):")
print(R)

# Apply to vector [1, 0]
v = np.array([1, 0])
v_rotated = R @ v
print("Rotated vector:", v_rotated)

# Verify: should be [cos(45°), sin(45°)] ≈ [0.707, 0.707]
expected = np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])
print("Expected:", expected)
print("Error:", np.linalg.norm(v_rotated - expected))
```

### Solution 3.1: LU Decomposition
```python
import scipy.linalg

A = np.array([[2, 3, 1], [4, 7, 7], [6, 18, 22]])
b = np.array([1, 5, 0])

# LU decomposition
P, L, U = scipy.linalg.lu(A)
print("L:")
print(L)
print("\nU:")
print(U)
print("\nP:")
print(P)

# Solve using LU decomposition
# PA = LU, so Ax = b becomes PAx = Pb, or LUx = Pb
# Let y = Ux, then Ly = Pb
# Solve Ly = Pb for y, then Ux = y for x

Pb = P @ b
y = scipy.linalg.solve_triangular(L, Pb, lower=True)
x = scipy.linalg.solve_triangular(U, y, lower=False)

print("\nSolution x:", x)

# Verify
print("Ax =", A @ x)
print("b =", b)
print("Error:", np.linalg.norm(A @ x - b))
```

### Solution 4.1: Linear Regression
```python
# Generate synthetic data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
true_beta = np.array([2, 3])
y = X @ true_beta + np.random.normal(0, 0.1, n_samples)

# Normal equations
X_with_bias = np.column_stack([np.ones(n_samples), X])
beta_normal = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
print("Normal equations solution:", beta_normal)

# Gradient descent
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples = X.shape[0]
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(X_with_bias.shape[1])
    
    for epoch in range(epochs):
        predictions = X_with_bias @ theta
        gradients = (2/n_samples) * X_with_bias.T @ (predictions - y)
        theta -= learning_rate * gradients
    
    return theta

beta_gd = gradient_descent(X, y)
print("Gradient descent solution:", beta_gd)

# Compare with true parameters
print("True parameters:", np.concatenate([[0], true_beta]))
```

## Final Project: Complete ML Pipeline

### Project: Face Recognition System

Build a complete face recognition system using linear algebra concepts:

1. **Data Preprocessing**: Use PCA for dimensionality reduction
2. **Feature Extraction**: Implement SVD for feature learning
3. **Classification**: Use linear SVM for face recognition
4. **Evaluation**: Measure accuracy and performance

```python
def face_recognition_pipeline():
    """
    Complete face recognition pipeline using linear algebra
    """
    # This is a framework - you would need actual face data
    # and more sophisticated implementations
    
    # 1. Load and preprocess face images
    # 2. Apply PCA for dimensionality reduction
    # 3. Use SVD for feature extraction
    # 4. Train linear SVM classifier
    # 5. Evaluate performance
    
    pass
```

## Key Takeaways and Next Steps

### What You've Learned
- **Mathematical Foundation**: Solid understanding of linear algebra concepts
- **Practical Implementation**: Python code for all major algorithms
- **ML Applications**: Real-world applications in machine learning
- **Numerical Methods**: Efficient algorithms for large-scale problems

### Skills Developed
- Matrix operations and decompositions
- Eigenvalue problems and diagonalization
- Linear transformations and their applications
- Numerical stability and conditioning
- Implementation of ML algorithms from scratch

### Next Steps
1. **Practice**: Work through all problems in this chapter
2. **Explore**: Study advanced topics like tensor decompositions
3. **Apply**: Use these concepts in your ML projects
4. **Extend**: Learn about optimization and deep learning
5. **Contribute**: Share your implementations and improvements

### Resources for Further Learning
- **Books**: "Linear Algebra Done Right" by Sheldon Axler
- **Courses**: MIT 18.06 Linear Algebra, Stanford CS229
- **Libraries**: NumPy, SciPy, scikit-learn documentation
- **Research Papers**: Recent papers on matrix factorizations and neural networks

## Conclusion

This comprehensive guide has covered the essential linear algebra concepts needed for AI/ML and data science. The combination of theoretical understanding and practical implementation will serve as a strong foundation for your work in machine learning, computer vision, natural language processing, and other AI applications.

Remember that linear algebra is not just a mathematical tool—it's the language of machine learning. Mastery of these concepts will enable you to understand, implement, and innovate in the field of artificial intelligence.

Happy learning and coding! 