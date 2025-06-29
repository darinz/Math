# Applications in Machine Learning

[![Chapter](https://img.shields.io/badge/Chapter-8-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-ML_Applications-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Advanced-red.svg)]()

## Introduction

Linear algebra is the mathematical foundation of machine learning. This chapter explores how linear algebra concepts are applied in various ML algorithms, from simple linear regression to complex neural networks.

## 1. Linear Regression

Linear regression models the relationship between features and target as a linear combination: $y = X\beta + \epsilon$.

### Normal Equation Solution
$$\beta = (X^T X)^{-1} X^T y$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 100, 3
X = np.random.randn(n_samples, n_features)
true_beta = np.array([2.5, -1.0, 0.8])
y = X @ true_beta + np.random.normal(0, 0.1, n_samples)

# Normal equation solution
X_with_bias = np.column_stack([np.ones(n_samples), X])
beta_normal = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
print("Normal equation solution:", beta_normal)

# Using sklearn
lr = LinearRegression()
lr.fit(X, y)
print("Sklearn solution:", np.concatenate([[lr.intercept_], lr.coef_]))

# Predictions
y_pred_normal = X_with_bias @ beta_normal
y_pred_sklearn = lr.predict(X)

print("MSE (Normal):", mean_squared_error(y, y_pred_normal))
print("MSE (Sklearn):", mean_squared_error(y, y_pred_sklearn))
```

### Ridge Regression (L2 Regularization)
$$\beta = (X^T X + \lambda I)^{-1} X^T y$$

```python
def ridge_regression(X, y, lambda_reg=1.0):
    n_features = X.shape[1]
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    I = np.eye(X_with_bias.shape[1])
    I[0, 0] = 0  # Don't regularize bias term
    
    beta = np.linalg.inv(X_with_bias.T @ X_with_bias + lambda_reg * I) @ X_with_bias.T @ y
    return beta

# Test ridge regression
beta_ridge = ridge_regression(X, y, lambda_reg=0.1)
print("Ridge solution:", beta_ridge)
```

## 2. Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in data by computing eigenvectors of the covariance matrix.

### Manual PCA Implementation
```python
def manual_pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    sorted_indices = np.argsort(eigenvals)[::-1]
    eigenvals_sorted = eigenvals[sorted_indices]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    
    # Project data
    X_pca = X_centered @ eigenvecs_sorted[:, :n_components]
    
    return X_pca, eigenvecs_sorted[:, :n_components], eigenvals_sorted

# Generate data with known structure
np.random.seed(42)
n_samples = 1000
# Create correlated features
X = np.random.randn(n_samples, 3)
X[:, 2] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n_samples)

# Apply PCA
X_pca, components, eigenvals = manual_pca(X, n_components=2)

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data (Features 1 vs 2)')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 2], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 3')
plt.title('Original Data (Features 1 vs 3)')

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Data')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", eigenvals[:2] / np.sum(eigenvals))
```

### PCA for Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
for i in range(3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f'Class {i}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs Components')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 3. Recommender Systems

### Matrix Factorization
```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    R: rating matrix
    P: user matrix
    Q: item matrix
    K: number of latent factors
    """
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:], Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T

# Example: Simple rating matrix
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

print("Original ratings:")
print(R)
print("\nPredicted ratings:")
print(nR)
```

## 4. Neural Networks

### Forward Pass
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        
        # Backpropagation
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train neural network
nn = SimpleNeuralNetwork(2, 4, 1)

for epoch in range(10000):
    # Forward pass
    output = nn.forward(X)
    
    # Backward pass
    nn.backward(X, y, learning_rate=0.1)
    
    if epoch % 1000 == 0:
        loss = np.mean((output - y) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
predictions = nn.forward(X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
```

## 5. Support Vector Machines (SVM)

### Linear SVM with Gradient Descent
```python
def linear_svm_gradient_descent(X, y, learning_rate=0.01, epochs=1000, lambda_reg=0.1):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        for i in range(n_samples):
            # Hinge loss gradient
            if y[i] * (X[i] @ w + b) < 1:
                dw = -y[i] * X[i] + lambda_reg * w
                db = -y[i]
            else:
                dw = lambda_reg * w
                db = 0
            
            w -= learning_rate * dw
            b -= learning_rate * db
    
    return w, b

# Generate linearly separable data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Train SVM
w, b = linear_svm_gradient_descent(X, y)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', label='Class -1')

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = np.sign(xx * w[0] + yy * w[1] + b)
plt.contour(xx, yy, Z, levels=[0], colors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear SVM Decision Boundary')
plt.legend()

plt.subplot(1, 2, 2)
# Show margin
margin_points = np.array([[-2, 2], [2, -2]])
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.6)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', alpha=0.6)
plt.plot(margin_points[:, 0], margin_points[:, 1], 'k--', alpha=0.5, label='Margin')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Margin')
plt.legend()

plt.tight_layout()
plt.show()
```

## 6. Clustering with Linear Algebra

### K-Means Clustering
```python
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Generate clustered data
np.random.seed(42)
n_samples = 300
centers = np.array([[0, 0], [4, 4], [8, 0]])
X = np.vstack([
    np.random.randn(n_samples//3, 2) + centers[0],
    np.random.randn(n_samples//3, 2) + centers[1],
    np.random.randn(n_samples//3, 2) + centers[2]
])

# Apply K-means
labels, centroids = kmeans(X, k=3)

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Original Data')

plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], alpha=0.6, label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering')
plt.legend()

plt.tight_layout()
plt.show()
```

## 7. Optimization in Machine Learning

### Gradient Descent for Linear Regression
```python
def gradient_descent_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(X_with_bias.shape[1])
    
    costs = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = X_with_bias @ theta
        
        # Compute cost
        cost = np.mean((predictions - y) ** 2)
        costs.append(cost)
        
        # Compute gradients
        gradients = (2/n_samples) * X_with_bias.T @ (predictions - y)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta, costs

# Test gradient descent
theta_gd, costs = gradient_descent_linear_regression(X, y)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], y, alpha=0.6, label='Data')
X_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
y_line = np.column_stack([np.ones(100), X_line]) @ theta_gd
plt.plot(X_line, y_line, 'r-', label='Gradient Descent')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.legend()

plt.tight_layout()
plt.show()
```

## Exercises

1. **Linear Regression**: Implement linear regression using both normal equations and gradient descent.
2. **PCA Implementation**: Implement PCA from scratch and compare with sklearn.
3. **Neural Network**: Build a neural network to solve the XOR problem.
4. **SVM**: Implement linear SVM using gradient descent.
5. **Matrix Factorization**: Implement matrix factorization for a simple recommender system.

## Solutions

```python
# Exercise 1: Linear Regression
def linear_regression_normal_equations(X, y):
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n_samples = X.shape[0]
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(X_with_bias.shape[1])
    
    for _ in range(epochs):
        predictions = X_with_bias @ theta
        gradients = (2/n_samples) * X_with_bias.T @ (predictions - y)
        theta -= learning_rate * gradients
    
    return theta

# Exercise 2: PCA Implementation
def pca_manual(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvals)[::-1]
    eigenvecs_sorted = eigenvecs[:, sorted_indices]
    return X_centered @ eigenvecs_sorted[:, :n_components]

# Exercise 3: Neural Network for XOR
def train_xor_network():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    nn = SimpleNeuralNetwork(2, 4, 1)
    for _ in range(10000):
        output = nn.forward(X)
        nn.backward(X, y, learning_rate=0.1)
    
    return nn

# Exercise 4: Linear SVM
def linear_svm(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(epochs):
        for i in range(n_samples):
            if y[i] * (X[i] @ w + b) < 1:
                w -= learning_rate * (-y[i] * X[i] + 0.1 * w)
                b -= learning_rate * (-y[i])
            else:
                w -= learning_rate * (0.1 * w)
    
    return w, b

# Exercise 5: Matrix Factorization
def simple_matrix_factorization(R, k=2, epochs=1000, learning_rate=0.01):
    n_users, n_items = R.shape
    P = np.random.randn(n_users, k)
    Q = np.random.randn(n_items, k)
    
    for _ in range(epochs):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:
                    eij = R[i, j] - P[i, :] @ Q[j, :]
                    P[i, :] += learning_rate * eij * Q[j, :]
                    Q[j, :] += learning_rate * eij * P[i, :]
    
    return P, Q
```

## Key Takeaways

- Linear algebra is fundamental to all machine learning algorithms
- Matrix operations enable efficient computation of gradients and predictions
- Eigenvalue decomposition (PCA) is crucial for dimensionality reduction
- Matrix factorization enables collaborative filtering in recommender systems
- Neural networks rely heavily on matrix multiplication for forward and backward passes
- Optimization algorithms use linear algebra for parameter updates

## Next Chapter

In the next chapter, we'll explore numerical linear algebra, focusing on numerical stability, conditioning, and efficient algorithms for large-scale problems. 