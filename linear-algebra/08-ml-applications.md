# Applications in Machine Learning

## Introduction

Linear algebra is the mathematical foundation of machine learning, providing the theoretical framework and computational tools that enable modern AI systems. This chapter explores how fundamental linear algebra concepts are applied in various machine learning algorithms, from simple linear regression to complex neural networks and deep learning systems.

**Mathematical Foundation:**
Machine learning algorithms can be viewed as optimization problems in high-dimensional vector spaces:
- **Feature Space**: Data points are vectors in $`\mathbb{R}^n`$
- **Parameter Space**: Model parameters are vectors in $`\mathbb{R}^p`$
- **Loss Functions**: Scalar functions mapping parameter vectors to real numbers
- **Gradients**: Vector derivatives indicating direction of steepest descent

**Key Linear Algebra Concepts in ML:**
1. **Vector Operations**: Dot products, norms, and projections for similarity and distance
2. **Matrix Operations**: Transformations, decompositions, and eigenvalue problems
3. **Optimization**: Gradient descent, Hessian matrices, and convex optimization
4. **Dimensionality Reduction**: Principal component analysis and matrix factorizations
5. **Kernel Methods**: Inner products and feature space transformations

**Geometric Interpretation:**
Machine learning can be understood geometrically:
- **Linear Models**: Find hyperplanes that best separate or fit data
- **Nonlinear Models**: Transform data into higher-dimensional spaces where linear separation is possible
- **Clustering**: Find centroids that minimize distances to data points
- **Dimensionality Reduction**: Project data onto lower-dimensional subspaces while preserving structure

## 1. Linear Regression

Linear regression is the foundation of supervised learning, modeling the relationship between features and target as a linear combination with additive noise.

### Mathematical Foundation

**Model Definition:**
For input features $`x \in \mathbb{R}^n`$ and target $`y \in \mathbb{R}`$, the linear regression model is:
```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon = x^T \beta + \varepsilon
```

where:
- $`\beta = [\beta_0, \beta_1, \ldots, \beta_n]^T`$ is the parameter vector (including bias term)
- $`x = [1, x_1, x_2, \ldots, x_n]^T`$ is the feature vector (with bias term)
- $`\varepsilon \sim \mathcal{N}(0, \sigma^2)`$ is the noise term

**Matrix Formulation:**
For a dataset with $`m`$ samples, we have:
```math
Y = X\beta + \varepsilon
```

where:
- $`Y \in \mathbb{R}^m`$ is the target vector
- $`X \in \mathbb{R}^{m \times (n+1)}`$ is the design matrix (with bias column)
- $`\beta \in \mathbb{R}^{n+1}`$ is the parameter vector
- $`\varepsilon \in \mathbb{R}^m`$ is the noise vector

**Objective Function:**
The least squares objective is:
```math
\min \|X\beta - Y\|_2^2 = \min \sum_{i=1}^m (x_i^T \beta - y_i)^2
```

**Geometric Interpretation:**
Linear regression finds the projection of $`Y`$ onto the column space of $`X`$, minimizing the orthogonal distance between $`Y`$ and the subspace spanned by $`X`$'s columns.

### Normal Equation Solution

**Mathematical Derivation:**
The normal equation is derived by setting the gradient of the objective function to zero:

```math
\nabla_\beta(\|X\beta - Y\|_2^2) = 2X^T(X\beta - Y) = 0
```

Solving for $`\beta`$:
```math
X^T X \beta = X^T Y
\beta = (X^T X)^{-1} X^T Y
```

**Properties:**
1. **Uniqueness**: Solution is unique if $`X`$ has full column rank
2. **Optimality**: Global minimum of the convex objective function
3. **Computational Cost**: $`O(n^3)`$ for matrix inversion, $`O(n^2)`$ for matrix multiplication
4. **Numerical Stability**: Can be ill-conditioned for nearly singular $`X^T X`$

**Algorithm:**
1. **Compute**: $`A = X^T X`$ and $`b = X^T Y`$
2. **Solve**: $`A \beta = b`$ using Cholesky decomposition or QR decomposition
3. **Predict**: $`\hat{y} = X \beta`$

### Ridge Regression (L2 Regularization)

**Mathematical Foundation:**
Ridge regression adds L2 regularization to prevent overfitting and handle multicollinearity:

**Objective Function:**
```math
\min \|X\beta - Y\|_2^2 + \lambda \|\beta\|_2^2
```

where $`\lambda > 0`$ is the regularization parameter.

**Solution:**
```math
\beta = (X^T X + \lambda I)^{-1} X^T Y
```

**Geometric Interpretation:**
Ridge regression shrinks parameter estimates toward zero, trading bias for variance reduction.

**Key Properties:**
1. **Bias-Variance Trade-off**: Increases bias, decreases variance
2. **Multicollinearity**: Handles correlated features effectively
3. **Numerical Stability**: Improves conditioning of $`X^T X`$
4. **Shrinkage**: All parameters are shrunk by the same factor

**Cross-Validation:**
- Use k-fold cross-validation to select optimal $`\lambda`$
- Grid search over $`\lambda \in [10^{-4}, 10^4]`$
- Monitor validation MSE to prevent overfitting

### Gradient Descent Implementation

**Mathematical Foundation:**
Gradient descent updates parameters iteratively:

```math
\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla_\beta L(\beta^{(t)})
```

where $`L(\beta) = \frac{1}{2m} \|X\beta - Y\|_2^2`$ is the loss function.

**Gradient Computation:**
```math
\nabla_\beta L(\beta) = \frac{1}{m} X^T(X\beta - Y)
```

**Algorithm:**
1. **Initialize**: $`\beta^{(0)} = 0`$
2. **For each iteration** $`t = 0, 1, \ldots, T-1`$:
   - Compute gradient: $`g = \frac{1}{m} X^T(X\beta^{(t)} - Y)`$
   - Update parameters: $`\beta^{(t+1)} = \beta^{(t)} - \alpha g`$
3. **Return**: $`\beta^{(T)}`$

**Learning Rate Selection:**
- Start with $`\alpha = 0.01`$
- Use line search or adaptive methods
- Monitor convergence: $`\|\beta^{(t+1)} - \beta^{(t)}\|_2 < \epsilon`$

## 2. Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in data by computing eigenvectors of the covariance matrix.

### Mathematical Foundation

**Covariance Matrix:**
For centered data $`X' = X - \bar{X}`$, the covariance matrix is:
```math
C = \frac{1}{n-1} X'^T X'
```

**Eigenvalue Decomposition:**
```math
C = V \Lambda V^T
```

where $`V`$ contains eigenvectors and $`\Lambda`$ contains eigenvalues.

**Principal Components:**
The principal components are the eigenvectors $`v_1, v_2, \ldots, v_d`$ ordered by decreasing eigenvalues $`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d`$.

**Projection:**
Data projected onto the first $`k`$ principal components:
```math
Y = X' V_k
```

where $`V_k`$ contains the first $`k`$ eigenvectors.

### Manual PCA Implementation

**Algorithm:**
1. **Center Data**: $`X' = X - \bar{X}`$
2. **Compute Covariance**: $`C = \frac{1}{n-1} X'^T X'`$
3. **Eigenvalue Decomposition**: $`C = V \Lambda V^T`$
4. **Sort**: Order eigenvectors by decreasing eigenvalues
5. **Project**: $`Y = X' V_k`$

**Properties:**
- **Variance Explained**: $`\frac{\lambda_i}{\sum_{j=1}^d \lambda_j}`$ for component $`i`$
- **Orthogonality**: Principal components are orthogonal
- **Optimality**: First $`k`$ components capture maximum variance

### PCA for Dimensionality Reduction

**Mathematical Foundation:**
PCA provides the optimal linear dimensionality reduction in terms of mean squared error.

**Reconstruction Error:**
```math
\|X' - X' V_k V_k^T\|_F^2 = \sum_{i=k+1}^d \lambda_i
```

**Selection of k:**
1. **Variance Threshold**: Choose $`k`$ such that $`\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i \geq 0.95`$
2. **Elbow Method**: Plot eigenvalues and find "elbow"
3. **Cross-Validation**: Use reconstruction error on validation set

**Applications:**
- **Data Visualization**: Project to 2D/3D for plotting
- **Feature Engineering**: Create new features from principal components
- **Noise Reduction**: Remove components with small eigenvalues
- **Compression**: Store only top-$`k`$ components

## 3. Recommender Systems

### Matrix Factorization

**Mathematical Foundation:**
Matrix factorization models user-item ratings as:
```math
R \approx U V^T
```

where:
- $`R \in \mathbb{R}^{m \times n}`$ is the rating matrix
- $`U \in \mathbb{R}^{m \times k}`$ contains user embeddings
- $`V \in \mathbb{R}^{n \times k}`$ contains item embeddings

**Objective Function:**
```math
\min_{U,V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i^T V_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)
```

where $`\Omega`$ is the set of observed ratings.

**Alternating Least Squares (ALS):**
1. **Fix V, solve for U**:
   ```math
   U_i = (V_{\Omega_i}^T V_{\Omega_i} + \lambda I)^{-1} V_{\Omega_i}^T R_{i,\Omega_i}
   ```
2. **Fix U, solve for V**:
   ```math
   V_j = (U_{\Omega_j}^T U_{\Omega_j} + \lambda I)^{-1} U_{\Omega_j}^T R_{\Omega_j,j}
   ```

**Stochastic Gradient Descent (SGD):**
For each observed rating $`(i, j, r_{ij})`$:
```math
\begin{align}
U_i &\leftarrow U_i + \alpha(e_{ij} V_j - \lambda U_i) \\
V_j &\leftarrow V_j + \alpha(e_{ij} U_i - \lambda V_j)
\end{align}
```

where $`e_{ij} = r_{ij} - U_i^T V_j`$ is the prediction error.

## 4. Neural Networks

### Forward Pass

**Mathematical Foundation:**
A neural network with $`L`$ layers computes:
```math
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})
```

where:
- $`a^{(l)}`$ is the activation at layer $`l`$
- $`W^{(l)}`$ is the weight matrix
- $`b^{(l)}`$ is the bias vector
- $`\sigma`$ is the activation function

**Matrix Formulation:**
For batch of $`m`$ samples:
```math
A^{(l)} = \sigma(W^{(l)} A^{(l-1)} + b^{(l)})
```

where $`A^{(l)} \in \mathbb{R}^{n_l \times m}`$ contains activations for all samples.

**Backpropagation:**
The gradient with respect to weights is:
```math
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
```

where $`\delta^{(l)}`$ is the error term at layer $`l`$.

### Backpropagation Algorithm

**Mathematical Foundation:**
Backpropagation computes gradients efficiently using the chain rule.

**Forward Pass:**
1. **Initialize**: $`a^{(0)} = x`$
2. **For each layer** $`l = 1, 2, \ldots, L`$:
   - $`z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}`$
   - $`a^{(l)} = \sigma(z^{(l)})`$

**Backward Pass:**
1. **Initialize**: $`\delta^{(L)} = \nabla_{a^{(L)}} L`$
2. **For each layer** $`l = L-1, L-2, \ldots, 1`$:
   - $`\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})`$
3. **Gradients**:
   - $`\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T`$
   - $`\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}`$

**Activation Functions:**
- **ReLU**: $`\sigma(x) = \max(0, x)`$, $`\sigma'(x) = \mathbb{I}(x > 0)`$
- **Sigmoid**: $`\sigma(x) = \frac{1}{1 + e^{-x}}`$, $`\sigma'(x) = \sigma(x)(1 - \sigma(x))`$
- **Tanh**: $`\sigma(x) = \tanh(x)`$, $`\sigma'(x) = 1 - \tanh^2(x)`$

## 5. Support Vector Machines (SVM)

### Linear SVM with Gradient Descent

**Mathematical Foundation:**
Linear SVM finds the hyperplane that maximizes the margin between classes.

**Primal Formulation:**
```math
\min_{w, b} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))
```

where $`C > 0`$ controls the trade-off between margin and misclassification.

**Hinge Loss:**
```math
L(w, b) = \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))
```

**Gradient:**
```math
\nabla_w L = w - C \sum_{i \in S} y_i x_i
```

where $`S = \{i : y_i(w^T x_i + b) < 1\}`$ is the set of support vectors.

**Algorithm:**
1. **Initialize**: $`w = 0`$, $`b = 0`$
2. **For each iteration**:
   - Compute gradient: $`g_w = w - C \sum_{i \in S} y_i x_i`$
   - Update: $`w \leftarrow w - \alpha g_w`$
   - Update bias: $`b \leftarrow b + \alpha C \sum_{i \in S} y_i`$

### Kernel SVM

**Mathematical Foundation:**
Kernel SVM maps data to higher-dimensional space using kernel functions.

**Kernel Trick:**
```math
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
```

**Common Kernels:**
- **Linear**: $`K(x_i, x_j) = x_i^T x_j`$
- **Polynomial**: $`K(x_i, x_j) = (x_i^T x_j + c)^d`$
- **RBF**: $`K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)`$

**Dual Formulation:**
```math
\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)
```

subject to $`0 \leq \alpha_i \leq C`$ and $`\sum_{i=1}^n \alpha_i y_i = 0`$.

## 6. Clustering with Linear Algebra

### K-Means Clustering

**Mathematical Foundation:**
K-means minimizes the sum of squared distances to cluster centroids.

**Objective Function:**
```math
\min_{\{S_k\}, \{\mu_k\}} \sum_{k=1}^K \sum_{x_i \in S_k} \|x_i - \mu_k\|_2^2
```

where $`S_k`$ is the set of points in cluster $`k`$ and $`\mu_k`$ is the centroid.

**Algorithm:**
1. **Initialize**: Random centroids $`\mu_1, \mu_2, \ldots, \mu_K`$
2. **Assignment**: $`S_k = \{x_i : k = \arg\min_j \|x_i - \mu_j\|_2\}`$
3. **Update**: $`\mu_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i`$
4. **Repeat** until convergence

**Matrix Formulation:**
Let $`Z \in \{0,1\}^{n \times K}`$ be the assignment matrix where $`Z_{ik} = 1`$ if point $`i`$ belongs to cluster $`k`$.

**Centroids:**
```math
\mu_k = \frac{\sum_{i=1}^n Z_{ik} x_i}{\sum_{i=1}^n Z_{ik}}
```

**Distance Matrix:**
```math
D_{ik} = \|x_i - \mu_k\|_2^2
```

### Spectral Clustering

**Mathematical Foundation:**
Spectral clustering uses eigenvectors of the Laplacian matrix.

**Similarity Matrix:**
```math
W_{ij} = \exp\left(-\frac{\|x_i - x_j\|_2^2}{2\sigma^2}\right)
```

**Laplacian Matrix:**
```math
L = D - W
```

where $`D_{ii} = \sum_{j=1}^n W_{ij}`$ is the degree matrix.

**Algorithm:**
1. **Compute**: Similarity matrix $`W`$
2. **Construct**: Laplacian $`L = D - W`$
3. **Eigenvalue Decomposition**: $`L = V \Lambda V^T`$
4. **Cluster**: Apply k-means to rows of $`V_k`$ (top $`k`$ eigenvectors)

## 7. Optimization in Machine Learning

### Gradient Descent for Linear Regression

**Mathematical Foundation:**
Gradient descent minimizes the loss function iteratively.

**Loss Function:**
```math
L(\beta) = \frac{1}{2m} \|X\beta - y\|_2^2
```

**Gradient:**
```math
\nabla_\beta L(\beta) = \frac{1}{m} X^T(X\beta - y)
```

**Update Rule:**
```math
\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla_\beta L(\beta^{(t)})
```

**Convergence:**
- **Lipschitz Continuity**: $`\|\nabla L(\beta_1) - \nabla L(\beta_2)\|_2 \leq L \|\beta_1 - \beta_2\|_2`$
- **Convergence Rate**: $`L(\beta^{(t)}) - L(\beta^*) \leq \frac{L}{2t} \|\beta^{(0)} - \beta^*\|_2^2`$

### Stochastic Gradient Descent (SGD)

**Mathematical Foundation:**
SGD uses random subsets of data to estimate gradients.

**Mini-batch Gradient:**
```math
\nabla_\beta L(\beta) \approx \frac{1}{|B|} \sum_{i \in B} \nabla_\beta L_i(\beta)
```

where $`B`$ is a mini-batch of samples.

**Update Rule:**
```math
\beta^{(t+1)} = \beta^{(t)} - \alpha_t \frac{1}{|B|} \sum_{i \in B} \nabla_\beta L_i(\beta^{(t)})
```

**Learning Rate Scheduling:**
- **Constant**: $`\alpha_t = \alpha_0`$
- **Decay**: $`\alpha_t = \alpha_0 / (1 + \gamma t)`$
- **Adaptive**: Adam, RMSprop, AdaGrad

### Second-Order Methods

**Newton's Method:**
```math
\beta^{(t+1)} = \beta^{(t)} - H^{-1} \nabla_\beta L(\beta^{(t)})
```

where $`H = \nabla^2_\beta L(\beta)`$ is the Hessian matrix.

**Quasi-Newton Methods:**
- **BFGS**: Approximates Hessian inverse using gradient differences
- **L-BFGS**: Limited memory version for large-scale problems

## Exercises

### Exercise 1: Linear Regression Implementation

**Objective**: Implement linear regression using both normal equations and gradient descent.

**Tasks:**
1. Generate synthetic data: $`y = 2x_1 + 3x_2 + 1 + \varepsilon`$ where $`\varepsilon \sim \mathcal{N}(0, 0.1)`$
2. Implement normal equation solution
3. Implement gradient descent with learning rate $`\alpha = 0.01`$
4. Compare solutions and convergence rates
5. Add ridge regularization and tune hyperparameter $`\lambda`$

### Exercise 2: PCA Implementation

**Objective**: Implement PCA from scratch and compare with sklearn.

**Tasks:**
1. Generate 3D data with known structure
2. Implement PCA using eigenvalue decomposition
3. Implement PCA using SVD
4. Compare with sklearn.decomposition.PCA
5. Analyze explained variance ratio
6. Visualize data in principal component space

### Exercise 3: Neural Network for XOR

**Objective**: Build a neural network to solve the XOR problem.

**Tasks:**
1. Create XOR dataset: $`\{(0,0,0), (0,1,1), (1,0,1), (1,1,0)\}`$
2. Implement 2-layer neural network with sigmoid activation
3. Train using backpropagation
4. Visualize decision boundary
5. Analyze weight matrices and hidden layer representations

### Exercise 4: Linear SVM Implementation

**Objective**: Implement linear SVM using gradient descent.

**Tasks:**
1. Generate linearly separable 2D data
2. Implement SVM with hinge loss
3. Train using gradient descent
4. Visualize decision boundary and support vectors
5. Compare with sklearn.svm.SVC

### Exercise 5: Matrix Factorization for Recommender Systems

**Objective**: Implement matrix factorization for a simple recommender system.

**Tasks:**
1. Create synthetic rating matrix with known structure
2. Implement alternating least squares (ALS)
3. Implement stochastic gradient descent (SGD)
4. Compare convergence rates and final RMSE
5. Analyze user and item embeddings

### Exercise 6: K-Means Clustering

**Objective**: Implement k-means clustering and analyze convergence.

**Tasks:**
1. Generate 3 clusters of 2D data
2. Implement k-means algorithm
3. Analyze convergence and final objective value
4. Compare with sklearn.cluster.KMeans
5. Visualize clustering results and centroids

## Solutions

### Solution 1: Linear Regression Implementation

**Step 1: Data Generation**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
true_beta = np.array([2, 3, 1])  # [beta_1, beta_2, beta_0]
X_with_bias = np.column_stack([X, np.ones(n_samples)])
y = X_with_bias @ true_beta + np.random.normal(0, 0.1, n_samples)
```

**Step 2: Normal Equation Solution**
```python
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

beta_normal = normal_equation(X_with_bias, y)
print("Normal equation solution:", beta_normal)
```

**Step 3: Gradient Descent**
```python
def gradient_descent(X, y, alpha=0.01, max_iter=1000, tol=1e-6):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    
    for i in range(max_iter):
        # Compute gradient
        grad = (1/n_samples) * X.T @ (X @ beta - y)
        
        # Update parameters
        beta_new = beta - alpha * grad
        
        # Check convergence
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    
    return beta

beta_gd = gradient_descent(X_with_bias, y)
print("Gradient descent solution:", beta_gd)
```

**Step 4: Ridge Regression**
```python
def ridge_regression(X, y, lambda_reg=1.0):
    n_features = X.shape[1]
    return np.linalg.inv(X.T @ X + lambda_reg * np.eye(n_features)) @ X.T @ y

beta_ridge = ridge_regression(X_with_bias, y, lambda_reg=0.1)
print("Ridge regression solution:", beta_ridge)
```

### Solution 2: PCA Implementation

**Step 1: Data Generation**
```python
# Generate 3D data with known structure
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
# Add correlation to create structure
X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
```

**Step 2: PCA using Eigenvalue Decomposition**
```python
def pca_eigen(X, n_components=2):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data
    X_pca = X_centered @ eigenvectors[:, :n_components]
    
    return X_pca, eigenvalues, eigenvectors

X_pca_eigen, eigenvals, eigenvecs = pca_eigen(X, 2)
```

**Step 3: PCA using SVD**
```python
def pca_svd(X, n_components=2):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Project data
    X_pca = X_centered @ Vt.T[:, :n_components]
    
    return X_pca, s, Vt

X_pca_svd, singular_vals, Vt = pca_svd(X, 2)
```

**Step 4: Comparison**
```python
from sklearn.decomposition import PCA

pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X)

print("Explained variance ratio (sklearn):", pca_sklearn.explained_variance_ratio_)
print("Explained variance ratio (manual):", eigenvals[:2] / np.sum(eigenvals))
```

### Solution 3: Neural Network for XOR

**Step 1: XOR Dataset**
```python
# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
```

**Step 2: Neural Network Implementation**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def neural_network_forward(X, W1, b1, W2, b2):
    # Hidden layer
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    
    # Output layer
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    return a1, a2

def neural_network_backward(X, y, W1, b1, W2, b2, learning_rate=0.1):
    m = X.shape[0]
    
    # Forward pass
    a1, a2 = neural_network_forward(X, W1, b1, W2, b2)
    
    # Backward pass
    delta2 = a2 - y.reshape(-1, 1)
    delta1 = (delta2 @ W2.T) * sigmoid_derivative(a1)
    
    # Gradients
    dW2 = a1.T @ delta2 / m
    db2 = np.sum(delta2, axis=0) / m
    dW1 = X.T @ delta1 / m
    db1 = np.sum(delta1, axis=0) / m
    
    # Update parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    return W1, b1, W2, b2
```

**Step 3: Training**
```python
# Initialize parameters
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros(4)
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

# Training loop
for epoch in range(10000):
    W1, b1, W2, b2 = neural_network_backward(X_xor, y_xor, W1, b1, W2, b2)
    
    if epoch % 1000 == 0:
        _, predictions = neural_network_forward(X_xor, W1, b1, W2, b2)
        loss = np.mean((predictions.flatten() - y_xor)**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Solution 4: Linear SVM Implementation

**Step 1: Data Generation**
```python
# Generate linearly separable data
np.random.seed(42)
n_samples = 100
X_pos = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
X_neg = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], n_samples//2)
X_svm = np.vstack([X_pos, X_neg])
y_svm = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
```

**Step 2: SVM Implementation**
```python
def hinge_loss(w, b, X, y, C=1.0):
    margins = y * (X @ w + b)
    loss = 0.5 * np.sum(w**2) + C * np.sum(np.maximum(0, 1 - margins))
    return loss

def svm_gradient(w, b, X, y, C=1.0):
    margins = y * (X @ w + b)
    support_vectors = margins < 1
    
    grad_w = w - C * np.sum(y[support_vectors, None] * X[support_vectors], axis=0)
    grad_b = -C * np.sum(y[support_vectors])
    
    return grad_w, grad_b

def train_svm(X, y, learning_rate=0.01, max_iter=1000, C=1.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for i in range(max_iter):
        grad_w, grad_b = svm_gradient(w, b, X, y, C)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        
        if i % 100 == 0:
            loss = hinge_loss(w, b, X, y, C)
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    return w, b
```

### Solution 5: Matrix Factorization

**Step 1: Synthetic Rating Matrix**
```python
# Create synthetic rating matrix
np.random.seed(42)
n_users, n_items = 50, 30
n_factors = 5

# True user and item factors
U_true = np.random.randn(n_users, n_factors)
V_true = np.random.randn(n_items, n_factors)

# Generate ratings
R_true = U_true @ V_true.T
R_true = np.clip(R_true, 1, 5)  # Clip to rating range

# Add noise and sparsity
R_observed = R_true + np.random.normal(0, 0.5, R_true.shape)
mask = np.random.random(R_true.shape) < 0.3  # 30% observed
R_observed[~mask] = 0
```

**Step 2: ALS Implementation**
```python
def als_step(R, U, V, lambda_reg=0.1):
    n_users, n_items = R.shape
    n_factors = U.shape[1]
    
    # Update U
    for i in range(n_users):
        observed_items = R[i, :] != 0
        if np.sum(observed_items) > 0:
            V_obs = V[observed_items, :]
            R_obs = R[i, observed_items]
            U[i, :] = np.linalg.solve(V_obs.T @ V_obs + lambda_reg * np.eye(n_factors), 
                                     V_obs.T @ R_obs)
    
    # Update V
    for j in range(n_items):
        observed_users = R[:, j] != 0
        if np.sum(observed_users) > 0:
            U_obs = U[observed_users, :]
            R_obs = R[observed_users, j]
            V[j, :] = np.linalg.solve(U_obs.T @ U_obs + lambda_reg * np.eye(n_factors), 
                                     U_obs.T @ R_obs)
    
    return U, V

def train_als(R, n_factors=5, max_iter=50, lambda_reg=0.1):
    n_users, n_items = R.shape
    U = np.random.randn(n_users, n_factors) * 0.1
    V = np.random.randn(n_items, n_factors) * 0.1
    
    for i in range(max_iter):
        U, V = als_step(R, U, V, lambda_reg)
        
        if i % 10 == 0:
            R_pred = U @ V.T
            rmse = np.sqrt(np.mean((R_pred[mask] - R_observed[mask])**2))
            print(f"Iteration {i}, RMSE: {rmse:.4f}")
    
    return U, V
```

### Solution 6: K-Means Clustering

**Step 1: Data Generation**
```python
# Generate 3 clusters
np.random.seed(42)
n_samples = 300
centers = np.array([[0, 0], [4, 4], [8, 0]])
X_kmeans = np.vstack([
    np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], n_samples//3),
    np.random.multivariate_normal(centers[1], [[1, 0.5], [0.5, 1]], n_samples//3),
    np.random.multivariate_normal(centers[2], [[1, 0.5], [0.5, 1]], n_samples//3)
])
```

**Step 2: K-Means Implementation**
```python
def kmeans(X, k=3, max_iter=100):
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for iteration in range(max_iter):
        # Assignment step
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update step
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

labels, centroids = kmeans(X_kmeans, k=3)
```

## Key Takeaways

- **Linear algebra is fundamental** to all machine learning algorithms
- **Matrix operations enable efficient computation** of gradients and predictions
- **Eigenvalue decomposition (PCA)** is crucial for dimensionality reduction
- **Matrix factorization enables collaborative filtering** in recommender systems
- **Neural networks rely heavily on matrix multiplication** for forward and backward passes
- **Optimization algorithms use linear algebra** for parameter updates
- **Geometric intuition** helps understand model behavior and convergence
- **Numerical stability** is crucial for reliable implementations

## Next Chapter

In the next chapter, we'll explore numerical linear algebra, focusing on numerical stability, conditioning, and efficient algorithms for large-scale problems. 