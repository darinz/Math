"""
Machine Learning Applications Implementation

This module provides comprehensive implementations of machine learning
algorithms using linear algebra concepts.

Key Concepts:
- Linear regression with normal equations and gradient descent
- Principal Component Analysis (PCA)
- Neural networks with backpropagation
- Support Vector Machines (SVM)
- Matrix factorization for recommender systems
- K-means clustering
- Optimization algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LinearRegressionML:
    """
    Linear regression implementation using linear algebra.
    """
    
    def __init__(self):
        """Initialize the linear regression toolkit."""
        self.beta = None
        self.X = None
        self.y = None
    
    def normal_equation(self, X, y):
        """
        Solve linear regression using normal equations.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix (with bias column)
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        np.ndarray : Parameter vector beta
        """
        # Normal equation: beta = (X^T X)^(-1) X^T y
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.X = X
        self.y = y
        return self.beta
    
    def ridge_regression(self, X, y, lambda_reg=1.0):
        """
        Solve ridge regression with L2 regularization.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix (with bias column)
        y : np.ndarray
            Target vector
        lambda_reg : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray : Parameter vector beta
        """
        n_features = X.shape[1]
        # Ridge solution: beta = (X^T X + lambda I)^(-1) X^T y
        self.beta = np.linalg.inv(X.T @ X + lambda_reg * np.eye(n_features)) @ X.T @ y
        self.X = X
        self.y = y
        return self.beta
    
    def gradient_descent(self, X, y, alpha=0.01, max_iter=1000, tol=1e-6):
        """
        Solve linear regression using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix (with bias column)
        y : np.ndarray
            Target vector
        alpha : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        np.ndarray : Parameter vector beta
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        
        for i in range(max_iter):
            # Compute gradient
            grad = (1/n_samples) * X.T @ (X @ self.beta - y)
            
            # Update parameters
            beta_new = self.beta - alpha * grad
            
            # Check convergence
            if np.linalg.norm(beta_new - self.beta) < tol:
                break
            self.beta = beta_new
        
        self.X = X
        self.y = y
        return self.beta
    
    def predict(self, X):
        """
        Make predictions using fitted model.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if self.beta is None:
            raise ValueError("Model not fitted. Call normal_equation, ridge_regression, or gradient_descent first.")
        return X @ self.beta
    
    def compute_loss(self, X, y):
        """
        Compute mean squared error loss.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        float : Mean squared error
        """
        predictions = self.predict(X)
        return np.mean((predictions - y)**2)

class PCAML:
    """
    Principal Component Analysis implementation.
    """
    
    def __init__(self):
        """Initialize the PCA toolkit."""
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None
    
    def fit_eigenvalue(self, X):
        """
        Fit PCA using eigenvalue decomposition.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
            
        Returns:
        --------
        self : PCAML instance
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        self.explained_variance = eigenvalues[idx]
        self.components = eigenvectors[:, idx]
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)
        
        return self
    
    def fit_svd(self, X):
        """
        Fit PCA using SVD decomposition.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
            
        Returns:
        --------
        self : PCAML instance
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # SVD decomposition
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Components are right singular vectors
        self.components = Vt.T
        self.explained_variance = s**2 / (len(X) - 1)
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)
        
        return self
    
    def transform(self, X, n_components=None):
        """
        Transform data to principal components.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
        n_components : int, optional
            Number of components to use
            
        Returns:
        --------
        np.ndarray : Transformed data
        """
        if self.components is None:
            raise ValueError("PCA not fitted. Call fit_eigenvalue or fit_svd first.")
        
        if n_components is None:
            n_components = self.components.shape[1]
        
        X_centered = X - self.mean
        return X_centered @ self.components[:, :n_components]
    
    def inverse_transform(self, X_transformed, n_components=None):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : np.ndarray
            Transformed data
        n_components : int, optional
            Number of components used
            
        Returns:
        --------
        np.ndarray : Data in original space
        """
        if n_components is None:
            n_components = X_transformed.shape[1]
        
        return X_transformed @ self.components[:, :n_components].T + self.mean

class NeuralNetwork:
    """
    Simple neural network implementation with backpropagation.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        layer_sizes : list
            List of layer sizes [input_size, hidden_size, ..., output_size]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        list : List of activations for each layer
        """
        activations = [X]
        
        for i in range(len(self.weights)):
            z = self.weights[i] @ activations[-1].T + self.biases[i].reshape(-1, 1)
            a = self.sigmoid(z)
            activations.append(a.T)
        
        return activations
    
    def backward(self, X, y, learning_rate=0.1):
        """
        Backward pass (backpropagation).
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target data
        learning_rate : float
            Learning rate
        """
        m = X.shape[0]
        
        # Forward pass
        activations = self.forward(X)
        
        # Backward pass
        delta = activations[-1] - y.reshape(-1, 1)
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = delta.T @ activations[i] / m
            db = np.sum(delta, axis=0) / m
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Compute delta for next layer
            if i > 0:
                delta = (delta @ self.weights[i]) * self.sigmoid_derivative(activations[i])
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target data
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        """
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                predictions = self.forward(X)[-1]
                loss = np.mean((predictions - y.reshape(-1, 1))**2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        return self.forward(X)[-1]

class SVM:
    """
    Support Vector Machine implementation.
    """
    
    def __init__(self, C=1.0):
        """
        Initialize SVM.
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        """
        self.C = C
        self.w = None
        self.b = None
    
    def hinge_loss(self, w, b, X, y):
        """
        Compute hinge loss.
        
        Parameters:
        -----------
        w : np.ndarray
            Weight vector
        b : float
            Bias term
        X : np.ndarray
            Input data
        y : np.ndarray
            Target labels (-1 or 1)
            
        Returns:
        --------
        float : Hinge loss
        """
        margins = y * (X @ w + b)
        loss = 0.5 * np.sum(w**2) + self.C * np.sum(np.maximum(0, 1 - margins))
        return loss
    
    def gradient(self, w, b, X, y):
        """
        Compute gradient of hinge loss.
        
        Parameters:
        -----------
        w : np.ndarray
            Weight vector
        b : float
            Bias term
        X : np.ndarray
            Input data
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        tuple : (gradient_w, gradient_b)
        """
        margins = y * (X @ w + b)
        support_vectors = margins < 1
        
        grad_w = w - self.C * np.sum(y[support_vectors, None] * X[support_vectors], axis=0)
        grad_b = -self.C * np.sum(y[support_vectors])
        
        return grad_w, grad_b
    
    def train(self, X, y, learning_rate=0.01, max_iter=1000):
        """
        Train SVM using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target labels
        learning_rate : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(max_iter):
            grad_w, grad_b = self.gradient(self.w, self.b, X, y)
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b
            
            if i % 100 == 0:
                loss = self.hinge_loss(self.w, self.b, X, y)
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray : Predictions (-1 or 1)
        """
        if self.w is None:
            raise ValueError("SVM not trained. Call train first.")
        return np.sign(X @ self.w + self.b)

class MatrixFactorization:
    """
    Matrix factorization for recommender systems.
    """
    
    def __init__(self, n_factors=10, lambda_reg=0.1):
        """
        Initialize matrix factorization.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        lambda_reg : float
            Regularization parameter
        """
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.U = None
        self.V = None
    
    def als_step(self, R):
        """
        One step of Alternating Least Squares.
        
        Parameters:
        -----------
        R : np.ndarray
            Rating matrix (NaN for missing ratings)
        """
        n_users, n_items = R.shape
        
        # Update U
        for i in range(n_users):
            observed_items = ~np.isnan(R[i, :])
            if np.sum(observed_items) > 0:
                V_obs = self.V[observed_items, :]
                R_obs = R[i, observed_items]
                self.U[i, :] = np.linalg.solve(
                    V_obs.T @ V_obs + self.lambda_reg * np.eye(self.n_factors),
                    V_obs.T @ R_obs
                )
        
        # Update V
        for j in range(n_items):
            observed_users = ~np.isnan(R[:, j])
            if np.sum(observed_users) > 0:
                U_obs = self.U[observed_users, :]
                R_obs = R[observed_users, j]
                self.V[j, :] = np.linalg.solve(
                    U_obs.T @ U_obs + self.lambda_reg * np.eye(self.n_factors),
                    U_obs.T @ R_obs
                )
    
    def sgd_step(self, R, learning_rate=0.01):
        """
        One step of Stochastic Gradient Descent.
        
        Parameters:
        -----------
        R : np.ndarray
            Rating matrix
        learning_rate : float
            Learning rate
        """
        n_users, n_items = R.shape
        
        # Sample random rating
        observed_ratings = np.where(~np.isnan(R))
        if len(observed_ratings[0]) > 0:
            idx = np.random.randint(len(observed_ratings[0]))
            i, j = observed_ratings[0][idx], observed_ratings[1][idx]
            r_ij = R[i, j]
            
            # Prediction
            pred = self.U[i, :] @ self.V[j, :]
            error = r_ij - pred
            
            # Update
            self.U[i, :] += learning_rate * (error * self.V[j, :] - self.lambda_reg * self.U[i, :])
            self.V[j, :] += learning_rate * (error * self.U[i, :] - self.lambda_reg * self.V[j, :])
    
    def fit(self, R, method='als', max_iter=100, learning_rate=0.01):
        """
        Fit matrix factorization model.
        
        Parameters:
        -----------
        R : np.ndarray
            Rating matrix
        method : str
            'als' or 'sgd'
        max_iter : int
            Maximum number of iterations
        learning_rate : float
            Learning rate for SGD
        """
        n_users, n_items = R.shape
        
        # Initialize U and V
        self.U = np.random.randn(n_users, self.n_factors) * 0.1
        self.V = np.random.randn(n_items, self.n_factors) * 0.1
        
        for i in range(max_iter):
            if method == 'als':
                self.als_step(R)
            elif method == 'sgd':
                for _ in range(n_users * n_items // 10):  # Multiple SGD steps
                    self.sgd_step(R, learning_rate)
            
            if i % 10 == 0:
                R_pred = self.U @ self.V.T
                mask = ~np.isnan(R)
                rmse = np.sqrt(np.mean((R_pred[mask] - R[mask])**2))
                print(f"Iteration {i}, RMSE: {rmse:.4f}")
    
    def predict(self, user_idx, item_idx):
        """
        Predict rating for user-item pair.
        
        Parameters:
        -----------
        user_idx : int
            User index
        item_idx : int
            Item index
            
        Returns:
        --------
        float : Predicted rating
        """
        if self.U is None or self.V is None:
            raise ValueError("Model not fitted. Call fit first.")
        return self.U[user_idx, :] @ self.V[item_idx, :]

class KMeansML:
    """
    K-means clustering implementation.
    """
    
    def __init__(self, k=3, max_iter=100):
        """
        Initialize K-means.
        
        Parameters:
        -----------
        k : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
        """
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        """
        Fit K-means clustering.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
            
        Returns:
        --------
        self : KMeansML instance
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for iteration in range(self.max_iter):
            # Assignment step
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update step
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
            
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        if self.centroids is None:
            raise ValueError("K-means not fitted. Call fit first.")
        
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

def demonstrate_linear_regression():
    """
    Demonstrate linear regression implementations.
    """
    print("=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)
    
    lr = LinearRegressionML()
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_beta = np.array([2, 3, 1])  # [beta_1, beta_2, beta_0]
    X_with_bias = np.column_stack([X, np.ones(n_samples)])
    y = X_with_bias @ true_beta + np.random.normal(0, 0.1, n_samples)
    
    print(f"True parameters: {true_beta}")
    print(f"Data shape: {X.shape}")
    
    # Normal equation solution
    beta_normal = lr.normal_equation(X_with_bias, y)
    print(f"\nNormal equation solution: {beta_normal}")
    print(f"Error: {np.linalg.norm(beta_normal - true_beta):.6f}")
    
    # Gradient descent solution
    beta_gd = lr.gradient_descent(X_with_bias, y, alpha=0.01, max_iter=1000)
    print(f"\nGradient descent solution: {beta_gd}")
    print(f"Error: {np.linalg.norm(beta_gd - true_beta):.6f}")
    
    # Ridge regression
    beta_ridge = lr.ridge_regression(X_with_bias, y, lambda_reg=0.1)
    print(f"\nRidge regression solution: {beta_ridge}")
    print(f"Error: {np.linalg.norm(beta_ridge - true_beta):.6f}")
    
    # Compare with sklearn
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X, y)
    sklearn_beta = np.append(sklearn_lr.coef_, sklearn_lr.intercept_)
    print(f"\nSklearn solution: {sklearn_beta}")
    print(f"Error: {np.linalg.norm(sklearn_beta - true_beta):.6f}")

def demonstrate_pca():
    """
    Demonstrate PCA implementations.
    """
    print("\n\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 60)
    
    pca = PCAML()
    
    # Generate synthetic data with known structure
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    # Add correlation to create structure
    X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    print(f"Data shape: {X.shape}")
    print(f"Original rank: {np.linalg.matrix_rank(X)}")
    
    # PCA using eigenvalue decomposition
    pca.fit_eigenvalue(X)
    X_pca_eigen = pca.transform(X, n_components=2)
    
    print(f"\nEigenvalue decomposition PCA:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio[:2]}")
    print(f"Reduced data shape: {X_pca_eigen.shape}")
    
    # PCA using SVD
    pca_svd = PCAML()
    pca_svd.fit_svd(X)
    X_pca_svd = pca_svd.transform(X, n_components=2)
    
    print(f"\nSVD PCA:")
    print(f"Explained variance ratio: {pca_svd.explained_variance_ratio[:2]}")
    print(f"Reduced data shape: {X_pca_svd.shape}")
    
    # Compare with sklearn
    sklearn_pca = PCA(n_components=2)
    X_pca_sklearn = sklearn_pca.fit_transform(X)
    
    print(f"\nSklearn PCA:")
    print(f"Explained variance ratio: {sklearn_pca.explained_variance_ratio_}")
    print(f"Reduced data shape: {X_pca_sklearn.shape}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_pca_eigen[:, 0], X_pca_eigen[:, 1])
    plt.title('PCA (Eigenvalue)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca_svd[:, 0], X_pca_svd[:, 1])
    plt.title('PCA (SVD)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1])
    plt.title('PCA (Sklearn)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    plt.show()

def demonstrate_neural_network():
    """
    Demonstrate neural network implementation.
    """
    print("\n\n" + "=" * 60)
    print("NEURAL NETWORK")
    print("=" * 60)
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print(f"XOR dataset:")
    print(f"X: {X_xor}")
    print(f"y: {y_xor}")
    
    # Create neural network
    nn = NeuralNetwork([2, 4, 1])
    
    # Train network
    nn.train(X_xor, y_xor, epochs=10000, learning_rate=0.1)
    
    # Make predictions
    predictions = nn.predict(X_xor)
    print(f"\nPredictions: {predictions.flatten()}")
    print(f"Targets: {y_xor}")
    print(f"Accuracy: {np.mean(np.round(predictions.flatten()) == y_xor):.2f}")
    
    # Visualize decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = nn.predict(grid_points).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_predictions, alpha=0.8)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, edgecolors='black')
    plt.title('Neural Network Decision Boundary (XOR)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def demonstrate_svm():
    """
    Demonstrate SVM implementation.
    """
    print("\n\n" + "=" * 60)
    print("SUPPORT VECTOR MACHINE")
    print("=" * 60)
    
    # Generate linearly separable data
    np.random.seed(42)
    n_samples = 100
    X_pos = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    X_neg = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], n_samples//2)
    X_svm = np.vstack([X_pos, X_neg])
    y_svm = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
    
    print(f"Data shape: {X_svm.shape}")
    print(f"Class distribution: {np.bincount((y_svm + 1) // 2)}")
    
    # Train SVM
    svm = SVM(C=1.0)
    svm.train(X_svm, y_svm, learning_rate=0.01, max_iter=1000)
    
    # Make predictions
    predictions = svm.predict(X_svm)
    accuracy = np.mean(predictions == y_svm)
    print(f"\nTraining accuracy: {accuracy:.4f}")
    
    # Visualize decision boundary
    x_min, x_max = X_svm[:, 0].min() - 1, X_svm[:, 0].max() + 1
    y_min, y_max = X_svm[:, 1].min() - 1, X_svm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = svm.predict(grid_points).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_predictions, alpha=0.8)
    plt.scatter(X_svm[:, 0], X_svm[:, 1], c=y_svm, s=50, edgecolors='black')
    plt.title('SVM Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def demonstrate_matrix_factorization():
    """
    Demonstrate matrix factorization for recommender systems.
    """
    print("\n\n" + "=" * 60)
    print("MATRIX FACTORIZATION")
    print("=" * 60)
    
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
    R_observed[~mask] = np.nan
    
    print(f"Rating matrix shape: {R_observed.shape}")
    print(f"Number of observed ratings: {np.sum(~np.isnan(R_observed))}")
    print(f"Sparsity: {np.sum(np.isnan(R_observed)) / R_observed.size:.2%}")
    
    # ALS method
    mf_als = MatrixFactorization(n_factors=5, lambda_reg=0.1)
    mf_als.fit(R_observed, method='als', max_iter=50)
    
    # SGD method
    mf_sgd = MatrixFactorization(n_factors=5, lambda_reg=0.1)
    mf_sgd.fit(R_observed, method='sgd', max_iter=50, learning_rate=0.01)
    
    # Evaluate
    R_pred_als = mf_als.U @ mf_sgd.V.T
    R_pred_sgd = mf_sgd.U @ mf_sgd.V.T
    
    mask_eval = ~np.isnan(R_observed)
    rmse_als = np.sqrt(np.mean((R_pred_als[mask_eval] - R_observed[mask_eval])**2))
    rmse_sgd = np.sqrt(np.mean((R_pred_sgd[mask_eval] - R_observed[mask_eval])**2))
    
    print(f"\nALS RMSE: {rmse_als:.4f}")
    print(f"SGD RMSE: {rmse_sgd:.4f}")

def demonstrate_kmeans():
    """
    Demonstrate K-means clustering.
    """
    print("\n\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)
    
    # Generate 3 clusters
    np.random.seed(42)
    n_samples = 300
    centers = np.array([[0, 0], [4, 4], [8, 0]])
    X_kmeans = np.vstack([
        np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], n_samples//3),
        np.random.multivariate_normal(centers[1], [[1, 0.5], [0.5, 1]], n_samples//3),
        np.random.multivariate_normal(centers[2], [[1, 0.5], [0.5, 1]], n_samples//3)
    ])
    
    print(f"Data shape: {X_kmeans.shape}")
    print(f"True centers: {centers}")
    
    # Apply K-means
    kmeans = KMeansML(k=3, max_iter=100)
    kmeans.fit(X_kmeans)
    
    print(f"\nK-means centers: {kmeans.centroids}")
    print(f"Cluster sizes: {np.bincount(kmeans.labels)}")
    
    # Compare with sklearn
    sklearn_kmeans = KMeans(n_clusters=3, random_state=42)
    sklearn_labels = sklearn_kmeans.fit_predict(X_kmeans)
    
    print(f"\nSklearn centers: {sklearn_kmeans.cluster_centers_}")
    print(f"Sklearn cluster sizes: {np.bincount(sklearn_labels)}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=kmeans.labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-means Clustering (Custom)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=sklearn_labels, cmap='viridis')
    plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-means Clustering (Sklearn)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_exercises():
    """
    Create exercises for ML applications.
    """
    print("\n\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    
    # Exercise 1: Linear regression with different methods
    print("Exercise 1: Linear Regression Comparison")
    print("-" * 50)
    
    # Generate data with multicollinearity
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    X[:, 2] = 0.8 * X[:, 0] + 0.2 * X[:, 1]  # Create multicollinearity
    true_beta = np.array([2, 3, 1])
    X_with_bias = np.column_stack([X, np.ones(n_samples)])
    y = X_with_bias @ true_beta + np.random.normal(0, 0.1, n_samples)
    
    lr = LinearRegressionML()
    
    # Compare methods
    beta_normal = lr.normal_equation(X_with_bias, y)
    beta_ridge = lr.ridge_regression(X_with_bias, y, lambda_reg=0.1)
    
    print(f"Normal equation solution: {beta_normal}")
    print(f"Ridge regression solution: {beta_ridge}")
    print(f"True parameters: {true_beta}")
    
    # Exercise 2: PCA on iris dataset
    print("\n\nExercise 2: PCA on Iris Dataset")
    print("-" * 50)
    
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    pca = PCAML()
    pca.fit_eigenvalue(X_iris)
    X_pca = pca.transform(X_iris, n_components=2)
    
    print(f"Original data shape: {X_iris.shape}")
    print(f"Reduced data shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio[:2]}")
    
    # Exercise 3: Neural network for classification
    print("\n\nExercise 3: Neural Network Classification")
    print("-" * 50)
    
    # Create simple classification dataset
    X_class = np.random.randn(100, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    
    nn = NeuralNetwork([2, 4, 1])
    nn.train(X_class, y_class, epochs=1000, learning_rate=0.1)
    
    predictions = nn.predict(X_class)
    accuracy = np.mean(np.round(predictions.flatten()) == y_class)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Exercise 4: SVM with different kernels
    print("\n\nExercise 4: SVM Analysis")
    print("-" * 50)
    
    # Create non-linearly separable data
    X_svm_nl = np.random.randn(100, 2)
    y_svm_nl = ((X_svm_nl[:, 0]**2 + X_svm_nl[:, 1]**2) > 1).astype(int) * 2 - 1
    
    svm = SVM(C=1.0)
    svm.train(X_svm_nl, y_svm_nl, learning_rate=0.01, max_iter=1000)
    
    predictions = svm.predict(X_svm_nl)
    accuracy = np.mean(predictions == y_svm_nl)
    print(f"SVM accuracy: {accuracy:.4f}")
    
    # Exercise 5: Matrix factorization analysis
    print("\n\nExercise 5: Matrix Factorization Analysis")
    print("-" * 50)
    
    # Create rating matrix with known structure
    n_users, n_items = 20, 15
    U_true = np.random.randn(n_users, 3)
    V_true = np.random.randn(n_items, 3)
    R_true = U_true @ V_true.T
    R_true = np.clip(R_true, 1, 5)
    
    # Add sparsity
    mask = np.random.random(R_true.shape) < 0.4
    R_observed = R_true.copy()
    R_observed[~mask] = np.nan
    
    mf = MatrixFactorization(n_factors=3, lambda_reg=0.1)
    mf.fit(R_observed, method='als', max_iter=30)
    
    R_pred = mf.U @ mf.V.T
    mask_eval = ~np.isnan(R_observed)
    rmse = np.sqrt(np.mean((R_pred[mask_eval] - R_observed[mask_eval])**2))
    print(f"Matrix factorization RMSE: {rmse:.4f}")
    
    # Exercise 6: K-means with different k
    print("\n\nExercise 6: K-means with Different k")
    print("-" * 50)
    
    # Generate data with 4 clusters
    centers_4 = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
    X_kmeans_4 = np.vstack([
        np.random.multivariate_normal(centers_4[i], [[0.5, 0.1], [0.1, 0.5]], 50)
        for i in range(4)
    ])
    
    for k in [2, 3, 4, 5]:
        kmeans = KMeansML(k=k, max_iter=100)
        kmeans.fit(X_kmeans_4)
        print(f"k={k}: Cluster sizes = {np.bincount(kmeans.labels)}")

def main():
    """
    Main function to run all demonstrations and exercises.
    """
    print("MACHINE LEARNING APPLICATIONS IMPLEMENTATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_linear_regression()
    demonstrate_pca()
    demonstrate_neural_network()
    demonstrate_svm()
    demonstrate_matrix_factorization()
    demonstrate_kmeans()
    
    # Run exercises
    create_exercises()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key ML Algorithms Covered:
    1. Linear regression with normal equations and gradient descent
    2. Principal Component Analysis (PCA) for dimensionality reduction
    3. Neural networks with backpropagation for classification
    4. Support Vector Machines (SVM) for binary classification
    5. Matrix factorization for recommender systems
    6. K-means clustering for unsupervised learning
    
    Key Linear Algebra Concepts:
    - Matrix operations for efficient computation
    - Eigenvalue decomposition for PCA
    - Gradient computation for optimization
    - Matrix factorization for collaborative filtering
    - Vector operations for similarity and distance
    
    Key Takeaways:
    - Linear algebra is fundamental to all ML algorithms
    - Matrix operations enable efficient computation
    - Understanding linear algebra improves algorithm design
    - Numerical stability is crucial for reliable implementations
    - Geometric intuition helps understand model behavior
    """)

if __name__ == "__main__":
    main() 