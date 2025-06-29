# Calculus in Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-blue.svg)](https://scikit-learn.org/)

## Introduction

Calculus is fundamental to machine learning, providing the mathematical foundation for optimization algorithms, neural network training, and understanding model behavior. This section explores key applications of calculus in modern machine learning.

## 9.1 Gradient Descent and Optimization

### Basic Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simple gradient descent implementation
def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """
    Gradient descent optimization
    
    Parameters:
    f: objective function
    grad_f: gradient function
    x0: initial point
    learning_rate: step size
    max_iter: maximum iterations
    tol: tolerance for convergence
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        
        # Update rule
        x_new = x - learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# Example: Minimize f(x) = x² + 2x + 1
def objective_function(x):
    return x**2 + 2*x + 1

def gradient_function(x):
    return 2*x + 2

# Run gradient descent
x0 = 5.0
optimal_x, history = gradient_descent(objective_function, gradient_function, x0, learning_rate=0.1)

print(f"Optimal x: {optimal_x:.6f}")
print(f"Optimal value: {objective_function(optimal_x):.6f}")
print(f"Number of iterations: {len(history)}")

# Visualize optimization
x_vals = np.linspace(-2, 6, 100)
y_vals = objective_function(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
plt.plot(history, [objective_function(x) for x in history], 'ro-', 
         markersize=4, label='Optimization path')
plt.scatter(optimal_x, objective_function(optimal_x), c='red', s=100, 
           label=f'Optimum: ({optimal_x:.3f}, {objective_function(optimal_x):.3f})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()
```

### Stochastic Gradient Descent (SGD)

```python
# Stochastic Gradient Descent for linear regression
def sgd_linear_regression(X, y, learning_rate=0.01, epochs=100, batch_size=32):
    """
    SGD for linear regression: y = w*x + b
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    
    history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            # Mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Predictions
            y_pred = X_batch @ w + b
            
            # Gradients
            dw = (2/batch_size) * X_batch.T @ (y_pred - y_batch)
            db = (2/batch_size) * np.sum(y_pred - y_batch)
            
            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Record loss
        y_pred_full = X @ w + b
        loss = np.mean((y_pred_full - y)**2)
        history.append(loss)
    
    return w, b, history

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 2)
true_w = np.array([2.0, -1.5])
true_b = 1.0
y = X @ true_w + true_b + 0.1 * np.random.randn(n_samples)

# Run SGD
w_learned, b_learned, loss_history = sgd_linear_regression(X, y, learning_rate=0.01, epochs=50)

print(f"True weights: {true_w}")
print(f"Learned weights: {w_learned}")
print(f"True bias: {true_b}")
print(f"Learned bias: {b_learned}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('SGD Training Loss')
plt.grid(True)
plt.show()
```

## 9.2 Backpropagation in Neural Networks

### Simple Neural Network with Backpropagation

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        
        # Backward pass
        dz2 = self.a2 - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = dz2 @ self.W2.T * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test predictions
predictions = nn.forward(X)
print("\nPredictions:")
for i, (x, pred, true) in enumerate(zip(X, predictions, y)):
    print(f"Input: {x}, Prediction: {pred[0]:.3f}, True: {true[0]}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss')
plt.yscale('log')
plt.grid(True)
plt.show()
```

## 9.3 Loss Functions and Their Derivatives

### Common Loss Functions

```python
def mse_loss(y_pred, y_true):
    """Mean Squared Error loss"""
    return np.mean((y_pred - y_true)**2)

def mse_derivative(y_pred, y_true):
    """Derivative of MSE with respect to predictions"""
    return 2 * (y_pred - y_true) / len(y_pred)

def cross_entropy_loss(y_pred, y_true):
    """Binary cross-entropy loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_derivative(y_pred, y_true):
    """Derivative of binary cross-entropy with respect to predictions"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss - combines MSE and MAE"""
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

def huber_derivative(y_pred, y_true, delta=1.0):
    """Derivative of Huber loss"""
    error = y_pred - y_true
    abs_error = np.abs(error)
    return np.where(abs_error <= delta, error, delta * np.sign(error))

# Demonstrate loss functions
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.3, 0.9])

print("Loss Function Comparison:")
print(f"MSE Loss: {mse_loss(y_pred, y_true):.4f}")
print(f"Cross-Entropy Loss: {cross_entropy_loss(y_pred, y_true):.4f}")
print(f"Huber Loss: {huber_loss(y_pred, y_true):.4f}")

print("\nDerivatives:")
print(f"MSE Derivative: {mse_derivative(y_pred, y_true)}")
print(f"Cross-Entropy Derivative: {cross_entropy_derivative(y_pred, y_true)}")
print(f"Huber Derivative: {huber_derivative(y_pred, y_true)}")

# Visualize loss functions
x_vals = np.linspace(0, 1, 100)
y_true_fixed = 1.0

mse_vals = [(x - y_true_fixed)**2 for x in x_vals]
ce_vals = [-y_true_fixed * np.log(x) - (1 - y_true_fixed) * np.log(1 - x) for x in x_vals]
huber_vals = [huber_loss(x, y_true_fixed) for x in x_vals]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_vals, mse_vals, 'b-', linewidth=2)
plt.xlabel('Prediction')
plt.ylabel('MSE Loss')
plt.title('Mean Squared Error')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_vals, ce_vals, 'r-', linewidth=2)
plt.xlabel('Prediction')
plt.ylabel('Cross-Entropy Loss')
plt.title('Binary Cross-Entropy')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_vals, huber_vals, 'g-', linewidth=2)
plt.xlabel('Prediction')
plt.ylabel('Huber Loss')
plt.title('Huber Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 9.4 Regularization and Gradient Clipping

### L1 and L2 Regularization

```python
def l1_regularization(weights, lambda_param=0.01):
    """L1 regularization (Lasso)"""
    return lambda_param * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_param=0.01):
    """L2 regularization (Ridge)"""
    return lambda_param * np.sum(weights**2)

def l1_derivative(weights, lambda_param=0.01):
    """Derivative of L1 regularization"""
    return lambda_param * np.sign(weights)

def l2_derivative(weights, lambda_param=0.01):
    """Derivative of L2 regularization"""
    return 2 * lambda_param * weights

# Demonstrate regularization effects
weights = np.array([0.5, -0.3, 0.8, -0.1])

print("Regularization Comparison:")
print(f"L1 regularization: {l1_regularization(weights):.4f}")
print(f"L2 regularization: {l2_regularization(weights):.4f}")
print(f"L1 derivative: {l1_derivative(weights)}")
print(f"L2 derivative: {l2_derivative(weights)}")

# Visualize regularization effects
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Original loss function: f(x,y) = x² + y²
Z_original = X**2 + Y**2

# With L2 regularization
lambda_param = 0.1
Z_l2 = Z_original + lambda_param * (X**2 + Y**2)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_original, cmap='viridis')
ax1.set_title('Original Loss Function')
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Loss')

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_l2, cmap='viridis')
ax2.set_title('Loss with L2 Regularization')
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.show()
```

### Gradient Clipping

```python
def gradient_clipping(gradients, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    norm = np.linalg.norm(gradients)
    if norm > max_norm:
        gradients = gradients * max_norm / norm
    return gradients

# Demonstrate gradient clipping
def demonstrate_gradient_clipping():
    # Simulate large gradients
    large_gradients = np.array([10.0, -15.0, 8.0, -12.0])
    
    print("Original gradients:", large_gradients)
    print("Original norm:", np.linalg.norm(large_gradients))
    
    clipped_gradients = gradient_clipping(large_gradients, max_norm=5.0)
    print("Clipped gradients:", clipped_gradients)
    print("Clipped norm:", np.linalg.norm(clipped_gradients))

demonstrate_gradient_clipping()
```

## 9.5 Advanced Optimization Algorithms

### Adam Optimizer

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, params, gradients):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params

# Test Adam optimizer
def test_adam_optimizer():
    # Rosenbrock function
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_gradient(x):
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    # Initialize
    x = np.array([-1.0, -1.0])
    optimizer = AdamOptimizer(learning_rate=0.01)
    history = [x.copy()]
    
    # Optimize
    for i in range(1000):
        gradient = rosenbrock_gradient(x)
        x = optimizer.update(x, gradient)
        history.append(x.copy())
    
    history = np.array(history)
    
    print(f"Final point: {x}")
    print(f"Final value: {rosenbrock(x):.6f}")
    
    # Visualize optimization path
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = rosenbrock([X, Y])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.plot(history[:, 0], history[:, 1], 'r-', linewidth=2, label='Adam optimization')
    plt.scatter(history[0, 0], history[0, 1], c='red', s=100, label='Start')
    plt.scatter(history[-1, 0], history[-1, 1], c='green', s=100, label='End')
    plt.title('Adam Optimizer on Rosenbrock Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

test_adam_optimizer()
```

## 9.6 Calculus in Deep Learning

### Automatic Differentiation

```python
# Simple automatic differentiation example
class AutoDiff:
    def __init__(self, value, derivative=1.0):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        if isinstance(other, AutoDiff):
            return AutoDiff(self.value + other.value, self.derivative + other.derivative)
        else:
            return AutoDiff(self.value + other, self.derivative)
    
    def __mul__(self, other):
        if isinstance(other, AutoDiff):
            return AutoDiff(self.value * other.value, 
                          self.derivative * other.value + self.value * other.derivative)
        else:
            return AutoDiff(self.value * other, self.derivative * other)
    
    def __pow__(self, power):
        return AutoDiff(self.value**power, power * self.value**(power-1) * self.derivative)

# Test automatic differentiation
def test_autodiff():
    # f(x) = x² + 2x + 1
    x = AutoDiff(3.0)  # x = 3, dx/dx = 1
    f = x**2 + 2*x + 1
    
    print(f"f(3) = {f.value}")
    print(f"f'(3) = {f.derivative}")
    
    # Verify with symbolic differentiation
    # f'(x) = 2x + 2
    # f'(3) = 2*3 + 2 = 8
    print(f"Expected f'(3) = 8")

test_autodiff()
```

## Summary

- **Gradient descent** is the foundation of most ML optimization algorithms
- **Backpropagation** uses the chain rule to compute gradients in neural networks
- **Loss functions** and their derivatives are crucial for model training
- **Regularization** helps prevent overfitting and improves generalization
- **Advanced optimizers** like Adam combine momentum and adaptive learning rates
- **Automatic differentiation** enables efficient gradient computation in deep learning frameworks

## Next Steps

Understanding calculus in machine learning enables you to implement custom loss functions, design new optimization algorithms, and debug training issues. The next section covers numerical methods for when analytical solutions are not available. 