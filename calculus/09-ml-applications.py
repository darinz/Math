# Python code extracted from 09-ml-applications.md
# This file contains Python code examples from the corresponding markdown file

# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simple gradient descent implementation
def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """
    Gradient descent optimization algorithm.
    
    Mathematical foundation:
    - Uses the update rule: x_{k+1} = x_k - α∇f(x_k)
    - Converges to local minima for convex functions
    - Learning rate α controls convergence speed and stability
    
    Parameters:
    f: objective function (scalar-valued)
    grad_f: gradient function (vector-valued)
    x0: initial point
    learning_rate: step size (α)
    max_iter: maximum iterations
    tol: tolerance for convergence
    
    Returns:
    optimal_x: approximate minimizer
    history: optimization trajectory
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        
        # Update rule: x_new = x - α∇f(x)
        x_new = x - learning_rate * gradient
        
        # Check convergence: ||x_new - x|| < tolerance
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# Example: Minimize f(x) = x² + 2x + 1
# This is a convex quadratic function with minimum at x = -1
def objective_function(x):
    """
    Objective function: f(x) = x² + 2x + 1
    - Convex function (second derivative = 2 > 0)
    - Global minimum at x = -1
    - f(-1) = 0
    """
    return x**2 + 2*x + 1

def gradient_function(x):
    """
    Gradient: f'(x) = 2x + 2
    - Linear function
    - Zero at x = -1 (critical point)
    """
    return 2*x + 2

# Run gradient descent
x0 = 5.0  # Start far from the minimum
optimal_x, history = gradient_descent(objective_function, gradient_function, x0, learning_rate=0.1)

print(f"Optimal x: {optimal_x:.6f}")
print(f"Optimal value: {objective_function(optimal_x):.6f}")
print(f"Number of iterations: {len(history)}")
print(f"Gradient at optimum: {gradient_function(optimal_x):.6f}")

# Visualize optimization
x_vals = np.linspace(-2, 6, 100)
y_vals = objective_function(x_vals)

plt.figure(figsize=(12, 8))

# Plot objective function
plt.subplot(2, 1, 1)
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

# Plot gradient and convergence
plt.subplot(2, 1, 2)
grad_vals = gradient_function(x_vals)
plt.plot(x_vals, grad_vals, 'g-', linewidth=2, label="f'(x) = 2x + 2")
plt.plot(history, [gradient_function(x) for x in history], 'mo-', 
         markersize=4, label='Gradient along path')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Gradient Function and Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Code Block 2
# Stochastic Gradient Descent for linear regression
def sgd_linear_regression(X, y, learning_rate=0.01, epochs=100, batch_size=32):
    """
    SGD for linear regression: y = Xw + b
    
    Mathematical foundation:
    - Loss function: L(w,b) = (1/n)∑(y_i - (X_i^T w + b))²
    - Gradients: ∇_w L = -(2/n)X^T(y - Xw - b), ∇_b L = -(2/n)∑(y - Xw - b)
    - Update rule: w = w - α∇_w L, b = b - α∇_b L
    
    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target vector (n_samples,)
    learning_rate: step size
    epochs: number of passes through the dataset
    batch_size: size of mini-batches
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights to zero
    b = 0.0                   # Initialize bias to zero
    
    history = []
    
    for epoch in range(epochs):
        # Shuffle data for stochastic sampling
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            # Mini-batch: sample a subset of data
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass: compute predictions
            y_pred = X_batch @ w + b
            
            # Compute gradients using the mini-batch
            error = y_pred - y_batch
            dw = (2/batch_size) * X_batch.T @ error  # Gradient w.r.t. weights
            db = (2/batch_size) * np.sum(error)      # Gradient w.r.t. bias
            
            # Update parameters using gradient descent
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Record loss for monitoring
        y_pred_full = X @ w + b
        loss = np.mean((y_pred_full - y)**2)
        history.append(loss)
    
    return w, b, history

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 2)  # 2 features
true_w = np.array([2.0, -1.5])     # True weights
true_b = 1.0                       # True bias
y = X @ true_w + true_b + 0.1 * np.random.randn(n_samples)  # Add noise

# Run SGD
w_learned, b_learned, loss_history = sgd_linear_regression(X, y, learning_rate=0.01, epochs=50)

print(f"True weights: {true_w}")
print(f"Learned weights: {w_learned}")
print(f"True bias: {true_b}")
print(f"Learned bias: {b_learned}")
print(f"Final loss: {loss_history[-1]:.6f}")

# Plot training loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('SGD Training Loss')
plt.grid(True)

# Visualize convergence in parameter space
plt.subplot(1, 2, 2)
plt.scatter(true_w[0], true_w[1], c='red', s=100, label='True weights', zorder=5)
plt.scatter(w_learned[0], w_learned[1], c='blue', s=100, label='Learned weights', zorder=5)
plt.xlabel('w₁')
plt.ylabel('w₂')
plt.title('Weight Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Code Block 3
class SimpleNeuralNetwork:
    """
    Simple neural network with backpropagation.
    
    Mathematical foundation:
    - Forward pass: z_l = W_l a_{l-1} + b_l, a_l = σ(z_l)
    - Loss: L = (1/m)∑(y_pred - y)²
    - Backward pass: δ_l = (∂L/∂a_l) ⊙ σ'(z_l)
    - Gradients: ∂L/∂W_l = δ_l a_{l-1}^T, ∂L/∂b_l = δ_l
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        # Xavier initialization: scale by 1/sqrt(input_size)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: σ(x) = 1/(1 + e^(-x))
        - Range: (0, 1)
        - Derivative: σ'(x) = σ(x)(1 - σ(x))
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))
        - Used in backpropagation for gradient computation
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward pass through the network.
        - Computes activations for each layer
        - Stores intermediate values for backpropagation
        """
        # Layer 1: z1 = XW1 + b1, a1 = σ(z1)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2: z2 = a1W2 + b2, a2 = σ(z2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        """
        Backward pass (backpropagation).
        - Computes gradients using chain rule
        - Updates parameters using gradient descent
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        # δ2 = ∂L/∂a2 = (a2 - y)
        dz2 = self.a2 - y
        
        # ∂L/∂W2 = (1/m) a1^T δ2
        dW2 = (1/m) * self.a1.T @ dz2
        # ∂L/∂b2 = (1/m) ∑δ2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        # δ1 = δ2 W2^T ⊙ σ'(a1)
        dz1 = dz2 @ self.W2.T * self.sigmoid_derivative(self.a1)
        
        # ∂L/∂W1 = (1/m) X^T δ1
        dW1 = (1/m) * X.T @ dz1
        # ∂L/∂b1 = (1/m) ∑δ1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters using gradient descent
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """
        Train the neural network.
        - Performs forward and backward passes for each epoch
        - Monitors loss for convergence
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss: MSE
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# XOR problem: non-linear classification task
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
plt.yscale('log')  # Log scale to see convergence
plt.grid(True)
plt.show()

# Code Block 4
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

# Code Block 5
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

# Code Block 6
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

# Code Block 7
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

# Code Block 8
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

# --- Advanced ML Topics: Loss Functions, Optimizers, Normalization, Attention, RL, Generative ---

# Advanced Loss Functions

def hinge_loss(y_true, y_pred):
    """Hinge loss for binary classification: max(0, 1 - y_true * y_pred)."""
    return np.maximum(0, 1 - y_true * y_pred)

def hinge_loss_derivative(y_true, y_pred):
    """Derivative of hinge loss w.r.t. predictions."""
    return np.where(y_true * y_pred < 1, -y_true, 0)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced classification."""
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return -alpha_t * (1 - p_t)**gamma * np.log(p_t)

def triplet_loss(anchor, positive, negative, margin=0.3):
    """Triplet loss for metric learning."""
    pos_dist = np.sum((anchor - positive)**2, axis=1)
    neg_dist = np.sum((anchor - negative)**2, axis=1)
    return np.maximum(0, pos_dist - neg_dist + margin)

# Advanced Optimizers

class RMSpropOptimizer:
    """RMSprop optimizer implementation."""
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = None
    
    def update(self, params, gradients):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1 - self.beta) * gradients**2
        return params - self.lr * gradients / (np.sqrt(self.v) + self.epsilon)

class AdaGradOptimizer:
    """AdaGrad optimizer implementation."""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.v = None
    
    def update(self, params, gradients):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v += gradients**2
        return params - self.lr * gradients / (np.sqrt(self.v) + self.epsilon)

# Normalization Techniques

def batch_normalization(x, gamma=1.0, beta=0.0, epsilon=1e-8):
    """Batch normalization: normalize across batch dimension."""
    mu = np.mean(x, axis=0)
    sigma2 = np.var(x, axis=0)
    x_norm = (x - mu) / np.sqrt(sigma2 + epsilon)
    return gamma * x_norm + beta

def layer_normalization(x, gamma=1.0, beta=0.0, epsilon=1e-8):
    """Layer normalization: normalize across feature dimension."""
    mu = np.mean(x, axis=1, keepdims=True)
    sigma2 = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(sigma2 + epsilon)
    return gamma * x_norm + beta

# Attention Mechanism

def self_attention(Q, K, V, d_k):
    """Simple self-attention mechanism."""
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = np.softmax(scores, axis=-1)
    return attention_weights @ V

def multi_head_attention(X, W_Q, W_K, W_V, W_O, d_k):
    """Multi-head attention implementation."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    attention_output = self_attention(Q, K, V, d_k)
    return attention_output @ W_O

# Residual Connections

def residual_block(x, F, gamma=1.0):
    """Residual block: y = F(x) + x."""
    return F(x) + gamma * x

def residual_gradient(dy_dx, dF_dx):
    """Gradient through residual connection."""
    return dy_dx * (dF_dx + 1)  # d/dx(F(x) + x) = F'(x) + 1

# Basic Reinforcement Learning

def policy_gradient_example():
    """Simple policy gradient example for a 2-armed bandit."""
    # Policy: π(a|θ) = softmax(θ)
    theta = np.array([0.0, 0.0])
    alpha = 0.1
    
    def policy(theta):
        return np.exp(theta) / np.sum(np.exp(theta))
    
    def policy_gradient(theta, action, reward):
        pi = policy(theta)
        grad = np.zeros_like(theta)
        grad[action] = reward * (1 - pi[action])
        return grad
    
    # Simulate episodes
    for episode in range(100):
        pi = policy(theta)
        action = np.random.choice(2, p=pi)
        reward = np.random.normal(action, 0.1)  # Action 1 has higher reward
        grad = policy_gradient(theta, action, reward)
        theta += alpha * grad
    
    print(f"Learned policy: {policy(theta)}")

policy_gradient_example()

# Simple Generative Model Concepts

def simple_gan_example():
    """Simple GAN-like training loop (conceptual)."""
    def generator(z):
        return z * 2 + 1  # Simple linear generator
    
    def discriminator(x):
        return 1 / (1 + np.exp(-x))  # Simple sigmoid discriminator
    
    def generator_loss(fake_outputs):
        return -np.mean(np.log(fake_outputs + 1e-8))
    
    def discriminator_loss(real_outputs, fake_outputs):
        return -np.mean(np.log(real_outputs + 1e-8) + np.log(1 - fake_outputs + 1e-8))
    
    # Training loop (simplified)
    for step in range(10):
        z = np.random.randn(10)
        fake_data = generator(z)
        fake_outputs = discriminator(fake_data)
        
        gen_loss = generator_loss(fake_outputs)
        print(f"Step {step}: Generator loss = {gen_loss:.4f}")

simple_gan_example()

# Test the advanced components
def test_advanced_components():
    """Test the advanced ML components."""
    print("=== Testing Advanced ML Components ===")
    
    # Test loss functions
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([0.8, -0.3, 0.9, -0.1])
    hinge = hinge_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    print(f"Hinge loss: {np.mean(hinge):.4f}")
    print(f"Focal loss: {np.mean(focal):.4f}")
    
    # Test optimizers
    params = np.array([1.0, 2.0])
    gradients = np.array([0.1, -0.2])
    rmsprop = RMSpropOptimizer()
    adagrad = AdaGradOptimizer()
    new_params_rms = rmsprop.update(params, gradients)
    new_params_ada = adagrad.update(params, gradients)
    print(f"RMSprop update: {new_params_rms}")
    print(f"AdaGrad update: {new_params_ada}")
    
    # Test normalization
    x = np.random.randn(5, 3)
    x_bn = batch_normalization(x)
    x_ln = layer_normalization(x)
    print(f"Batch norm shape: {x_bn.shape}")
    print(f"Layer norm shape: {x_ln.shape}")

test_advanced_components()

