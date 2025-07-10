"""
Matrix Decompositions Implementation

This module provides comprehensive implementations of matrix decompositions,
including LU, QR, SVD, Cholesky, and eigenvalue decompositions.

Key Concepts:
- LU decomposition with partial pivoting
- QR decomposition using Householder reflections
- Singular Value Decomposition (SVD)
- Cholesky decomposition for positive definite matrices
- Eigenvalue decomposition and power iteration
- Applications in ML (PCA, recommender systems, neural networks)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu, qr, svd, cholesky, eig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MatrixDecompositions:
    """
    Comprehensive matrix decomposition toolkit.
    
    This class provides methods for:
    - LU decomposition with partial pivoting
    - QR decomposition using Householder reflections
    - SVD decomposition and low-rank approximation
    - Cholesky decomposition for positive definite matrices
    - Eigenvalue decomposition and power iteration
    - Machine learning applications
    """
    
    def __init__(self):
        """Initialize the matrix decomposition toolkit."""
        self.decomposition_results = {}
    
    def lu_decomposition(self, A):
        """
        Perform LU decomposition with partial pivoting.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix to decompose
            
        Returns:
        --------
        tuple : (P, L, U) where PA = LU
        """
        n = len(A)
        L = np.eye(n)
        U = A.copy()
        P = np.eye(n)
        
        for k in range(n-1):
            # Find pivot
            p = k + np.argmax(np.abs(U[k:, k]))
            
            # Exchange rows
            U[k], U[p] = U[p].copy(), U[k].copy()
            L[k, :k], L[p, :k] = L[p, :k].copy(), L[k, :k].copy()
            P[k], P[p] = P[p].copy(), P[k].copy()
            
            # Eliminate
            for i in range(k+1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]
        
        return P, L, U
    
    def solve_with_lu(self, A, b):
        """
        Solve linear system Ax = b using LU decomposition.
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        np.ndarray : Solution vector x
        """
        P, L, U = self.lu_decomposition(A)
        
        # Forward substitution: Ly = Pb
        y = np.linalg.solve(L, P @ b)
        
        # Backward substitution: Ux = y
        x = np.linalg.solve(U, y)
        
        return x
    
    def householder_qr(self, A):
        """
        Perform QR decomposition using Householder reflections.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to decompose
            
        Returns:
        --------
        tuple : (Q, R) where A = QR
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy()
        
        for k in range(min(m-1, n)):
            # Householder vector
            x = R[k:, k]
            e1 = np.zeros_like(x)
            e1[0] = 1
            
            # Compute Householder vector
            v = x - np.sign(x[0]) * np.linalg.norm(x) * e1
            
            # Skip if v is zero
            if np.linalg.norm(v) < 1e-15:
                continue
            
            # Householder matrix
            H = np.eye(m)
            H[k:, k:] -= 2 * np.outer(v, v) / (v @ v)
            
            # Apply transformation
            R = H @ R
            Q = Q @ H
        
        return Q, R
    
    def solve_least_squares_qr(self, A, b):
        """
        Solve least squares problem using QR decomposition.
        
        Parameters:
        -----------
        A : np.ndarray
            Design matrix
        b : np.ndarray
            Response vector
            
        Returns:
        --------
        np.ndarray : Least squares solution
        """
        Q, R = self.householder_qr(A)
        
        # Solve Rx = Q^T b
        c = Q.T @ b
        x = np.linalg.solve(R[:A.shape[1], :A.shape[1]], c[:A.shape[1]])
        
        return x
    
    def svd_decomposition(self, A, full_matrices=False):
        """
        Perform Singular Value Decomposition.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to decompose
        full_matrices : bool
            Whether to return full matrices
            
        Returns:
        --------
        tuple : (U, s, Vt) where A = U @ diag(s) @ Vt
        """
        return np.linalg.svd(A, full_matrices=full_matrices)
    
    def low_rank_approximation(self, A, k):
        """
        Compute best rank-k approximation using SVD.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to approximate
        k : int
            Rank of approximation
            
        Returns:
        --------
        tuple : (A_k, s) where A_k is the rank-k approximation
        """
        U, s, Vt = self.svd_decomposition(A, full_matrices=False)
        
        # Truncate to rank k
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct
        A_k = U_k @ np.diag(s_k) @ Vt_k
        
        return A_k, s
    
    def cholesky_decomposition(self, A):
        """
        Perform Cholesky decomposition for positive definite matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Symmetric positive definite matrix
            
        Returns:
        --------
        np.ndarray : Lower triangular matrix L where A = LL^T
        """
        n = len(A)
        L = np.zeros_like(A)
        
        for j in range(n):
            # Diagonal element
            L[j, j] = np.sqrt(A[j, j] - np.sum(L[j, :j]**2))
            
            # Off-diagonal elements
            for i in range(j+1, n):
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
        
        return L
    
    def solve_with_cholesky(self, A, b):
        """
        Solve linear system Ax = b using Cholesky decomposition.
        
        Parameters:
        -----------
        A : np.ndarray
            Symmetric positive definite coefficient matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        np.ndarray : Solution vector x
        """
        L = self.cholesky_decomposition(A)
        
        # Solve LL^T x = b
        y = np.linalg.solve(L, b)
        x = np.linalg.solve(L.T, y)
        
        return x
    
    def power_iteration(self, A, max_iter=1000, tol=1e-10):
        """
        Find dominant eigenvalue and eigenvector using power iteration.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        tuple : (eigenvalue, eigenvector, iterations)
        """
        n = A.shape[0]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for iteration in range(max_iter):
            x_new = A @ x
            x_new = x_new / np.linalg.norm(x_new)
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        # Compute eigenvalue using Rayleigh quotient
        eigenvalue = (x.T @ A @ x) / (x.T @ x)
        
        return eigenvalue, x, iteration + 1
    
    def eigenvalue_decomposition(self, A):
        """
        Perform eigenvalue decomposition for diagonalizable matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix
            
        Returns:
        --------
        tuple : (eigenvalues, eigenvectors)
        """
        return np.linalg.eig(A)

class DecompositionAnalysis:
    """
    Class for analyzing and comparing different matrix decompositions.
    """
    
    def __init__(self):
        """Initialize the decomposition analysis toolkit."""
        self.md = MatrixDecompositions()
    
    def compare_decompositions(self, A):
        """
        Compare different decompositions for a matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to analyze
            
        Returns:
        --------
        dict : Comparison results
        """
        results = {}
        
        # LU decomposition
        try:
            P, L, U = self.md.lu_decomposition(A)
            lu_error = np.linalg.norm(P @ A - L @ U)
            results['LU'] = {
                'success': True,
                'error': lu_error,
                'P': P, 'L': L, 'U': U
            }
        except Exception as e:
            results['LU'] = {'success': False, 'error': str(e)}
        
        # QR decomposition
        try:
            Q, R = self.md.householder_qr(A)
            qr_error = np.linalg.norm(A - Q @ R)
            orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
            results['QR'] = {
                'success': True,
                'error': qr_error,
                'orthogonality_error': orthogonality_error,
                'Q': Q, 'R': R
            }
        except Exception as e:
            results['QR'] = {'success': False, 'error': str(e)}
        
        # SVD decomposition
        try:
            U, s, Vt = self.md.svd_decomposition(A)
            svd_error = np.linalg.norm(A - U @ np.diag(s) @ Vt)
            results['SVD'] = {
                'success': True,
                'error': svd_error,
                'singular_values': s,
                'rank': np.sum(s > 1e-10),
                'U': U, 's': s, 'Vt': Vt
            }
        except Exception as e:
            results['SVD'] = {'success': False, 'error': str(e)}
        
        # Cholesky decomposition (only for symmetric positive definite)
        try:
            if np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0):
                L = self.md.cholesky_decomposition(A)
                cholesky_error = np.linalg.norm(A - L @ L.T)
                results['Cholesky'] = {
                    'success': True,
                    'error': cholesky_error,
                    'L': L
                }
            else:
                results['Cholesky'] = {'success': False, 'error': 'Not symmetric positive definite'}
        except Exception as e:
            results['Cholesky'] = {'success': False, 'error': str(e)}
        
        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = self.md.eigenvalue_decomposition(A)
            # Check if matrix is diagonalizable
            if np.linalg.matrix_rank(eigenvectors) == A.shape[0]:
                eig_error = np.linalg.norm(A - eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors))
                results['Eigenvalue'] = {
                    'success': True,
                    'error': eig_error,
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors
                }
            else:
                results['Eigenvalue'] = {'success': False, 'error': 'Matrix not diagonalizable'}
        except Exception as e:
            results['Eigenvalue'] = {'success': False, 'error': str(e)}
        
        return results
    
    def analyze_conditioning(self, A):
        """
        Analyze the conditioning of different decompositions.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to analyze
            
        Returns:
        --------
        dict : Conditioning analysis
        """
        analysis = {}
        
        # Original matrix conditioning
        analysis['original'] = {
            'condition_number': np.linalg.cond(A),
            'rank': np.linalg.matrix_rank(A),
            'determinant': np.linalg.det(A) if A.shape[0] == A.shape[1] else None
        }
        
        # SVD analysis
        try:
            U, s, Vt = self.md.svd_decomposition(A)
            analysis['SVD'] = {
                'singular_values': s,
                'condition_number': s[0] / s[-1] if len(s) > 1 else float('inf'),
                'rank': np.sum(s > 1e-10),
                'effective_rank': np.sum(s > s[0] * 1e-10)
            }
        except Exception as e:
            analysis['SVD'] = {'error': str(e)}
        
        # QR analysis
        try:
            Q, R = self.md.householder_qr(A)
            analysis['QR'] = {
                'R_condition_number': np.linalg.cond(R),
                'orthogonality_error': np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
            }
        except Exception as e:
            analysis['QR'] = {'error': str(e)}
        
        return analysis

class MLApplications:
    """
    Class for machine learning applications of matrix decompositions.
    """
    
    def __init__(self):
        """Initialize the ML applications toolkit."""
        self.md = MatrixDecompositions()
    
    def pca_implementation(self, X, n_components=None):
        """
        Implement PCA using SVD.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
        n_components : int, optional
            Number of components to keep
            
        Returns:
        --------
        dict : PCA results
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # SVD decomposition
        U, s, Vt = self.md.svd_decomposition(X_centered, full_matrices=False)
        
        # Compute explained variance
        explained_variance = s**2 / (len(X) - 1)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        
        # Determine number of components
        if n_components is None:
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Project data
        X_pca = X_centered @ Vt.T[:, :n_components]
        
        return {
            'X_pca': X_pca,
            'components': Vt.T[:, :n_components],
            'explained_variance': explained_variance[:n_components],
            'explained_variance_ratio': explained_variance_ratio[:n_components],
            'singular_values': s[:n_components],
            'n_components': n_components,
            'cumulative_variance': np.cumsum(explained_variance_ratio)
        }
    
    def matrix_factorization_recommender(self, R, k, max_iter=100, learning_rate=0.01, lambda_reg=0.1):
        """
        Implement matrix factorization for recommender systems.
        
        Parameters:
        -----------
        R : np.ndarray
            Rating matrix (NaN for missing ratings)
        k : int
            Number of latent factors
        max_iter : int
            Maximum number of iterations
        learning_rate : float
            Learning rate for gradient descent
        lambda_reg : float
            Regularization parameter
            
        Returns:
        --------
        tuple : (U, V) user and item embeddings
        """
        m, n = R.shape
        
        # Initialize embeddings
        U = np.random.randn(m, k) * 0.1
        V = np.random.randn(n, k) * 0.1
        
        # Find non-zero ratings
        mask = ~np.isnan(R)
        observed_ratings = R[mask]
        
        for iteration in range(max_iter):
            # Compute predictions
            R_pred = U @ V.T
            R_pred_masked = R_pred[mask]
            
            # Compute error
            error = observed_ratings - R_pred_masked
            
            # Update U
            for i in range(m):
                user_ratings = mask[i, :]
                if np.any(user_ratings):
                    V_user = V[user_ratings, :]
                    error_user = error[mask[i, :]]
                    U[i] += learning_rate * (error_user @ V_user - lambda_reg * U[i])
            
            # Update V
            for j in range(n):
                item_ratings = mask[:, j]
                if np.any(item_ratings):
                    U_item = U[item_ratings, :]
                    error_item = error[mask[:, j]]
                    V[j] += learning_rate * (error_item @ U_item - lambda_reg * V[j])
            
            # Compute loss
            if iteration % 10 == 0:
                loss = np.mean(error**2) + lambda_reg * (np.sum(U**2) + np.sum(V**2))
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
        
        return U, V
    
    def ridge_regression_qr(self, X, y, lambda_reg=0.1):
        """
        Solve ridge regression using QR decomposition.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Response vector
        lambda_reg : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray : Ridge regression solution
        """
        n, p = X.shape
        
        # Augment design matrix with regularization
        X_aug = np.vstack([X, np.sqrt(lambda_reg) * np.eye(p)])
        y_aug = np.concatenate([y, np.zeros(p)])
        
        # Solve using QR decomposition
        Q, R = self.md.householder_qr(X_aug)
        c = Q.T @ y_aug
        w = np.linalg.solve(R[:p, :p], c[:p])
        
        return w
    
    def orthogonal_initialization(self, shape):
        """
        Generate orthogonal initialization for neural networks.
        
        Parameters:
        -----------
        shape : tuple
            Shape of the weight matrix
            
        Returns:
        --------
        np.ndarray : Orthogonal weight matrix
        """
        W = np.random.randn(*shape)
        U, _, Vt = self.md.svd_decomposition(W, full_matrices=False)
        return U @ Vt

def demonstrate_lu_decomposition():
    """
    Demonstrate LU decomposition and its applications.
    """
    print("=" * 60)
    print("LU DECOMPOSITION")
    print("=" * 60)
    
    md = MatrixDecompositions()
    
    # Example 1: Basic LU decomposition
    print("\n1. Basic LU Decomposition")
    print("-" * 40)
    
    A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    print(f"Matrix A:\n{A}")
    
    P, L, U = md.lu_decomposition(A)
    print(f"\nPermutation matrix P:\n{P}")
    print(f"Lower triangular L:\n{L}")
    print(f"Upper triangular U:\n{U}")
    
    # Verify decomposition
    reconstruction_error = np.linalg.norm(P @ A - L @ U)
    print(f"\nReconstruction error: {reconstruction_error:.2e}")
    
    # Example 2: Solving linear systems
    print("\n\n2. Solving Linear Systems")
    print("-" * 40)
    
    b = np.array([1, -2, 7])
    print(f"Right-hand side b: {b}")
    
    x_lu = md.solve_with_lu(A, b)
    x_direct = np.linalg.solve(A, b)
    
    print(f"Solution using LU: {x_lu}")
    print(f"Solution using direct solve: {x_direct}")
    print(f"Error: {np.linalg.norm(x_lu - x_direct):.2e}")
    
    # Example 3: Multiple right-hand sides
    print("\n\n3. Multiple Right-hand Sides")
    print("-" * 40)
    
    B = np.random.randn(3, 2)
    print(f"Multiple right-hand sides B:\n{B}")
    
    # Solve using LU decomposition
    P, L, U = md.lu_decomposition(A)
    X_lu = np.linalg.solve(U, np.linalg.solve(L, P @ B))
    
    # Solve using direct method
    X_direct = np.linalg.solve(A, B)
    
    print(f"Solutions using LU:\n{X_lu}")
    print(f"Solutions using direct solve:\n{X_direct}")
    print(f"Error: {np.linalg.norm(X_lu - X_direct):.2e}")

def demonstrate_qr_decomposition():
    """
    Demonstrate QR decomposition and least squares.
    """
    print("\n\n" + "=" * 60)
    print("QR DECOMPOSITION")
    print("=" * 60)
    
    md = MatrixDecompositions()
    
    # Example 1: Basic QR decomposition
    print("\n1. Basic QR Decomposition")
    print("-" * 40)
    
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"Matrix A:\n{A}")
    
    Q, R = md.householder_qr(A)
    print(f"\nOrthogonal matrix Q:\n{Q}")
    print(f"Upper triangular R:\n{R}")
    
    # Verify decomposition
    qr_error = np.linalg.norm(A - Q @ R)
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
    print(f"\nQR decomposition error: {qr_error:.2e}")
    print(f"Orthogonality error: {orthogonality_error:.2e}")
    
    # Example 2: Least squares problem
    print("\n\n2. Least Squares Problem")
    print("-" * 40)
    
    b = np.array([1, 2, 3])
    print(f"Right-hand side b: {b}")
    
    x_qr = md.solve_least_squares_qr(A, b)
    x_direct = np.linalg.lstsq(A, b, rcond=None)[0]
    
    print(f"Solution using QR: {x_qr}")
    print(f"Solution using direct lstsq: {x_direct}")
    print(f"Error: {np.linalg.norm(x_qr - x_direct):.2e}")
    
    # Example 3: Overdetermined system
    print("\n\n3. Overdetermined System")
    print("-" * 40)
    
    # Create overdetermined system
    A_over = np.random.randn(10, 3)
    b_over = np.random.randn(10)
    
    x_qr_over = md.solve_least_squares_qr(A_over, b_over)
    x_direct_over = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
    
    print(f"QR solution: {x_qr_over}")
    print(f"Direct solution: {x_direct_over}")
    print(f"Error: {np.linalg.norm(x_qr_over - x_direct_over):.2e}")

def demonstrate_svd_decomposition():
    """
    Demonstrate SVD decomposition and low-rank approximation.
    """
    print("\n\n" + "=" * 60)
    print("SVD DECOMPOSITION")
    print("=" * 60)
    
    md = MatrixDecompositions()
    
    # Example 1: Basic SVD
    print("\n1. Basic SVD")
    print("-" * 40)
    
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Matrix A:\n{A}")
    
    U, s, Vt = md.svd_decomposition(A)
    print(f"\nLeft singular vectors U:\n{U}")
    print(f"Singular values s: {s}")
    print(f"Right singular vectors Vt:\n{Vt}")
    
    # Verify decomposition
    svd_error = np.linalg.norm(A - U @ np.diag(s) @ Vt)
    print(f"\nSVD reconstruction error: {svd_error:.2e}")
    
    # Example 2: Low-rank approximation
    print("\n\n2. Low-Rank Approximation")
    print("-" * 40)
    
    for k in [1, 2]:
        A_k, s_k = md.low_rank_approximation(A, k)
        error = np.linalg.norm(A - A_k, 'fro')
        print(f"Rank-{k} approximation error: {error:.4f}")
        print(f"Rank-{k} matrix:\n{A_k}")
    
    # Example 3: Image compression simulation
    print("\n\n3. Image Compression Simulation")
    print("-" * 40)
    
    # Create a "low-rank" matrix (simulating an image)
    np.random.seed(42)
    true_rank = 3
    U_true = np.random.randn(20, true_rank)
    V_true = np.random.randn(20, true_rank)
    A_image = U_true @ V_true.T
    
    print(f"Original matrix rank: {np.linalg.matrix_rank(A_image)}")
    
    # Approximate with different ranks
    for k in [1, 2, 3, 5, 10]:
        A_k, _ = md.low_rank_approximation(A_image, k)
        error = np.linalg.norm(A_image - A_k, 'fro')
        compression_ratio = (k * (A_image.shape[0] + A_image.shape[1])) / (A_image.shape[0] * A_image.shape[1])
        print(f"Rank {k}: Error = {error:.4f}, Compression = {compression_ratio:.2%}")

def demonstrate_cholesky_decomposition():
    """
    Demonstrate Cholesky decomposition for positive definite matrices.
    """
    print("\n\n" + "=" * 60)
    print("CHOLESKY DECOMPOSITION")
    print("=" * 60)
    
    md = MatrixDecompositions()
    
    # Example 1: Basic Cholesky decomposition
    print("\n1. Basic Cholesky Decomposition")
    print("-" * 40)
    
    # Create positive definite matrix
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    print(f"Positive definite matrix A:\n{A}")
    
    L = md.cholesky_decomposition(A)
    print(f"\nLower triangular L:\n{L}")
    
    # Verify decomposition
    cholesky_error = np.linalg.norm(A - L @ L.T)
    print(f"\nCholesky reconstruction error: {cholesky_error:.2e}")
    
    # Example 2: Solving systems with Cholesky
    print("\n\n2. Solving Systems with Cholesky")
    print("-" * 40)
    
    b = np.array([1, 2, 3])
    print(f"Right-hand side b: {b}")
    
    x_cholesky = md.solve_with_cholesky(A, b)
    x_direct = np.linalg.solve(A, b)
    
    print(f"Solution using Cholesky: {x_cholesky}")
    print(f"Solution using direct solve: {x_direct}")
    print(f"Error: {np.linalg.norm(x_cholesky - x_direct):.2e}")
    
    # Example 3: Efficiency comparison
    print("\n\n3. Efficiency Comparison")
    print("-" * 40)
    
    import time
    
    # Create larger positive definite matrix
    n = 100
    A_large = np.random.randn(n, n)
    A_large = A_large.T @ A_large + n * np.eye(n)  # Make it positive definite
    b_large = np.random.randn(n)
    
    # Time Cholesky solve
    start = time.time()
    x_chol = md.solve_with_cholesky(A_large, b_large)
    time_chol = time.time() - start
    
    # Time direct solve
    start = time.time()
    x_dir = np.linalg.solve(A_large, b_large)
    time_dir = time.time() - start
    
    print(f"Matrix size: {n}x{n}")
    print(f"Cholesky solve time: {time_chol:.4f}s")
    print(f"Direct solve time: {time_dir:.4f}s")
    print(f"Speedup: {time_dir/time_chol:.2f}x")

def demonstrate_eigenvalue_decomposition():
    """
    Demonstrate eigenvalue decomposition and power iteration.
    """
    print("\n\n" + "=" * 60)
    print("EIGENVALUE DECOMPOSITION")
    print("=" * 60)
    
    md = MatrixDecompositions()
    
    # Example 1: Power iteration
    print("\n1. Power Iteration")
    print("-" * 40)
    
    A = np.array([[2, 1], [1, 3]])
    print(f"Matrix A:\n{A}")
    
    lambda_1, v_1, iterations = md.power_iteration(A)
    print(f"Dominant eigenvalue (power iteration): {lambda_1:.6f}")
    print(f"Dominant eigenvector: {v_1}")
    print(f"Iterations: {iterations}")
    
    # Compare with exact eigenvalues
    eigenvalues, eigenvectors = md.eigenvalue_decomposition(A)
    print(f"\nExact eigenvalues: {eigenvalues}")
    print(f"Exact eigenvectors:\n{eigenvectors}")
    
    # Example 2: Eigenvalue decomposition
    print("\n\n2. Eigenvalue Decomposition")
    print("-" * 40)
    
    # Check if matrix is diagonalizable
    if np.linalg.matrix_rank(eigenvectors) == A.shape[0]:
        print("Matrix is diagonalizable")
        
        # Verify decomposition
        A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        eig_error = np.linalg.norm(A - A_reconstructed)
        print(f"Eigenvalue decomposition error: {eig_error:.2e}")
    else:
        print("Matrix is not diagonalizable")
    
    # Example 3: Matrix powers
    print("\n\n3. Matrix Powers")
    print("-" * 40)
    
    k = 3
    A_power_direct = np.linalg.matrix_power(A, k)
    A_power_eig = eigenvectors @ np.diag(eigenvalues**k) @ np.linalg.inv(eigenvectors)
    
    print(f"A^{k} using direct computation:\n{A_power_direct}")
    print(f"A^{k} using eigenvalue decomposition:\n{A_power_eig}")
    print(f"Error: {np.linalg.norm(A_power_direct - A_power_eig):.2e}")

def demonstrate_ml_applications():
    """
    Demonstrate machine learning applications of matrix decompositions.
    """
    print("\n\n" + "=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("=" * 60)
    
    ml = MLApplications()
    
    # Example 1: PCA implementation
    print("\n1. PCA Implementation")
    print("-" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Create some linear dependencies
    X[:, 2] = 0.7 * X[:, 0] + 0.3 * X[:, 1]
    X[:, 4] = 0.5 * X[:, 0] - 0.5 * X[:, 3]
    
    print(f"Data shape: {X.shape}")
    print(f"Original rank: {np.linalg.matrix_rank(X)}")
    
    # Apply PCA
    pca_results = ml.pca_implementation(X, n_components=3)
    
    print(f"Explained variance ratio: {pca_results['explained_variance_ratio']}")
    print(f"Cumulative variance: {pca_results['cumulative_variance'][:3]}")
    print(f"Reduced data shape: {pca_results['X_pca'].shape}")
    
    # Example 2: Matrix factorization for recommender systems
    print("\n\n2. Matrix Factorization for Recommender Systems")
    print("-" * 40)
    
    # Create synthetic rating matrix
    n_users, n_items = 10, 8
    R = np.random.randint(1, 6, (n_users, n_items))
    # Add some missing ratings
    mask = np.random.random((n_users, n_items)) < 0.3
    R[mask] = np.nan
    
    print(f"Rating matrix shape: {R.shape}")
    print(f"Number of ratings: {np.sum(~np.isnan(R))}")
    
    # Apply matrix factorization
    U, V = ml.matrix_factorization_recommender(R, k=3, max_iter=50)
    
    # Predict missing ratings
    R_pred = U @ V.T
    print(f"Predicted ratings shape: {R_pred.shape}")
    print(f"Sample predictions:\n{R_pred[:3, :3]}")
    
    # Example 3: Ridge regression with QR
    print("\n\n3. Ridge Regression with QR")
    print("-" * 40)
    
    # Generate synthetic data
    n_samples, n_features = 50, 10
    X_reg = np.random.randn(n_samples, n_features)
    y_reg = X_reg @ np.random.randn(n_features) + 0.1 * np.random.randn(n_samples)
    
    # Solve ridge regression
    lambda_reg = 0.1
    w_qr = ml.ridge_regression_qr(X_reg, y_reg, lambda_reg)
    w_direct = np.linalg.solve(X_reg.T @ X_reg + lambda_reg * np.eye(n_features), X_reg.T @ y_reg)
    
    print(f"QR solution: {w_qr[:5]}...")
    print(f"Direct solution: {w_direct[:5]}...")
    print(f"Error: {np.linalg.norm(w_qr - w_direct):.2e}")
    
    # Example 4: Orthogonal initialization for neural networks
    print("\n\n4. Orthogonal Initialization")
    print("-" * 40)
    
    # Create weight matrices
    shapes = [(10, 8), (8, 6), (6, 4)]
    
    for shape in shapes:
        W_orth = ml.orthogonal_initialization(shape)
        orthogonality_error = np.linalg.norm(W_orth.T @ W_orth - np.eye(shape[1]))
        print(f"Shape {shape}: Orthogonality error = {orthogonality_error:.2e}")

def create_exercises():
    """
    Create exercises for matrix decompositions.
    """
    print("\n\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    
    md = MatrixDecompositions()
    da = DecompositionAnalysis()
    ml = MLApplications()
    
    # Exercise 1: LU decomposition
    print("Exercise 1: LU Decomposition")
    print("-" * 50)
    
    A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    P, L, U = md.lu_decomposition(A)
    
    print(f"Matrix A:\n{A}")
    print(f"PA = LU verification: {np.allclose(P @ A, L @ U)}")
    
    b = np.array([1, -2, 7])
    x = md.solve_with_lu(A, b)
    print(f"Solution to Ax = b: {x}")
    print(f"Verification: {np.allclose(A @ x, b)}")
    
    # Exercise 2: QR decomposition
    print("\n\nExercise 2: QR Decomposition")
    print("-" * 50)
    
    A_qr = np.array([[1, 2], [3, 4], [5, 6]])
    Q, R = md.householder_qr(A_qr)
    
    print(f"Matrix A:\n{A_qr}")
    print(f"A = QR verification: {np.allclose(A_qr, Q @ R)}")
    print(f"Q orthogonal verification: {np.allclose(Q.T @ Q, np.eye(Q.shape[0]))}")
    
    b_qr = np.array([1, 2, 3])
    x_qr = md.solve_least_squares_qr(A_qr, b_qr)
    print(f"Least squares solution: {x_qr}")
    
    # Exercise 3: SVD and low-rank approximation
    print("\n\nExercise 3: SVD and Low-Rank Approximation")
    print("-" * 50)
    
    A_svd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    U, s, Vt = md.svd_decomposition(A_svd)
    
    print(f"Matrix A:\n{A_svd}")
    print(f"Singular values: {s}")
    print(f"Rank: {np.sum(s > 1e-10)}")
    
    # Test different rank approximations
    for k in [1, 2]:
        A_k, s_k = md.low_rank_approximation(A_svd, k)
        error = np.linalg.norm(A_svd - A_k, 'fro')
        print(f"Rank-{k} approximation error: {error:.4f}")
    
    # Exercise 4: Cholesky decomposition
    print("\n\nExercise 4: Cholesky Decomposition")
    print("-" * 50)
    
    # Create positive definite matrix
    A_chol = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    L = md.cholesky_decomposition(A_chol)
    
    print(f"Matrix A:\n{A_chol}")
    print(f"A = LL^T verification: {np.allclose(A_chol, L @ L.T)}")
    
    b_chol = np.array([1, 2, 3])
    x_chol = md.solve_with_cholesky(A_chol, b_chol)
    print(f"Solution to Ax = b: {x_chol}")
    
    # Exercise 5: Eigenvalue decomposition
    print("\n\nExercise 5: Eigenvalue Decomposition")
    print("-" * 50)
    
    A_eig = np.array([[2, 1], [1, 3]])
    lambda_1, v_1, iterations = md.power_iteration(A_eig)
    eigenvalues, eigenvectors = md.eigenvalue_decomposition(A_eig)
    
    print(f"Matrix A:\n{A_eig}")
    print(f"Power iteration eigenvalue: {lambda_1:.6f}")
    print(f"Exact eigenvalues: {eigenvalues}")
    print(f"Iterations: {iterations}")
    
    # Exercise 6: PCA implementation
    print("\n\nExercise 6: PCA Implementation")
    print("-" * 50)
    
    # Generate data with known structure
    X_pca = np.random.randn(100, 4)
    X_pca[:, 2] = 0.7 * X_pca[:, 0] + 0.3 * X_pca[:, 1]
    
    pca_results = ml.pca_implementation(X_pca, n_components=2)
    
    print(f"Data shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca_results['explained_variance_ratio']}")
    print(f"Reduced data shape: {pca_results['X_pca'].shape}")

def main():
    """
    Main function to run all demonstrations and exercises.
    """
    print("MATRIX DECOMPOSITIONS IMPLEMENTATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_lu_decomposition()
    demonstrate_qr_decomposition()
    demonstrate_svd_decomposition()
    demonstrate_cholesky_decomposition()
    demonstrate_eigenvalue_decomposition()
    demonstrate_ml_applications()
    
    # Run exercises
    create_exercises()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key Decompositions Covered:
    1. LU decomposition with partial pivoting for solving linear systems
    2. QR decomposition using Householder reflections for least squares
    3. SVD decomposition for dimensionality reduction and matrix approximation
    4. Cholesky decomposition for positive definite matrices
    5. Eigenvalue decomposition and power iteration for diagonalizable matrices
    
    Key Applications:
    - Linear system solving with numerical stability
    - Least squares problems and overdetermined systems
    - Low-rank matrix approximation and data compression
    - Efficient algorithms for positive definite systems
    - Principal component analysis and dimensionality reduction
    - Recommender systems and matrix factorization
    - Neural network initialization and optimization
    
    Key Takeaways:
    - Matrix decompositions reveal underlying structure
    - Different decompositions are optimal for different problems
    - Numerical stability is crucial for practical applications
    - Decompositions enable efficient algorithms for large-scale problems
    - Understanding decompositions is essential for ML algorithms
    """)

if __name__ == "__main__":
    main() 