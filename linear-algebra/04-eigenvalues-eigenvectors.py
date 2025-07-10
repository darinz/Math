"""
Eigenvalues and Eigenvectors Implementation

This module provides comprehensive implementations of eigenvalue and eigenvector
concepts, including numerical methods, geometric interpretations, and machine
learning applications.

Key Concepts:
- Eigenvalue/eigenvector definition and properties
- Characteristic equation and manual calculation
- Diagonalization and its applications
- Power method for iterative eigenvalue computation
- Applications in PCA, spectral clustering, and PageRank
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EigenvalueAnalysis:
    """
    Comprehensive eigenvalue and eigenvector analysis toolkit.
    
    This class provides methods for:
    - Computing eigenvalues and eigenvectors
    - Analyzing eigenvalue properties
    - Implementing the power method
    - Diagonalization
    - Machine learning applications
    """
    
    def __init__(self):
        """Initialize the eigenvalue analysis toolkit."""
        self.eigenvalues = None
        self.eigenvectors = None
        self.matrix = None
    
    def compute_eigenvalues_eigenvectors(self, A):
        """
        Compute eigenvalues and eigenvectors of a matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix to analyze
            
        Returns:
        --------
        eigenvalues : np.ndarray
            Array of eigenvalues
        eigenvectors : np.ndarray
            Matrix where each column is an eigenvector
        """
        self.matrix = A
        self.eigenvalues, self.eigenvectors = np.linalg.eig(A)
        return self.eigenvalues, self.eigenvectors
    
    def characteristic_polynomial(self, A, lambda_val):
        """
        Compute the characteristic polynomial det(A - λI).
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix
        lambda_val : float
            Value of λ to evaluate
            
        Returns:
        --------
        float : Value of the characteristic polynomial
        """
        n = A.shape[0]
        return np.linalg.det(A - lambda_val * np.eye(n))
    
    def verify_eigenvalue_properties(self, A):
        """
        Verify fundamental eigenvalue properties.
        
        Properties checked:
        - Trace equals sum of eigenvalues
        - Determinant equals product of eigenvalues
        - Reality of eigenvalues for symmetric matrices
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix to analyze
            
        Returns:
        --------
        dict : Dictionary with property verification results
        """
        eigenvalues, _ = np.linalg.eig(A)
        
        # Property 1: Trace equals sum of eigenvalues
        trace_sum = np.trace(A)
        eigenvalue_sum = np.sum(eigenvalues)
        trace_property = np.isclose(trace_sum, eigenvalue_sum, rtol=1e-10)
        
        # Property 2: Determinant equals product of eigenvalues
        det_product = np.linalg.det(A)
        eigenvalue_product = np.prod(eigenvalues)
        det_property = np.isclose(det_product, eigenvalue_product, rtol=1e-10)
        
        # Property 3: Reality for symmetric matrices
        is_symmetric = np.allclose(A, A.T)
        reality_property = True
        if is_symmetric:
            reality_property = np.allclose(eigenvalues.imag, 0, atol=1e-10)
        
        return {
            'trace_property': trace_property,
            'det_property': det_property,
            'reality_property': reality_property,
            'trace_sum': trace_sum,
            'eigenvalue_sum': eigenvalue_sum,
            'det_product': det_product,
            'eigenvalue_product': eigenvalue_product,
            'is_symmetric': is_symmetric
        }
    
    def power_method(self, A, max_iter=1000, tol=1e-10):
        """
        Implement the power method to find the dominant eigenvalue and eigenvector.
        
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
        tuple : (dominant_eigenvalue, dominant_eigenvector, iterations)
        """
        n = A.shape[0]
        
        # Initialize with random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for iteration in range(max_iter):
            # Compute Av
            v_new = A @ v
            
            # Normalize
            v_new = v_new / np.linalg.norm(v_new)
            
            # Check convergence
            if np.linalg.norm(v_new - v) < tol:
                break
                
            v = v_new
        
        # Compute eigenvalue using Rayleigh quotient
        eigenvalue = (v.T @ A @ v) / (v.T @ v)
        
        return eigenvalue, v, iteration + 1
    
    def inverse_power_method(self, A, shift=0, max_iter=1000, tol=1e-10):
        """
        Implement the inverse power method to find eigenvalues closest to a shift.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix
        shift : float
            Shift value (eigenvalue closest to this will be found)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        tuple : (eigenvalue, eigenvector, iterations)
        """
        n = A.shape[0]
        
        # Initialize with random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for iteration in range(max_iter):
            # Solve (A - shift*I)v_new = v
            try:
                v_new = np.linalg.solve(A - shift * np.eye(n), v)
            except np.linalg.LinAlgError:
                # Matrix is singular, try with small regularization
                v_new = np.linalg.solve(A - shift * np.eye(n) + 1e-12 * np.eye(n), v)
            
            # Normalize
            v_new = v_new / np.linalg.norm(v_new)
            
            # Check convergence
            if np.linalg.norm(v_new - v) < tol:
                break
                
            v = v_new
        
        # Compute eigenvalue
        eigenvalue = shift + 1 / (v.T @ np.linalg.solve(A - shift * np.eye(n), v))
        
        return eigenvalue, v, iteration + 1
    
    def diagonalize(self, A):
        """
        Diagonalize a matrix A = PDP^(-1).
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix to diagonalize
            
        Returns:
        --------
        tuple : (P, D, P_inv) where A = PDP^(-1)
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Check if matrix is diagonalizable
        if np.linalg.matrix_rank(eigenvectors) < A.shape[0]:
            raise ValueError("Matrix is not diagonalizable")
        
        P = eigenvectors
        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)
        
        return P, D, P_inv
    
    def verify_diagonalization(self, A, P, D, P_inv):
        """
        Verify that A = PDP^(-1).
        
        Parameters:
        -----------
        A : np.ndarray
            Original matrix
        P, D, P_inv : np.ndarray
            Diagonalization matrices
            
        Returns:
        --------
        bool : True if diagonalization is correct
        """
        reconstructed = P @ D @ P_inv
        return np.allclose(A, reconstructed, rtol=1e-10)
    
    def matrix_power(self, A, k, use_diagonalization=True):
        """
        Compute A^k using diagonalization if possible.
        
        Parameters:
        -----------
        A : np.ndarray
            Square matrix
        k : int
            Power to raise matrix to
        use_diagonalization : bool
            Whether to use diagonalization method
            
        Returns:
        --------
        np.ndarray : A^k
        """
        if use_diagonalization:
            try:
                P, D, P_inv = self.diagonalize(A)
                D_k = np.diag(np.diag(D) ** k)
                return P @ D_k @ P_inv
            except ValueError:
                # Matrix not diagonalizable, use direct method
                pass
        
        # Direct method
        result = np.eye(A.shape[0])
        for _ in range(k):
            result = result @ A
        return result

def demonstrate_eigenvalue_concepts():
    """
    Demonstrate fundamental eigenvalue and eigenvector concepts.
    """
    print("=" * 60)
    print("EIGENVALUE AND EIGENVECTOR CONCEPTS")
    print("=" * 60)
    
    # Create analysis object
    analyzer = EigenvalueAnalysis()
    
    # Example 1: Simple 2x2 matrix
    print("\n1. Basic Eigenvalue Computation")
    print("-" * 40)
    
    A = np.array([[4, 1], [2, 3]])
    eigenvalues, eigenvectors = analyzer.compute_eigenvalues_eigenvectors(A)
    
    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Verify eigenvector property: Av = λv
    print("\nVerifying eigenvector property Av = λv:")
    for i in range(len(eigenvalues)):
        Av = A @ eigenvectors[:, i]
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        error = np.linalg.norm(Av - lambda_v)
        print(f"Eigenvalue {i+1}: Error = {error:.2e}")
    
    # Example 2: Symmetric matrix
    print("\n\n2. Symmetric Matrix Properties")
    print("-" * 40)
    
    A_sym = np.array([[2, 1], [1, 3]])
    eigenvalues_sym, eigenvectors_sym = analyzer.compute_eigenvalues_eigenvectors(A_sym)
    
    print(f"Symmetric Matrix:\n{A_sym}")
    print(f"Eigenvalues: {eigenvalues_sym}")
    print(f"Eigenvectors are orthogonal: {np.isclose(eigenvectors_sym.T @ eigenvectors_sym, np.eye(2), atol=1e-10).all()}")
    
    # Example 3: Verify properties
    print("\n\n3. Eigenvalue Properties Verification")
    print("-" * 40)
    
    properties = analyzer.verify_eigenvalue_properties(A_sym)
    print(f"Trace property (trace = sum of eigenvalues): {properties['trace_property']}")
    print(f"Determinant property (det = product of eigenvalues): {properties['det_property']}")
    print(f"Reality property (eigenvalues real for symmetric): {properties['reality_property']}")
    print(f"Trace: {properties['trace_sum']:.6f}, Sum of eigenvalues: {properties['eigenvalue_sum']:.6f}")
    print(f"Determinant: {properties['det_product']:.6f}, Product of eigenvalues: {properties['eigenvalue_product']:.6f}")

def demonstrate_power_method():
    """
    Demonstrate the power method for finding dominant eigenvalues.
    """
    print("\n\n" + "=" * 60)
    print("POWER METHOD IMPLEMENTATION")
    print("=" * 60)
    
    analyzer = EigenvalueAnalysis()
    
    # Create test matrix
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    
    print(f"Test Matrix A:\n{A}")
    
    # Exact eigenvalues
    exact_eigenvalues, _ = np.linalg.eig(A)
    print(f"\nExact eigenvalues: {exact_eigenvalues}")
    
    # Power method for dominant eigenvalue
    dominant_eigenvalue, dominant_eigenvector, iterations = analyzer.power_method(A)
    print(f"\nPower Method Results:")
    print(f"Dominant eigenvalue: {dominant_eigenvalue:.6f}")
    print(f"Exact dominant eigenvalue: {np.max(np.abs(exact_eigenvalues)):.6f}")
    print(f"Error: {abs(dominant_eigenvalue - np.max(np.abs(exact_eigenvalues))):.2e}")
    print(f"Iterations: {iterations}")
    
    # Inverse power method for smallest eigenvalue
    smallest_eigenvalue, smallest_eigenvector, iterations = analyzer.inverse_power_method(A, shift=0)
    print(f"\nInverse Power Method Results:")
    print(f"Smallest eigenvalue: {smallest_eigenvalue:.6f}")
    print(f"Exact smallest eigenvalue: {np.min(np.abs(exact_eigenvalues)):.6f}")
    print(f"Error: {abs(smallest_eigenvalue - np.min(np.abs(exact_eigenvalues))):.2e}")
    print(f"Iterations: {iterations}")

def demonstrate_diagonalization():
    """
    Demonstrate matrix diagonalization.
    """
    print("\n\n" + "=" * 60)
    print("MATRIX DIAGONALIZATION")
    print("=" * 60)
    
    analyzer = EigenvalueAnalysis()
    
    # Example 1: Diagonalizable matrix
    A = np.array([[3, 1], [0, 2]])
    
    print(f"Matrix A:\n{A}")
    
    try:
        P, D, P_inv = analyzer.diagonalize(A)
        print(f"\nDiagonalization matrices:")
        print(f"P (eigenvectors):\n{P}")
        print(f"D (eigenvalues):\n{D}")
        print(f"P^(-1):\n{P_inv}")
        
        # Verify diagonalization
        is_correct = analyzer.verify_diagonalization(A, P, D, P_inv)
        print(f"\nDiagonalization correct: {is_correct}")
        
        # Demonstrate matrix power using diagonalization
        k = 3
        A_power_diag = analyzer.matrix_power(A, k, use_diagonalization=True)
        A_power_direct = analyzer.matrix_power(A, k, use_diagonalization=False)
        
        print(f"\nA^{k} using diagonalization:\n{A_power_diag}")
        print(f"A^{k} using direct multiplication:\n{A_power_direct}")
        print(f"Methods agree: {np.allclose(A_power_diag, A_power_direct)}")
        
    except ValueError as e:
        print(f"Matrix not diagonalizable: {e}")
    
    # Example 2: Non-diagonalizable matrix
    print(f"\n\nNon-diagonalizable matrix example:")
    A_non_diag = np.array([[1, 1], [0, 1]])
    print(f"Matrix:\n{A_non_diag}")
    
    try:
        P, D, P_inv = analyzer.diagonalize(A_non_diag)
        print("Matrix is diagonalizable")
    except ValueError:
        print("Matrix is not diagonalizable (as expected)")

def demonstrate_geometric_interpretation():
    """
    Demonstrate geometric interpretation of eigenvalues and eigenvectors.
    """
    print("\n\n" + "=" * 60)
    print("GEOMETRIC INTERPRETATION")
    print("=" * 60)
    
    # Create transformation matrix
    A = np.array([[2, 1], [1, 2]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"Transformation Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Create unit circle and transform it
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    transformed_circle = A @ unit_circle
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Original unit circle
    plt.subplot(1, 2, 1)
    plt.plot(unit_circle[0], unit_circle[1], 'b-', label='Unit Circle')
    plt.plot(eigenvectors[0, :], eigenvectors[1, :], 'ro', label='Eigenvectors')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Original Unit Circle and Eigenvectors')
    plt.legend()
    
    # Transformed circle
    plt.subplot(1, 2, 2)
    plt.plot(transformed_circle[0], transformed_circle[1], 'r-', label='Transformed Circle')
    plt.plot(eigenvalues[0] * eigenvectors[0, :], eigenvalues[0] * eigenvectors[1, :], 'go', label='Scaled Eigenvectors')
    plt.plot(eigenvalues[1] * eigenvectors[0, :], eigenvalues[1] * eigenvectors[1, :], 'mo', label='Scaled Eigenvectors')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Transformed Circle and Scaled Eigenvectors')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nGeometric interpretation:")
    print(f"- Eigenvalue 1: {eigenvalues[0]:.3f} (stretches eigenvector by this factor)")
    print(f"- Eigenvalue 2: {eigenvalues[1]:.3f} (stretches eigenvector by this factor)")
    print(f"- Eigenvectors show directions that remain unchanged under transformation")

def demonstrate_ml_applications():
    """
    Demonstrate machine learning applications of eigenvalues and eigenvectors.
    """
    print("\n\n" + "=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create correlated data
    X = np.random.randn(n_samples, n_features)
    # Add correlation between features
    X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]
    X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 2]
    
    print(f"Data shape: {X.shape}")
    print(f"Data correlation matrix:\n{np.corrcoef(X.T)}")
    
    # 1. Principal Component Analysis (PCA)
    print("\n1. Principal Component Analysis (PCA)")
    print("-" * 40)
    
    # Compute covariance matrix
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = X_centered.T @ X_centered / (n_samples - 1)
    
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Compute eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"Eigenvalues (variances): {eigenvalues}")
    print(f"Explained variance ratio: {eigenvalues / np.sum(eigenvalues)}")
    print(f"Principal components (eigenvectors):\n{eigenvectors}")
    
    # Project data onto principal components
    X_pca = X_centered @ eigenvectors
    print(f"Data projected onto principal components:\n{X_pca[:5]}")
    
    # 2. Spectral Clustering
    print("\n\n2. Spectral Clustering")
    print("-" * 40)
    
    # Create similarity matrix
    from sklearn.metrics.pairwise import rbf_kernel
    similarity_matrix = rbf_kernel(X, gamma=0.1)
    
    # Compute Laplacian matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix
    
    # Compute eigenvalues and eigenvectors of Laplacian
    laplacian_eigenvalues, laplacian_eigenvectors = np.linalg.eigh(laplacian_matrix)
    
    print(f"Laplacian eigenvalues: {laplacian_eigenvalues[:5]}")
    print(f"Number of connected components: {np.sum(laplacian_eigenvalues < 1e-10)}")
    
    # 3. PageRank Algorithm
    print("\n\n3. PageRank Algorithm")
    print("-" * 40)
    
    # Create simple web graph (adjacency matrix)
    n_pages = 5
    adjacency_matrix = np.array([
        [0, 1, 1, 0, 0],  # Page 1 links to pages 2 and 3
        [0, 0, 1, 1, 0],  # Page 2 links to pages 3 and 4
        [1, 0, 0, 0, 1],  # Page 3 links to pages 1 and 5
        [0, 0, 0, 0, 1],  # Page 4 links to page 5
        [0, 0, 0, 1, 0]   # Page 5 links to page 4
    ])
    
    # Create transition matrix
    out_degrees = np.sum(adjacency_matrix, axis=1)
    transition_matrix = adjacency_matrix / out_degrees[:, np.newaxis]
    transition_matrix = np.nan_to_num(transition_matrix, 0)  # Handle division by zero
    
    # Add damping factor
    damping = 0.85
    n_pages = transition_matrix.shape[0]
    transition_matrix = damping * transition_matrix + (1 - damping) / n_pages
    
    print(f"Transition matrix:\n{transition_matrix}")
    
    # Find dominant eigenvector (PageRank scores)
    pagerank_eigenvalue, pagerank_eigenvector = analyzer.power_method(transition_matrix)
    
    # Normalize to get probability distribution
    pagerank_scores = pagerank_eigenvector / np.sum(pagerank_eigenvector)
    
    print(f"PageRank scores: {pagerank_scores}")
    print(f"Page rankings: {np.argsort(pagerank_scores)[::-1] + 1}")

def demonstrate_advanced_concepts():
    """
    Demonstrate advanced eigenvalue concepts.
    """
    print("\n\n" + "=" * 60)
    print("ADVANCED EIGENVALUE CONCEPTS")
    print("=" * 60)
    
    analyzer = EigenvalueAnalysis()
    
    # 1. Jordan Canonical Form (for non-diagonalizable matrices)
    print("1. Jordan Canonical Form")
    print("-" * 40)
    
    # Create Jordan block
    J = np.array([[2, 1], [0, 2]])  # Jordan block with eigenvalue 2
    print(f"Jordan block:\n{J}")
    
    eigenvalues_J, eigenvectors_J = np.linalg.eig(J)
    print(f"Eigenvalues: {eigenvalues_J}")
    print(f"Eigenvectors:\n{eigenvectors_J}")
    
    # 2. Deflation method for finding multiple eigenvalues
    print("\n\n2. Deflation Method")
    print("-" * 40)
    
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    print(f"Original matrix:\n{A}")
    
    # Find dominant eigenvalue
    lambda_1, v_1, _ = analyzer.power_method(A)
    print(f"Dominant eigenvalue: {lambda_1}")
    print(f"Dominant eigenvector: {v_1}")
    
    # Deflate matrix: A' = A - λ₁v₁v₁^T
    A_deflated = A - lambda_1 * np.outer(v_1, v_1)
    print(f"Deflated matrix:\n{A_deflated}")
    
    # Find next eigenvalue
    lambda_2, v_2, _ = analyzer.power_method(A_deflated)
    print(f"Second eigenvalue: {lambda_2}")
    
    # 3. Eigenvalue sensitivity
    print("\n\n3. Eigenvalue Sensitivity")
    print("-" * 40)
    
    # Original matrix
    A_original = np.array([[2, 1], [1, 2]])
    eigenvalues_original, _ = np.linalg.eig(A_original)
    print(f"Original eigenvalues: {eigenvalues_original}")
    
    # Perturbed matrix
    perturbation = 0.01 * np.random.randn(2, 2)
    A_perturbed = A_original + perturbation
    eigenvalues_perturbed, _ = np.linalg.eig(A_perturbed)
    print(f"Perturbed eigenvalues: {eigenvalues_perturbed}")
    
    # Compute condition number
    condition_number = np.linalg.cond(A_original)
    print(f"Matrix condition number: {condition_number}")
    print(f"Eigenvalue sensitivity: {np.max(np.abs(eigenvalues_perturbed - eigenvalues_original))}")

def create_exercises():
    """
    Create exercises for eigenvalues and eigenvectors.
    """
    print("\n\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    
    analyzer = EigenvalueAnalysis()
    
    # Exercise 1: Eigenvalue properties verification
    print("Exercise 1: Eigenvalue Properties Verification")
    print("-" * 50)
    
    matrices = {
        'Symmetric': np.array([[3, 1], [1, 2]]),
        'Skew-symmetric': np.array([[0, 1], [-1, 0]]),
        'Triangular': np.array([[2, 1], [0, 3]]),
        'Random': np.random.randn(3, 3)
    }
    
    for name, matrix in matrices.items():
        print(f"\n{name} Matrix:\n{matrix}")
        properties = analyzer.verify_eigenvalue_properties(matrix)
        print(f"Trace property: {properties['trace_property']}")
        print(f"Determinant property: {properties['det_property']}")
        print(f"Reality property: {properties['reality_property']}")
    
    # Exercise 2: Power method variations
    print("\n\nExercise 2: Power Method Variations")
    print("-" * 50)
    
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    exact_eigenvalues, _ = np.linalg.eig(A)
    
    # Find dominant eigenvalue
    dominant_eigenvalue, _, iterations = analyzer.power_method(A)
    print(f"Dominant eigenvalue (power method): {dominant_eigenvalue:.6f}")
    print(f"Exact dominant eigenvalue: {np.max(np.abs(exact_eigenvalues)):.6f}")
    print(f"Iterations: {iterations}")
    
    # Find smallest eigenvalue
    smallest_eigenvalue, _, iterations = analyzer.inverse_power_method(A, shift=0)
    print(f"Smallest eigenvalue (inverse power): {smallest_eigenvalue:.6f}")
    print(f"Exact smallest eigenvalue: {np.min(np.abs(exact_eigenvalues)):.6f}")
    print(f"Iterations: {iterations}")
    
    # Exercise 3: Diagonalization testing
    print("\n\nExercise 3: Diagonalization Testing")
    print("-" * 50)
    
    test_matrices = {
        'Diagonalizable': np.array([[3, 1], [0, 2]]),
        'Symmetric': np.array([[2, 1], [1, 2]]),
        'Non-diagonalizable': np.array([[1, 1], [0, 1]])
    }
    
    for name, matrix in test_matrices.items():
        print(f"\n{name} Matrix:\n{matrix}")
        try:
            P, D, P_inv = analyzer.diagonalize(matrix)
            is_correct = analyzer.verify_diagonalization(matrix, P, D, P_inv)
            print(f"Diagonalizable: Yes")
            print(f"Reconstruction correct: {is_correct}")
        except ValueError:
            print(f"Diagonalizable: No")

def main():
    """
    Main function to run all demonstrations and exercises.
    """
    print("EIGENVALUES AND EIGENVECTORS IMPLEMENTATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_eigenvalue_concepts()
    demonstrate_power_method()
    demonstrate_diagonalization()
    demonstrate_geometric_interpretation()
    demonstrate_ml_applications()
    demonstrate_advanced_concepts()
    
    # Run exercises
    create_exercises()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key Concepts Covered:
    1. Eigenvalue/eigenvector definition and properties
    2. Characteristic equation and manual calculation
    3. Power method for iterative eigenvalue computation
    4. Matrix diagonalization and its applications
    5. Geometric interpretation of eigenvalues/eigenvectors
    6. Machine learning applications (PCA, spectral clustering, PageRank)
    7. Advanced concepts (Jordan form, deflation, sensitivity)
    
    Key Takeaways:
    - Eigenvalues reveal the fundamental structure of matrices
    - Eigenvectors provide natural coordinate systems
    - Diagonalization simplifies many matrix operations
    - Eigenvalues/eigenvectors are crucial for ML algorithms
    - Understanding these concepts is essential for advanced linear algebra
    """)

if __name__ == "__main__":
    main() 