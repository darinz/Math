"""
Linear Independence and Basis Implementation

This module provides comprehensive implementations of linear independence,
basis, coordinate systems, change of basis, and Gram-Schmidt process.

Key Concepts:
- Linear independence testing and analysis
- Basis finding and coordinate systems
- Change of basis transformations
- Gram-Schmidt orthogonalization
- Applications in ML (feature selection, PCA, neural networks)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LinearIndependence:
    """
    Comprehensive linear independence analysis toolkit.
    
    This class provides methods for:
    - Testing linear independence using multiple methods
    - Finding bases for vector spaces
    - Change of basis transformations
    - Gram-Schmidt orthogonalization
    - Machine learning applications
    """
    
    def __init__(self):
        """Initialize the linear independence analysis toolkit."""
        self.vectors = None
        self.matrix = None
        self.basis = None
        self.dimension = None
    
    def test_linear_independence(self, vectors, method='rank', tol=1e-10):
        """
        Test linear independence using multiple methods.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to test
        method : str
            Method to use ('rank', 'determinant', 'gram_schmidt')
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        dict : Results of independence test
        """
        if not vectors:
            return {'independent': True, 'rank': 0, 'method': method}
        
        # Convert to matrix
        matrix = np.column_stack(vectors)
        n_vectors = len(vectors)
        
        results = {}
        
        if method == 'rank':
            # Method 1: Rank test
            rank = np.linalg.matrix_rank(matrix, tol=tol)
            independent = rank == n_vectors
            results = {
                'independent': independent,
                'rank': rank,
                'method': 'rank',
                'matrix': matrix
            }
        
        elif method == 'determinant':
            # Method 2: Determinant test (only for square matrices)
            if matrix.shape[0] == matrix.shape[1]:
                det = np.linalg.det(matrix)
                independent = abs(det) > tol
                results = {
                    'independent': independent,
                    'determinant': det,
                    'method': 'determinant',
                    'matrix': matrix
                }
            else:
                raise ValueError("Determinant method requires square matrix")
        
        elif method == 'gram_schmidt':
            # Method 3: Gram-Schmidt test
            orthogonal_vectors = self.gram_schmidt(vectors)
            # Check if any vector became zero (indicating dependence)
            zero_vectors = sum(1 for v in orthogonal_vectors if np.allclose(v, 0, atol=tol))
            independent = zero_vectors == 0
            results = {
                'independent': independent,
                'zero_vectors': zero_vectors,
                'method': 'gram_schmidt',
                'orthogonal_vectors': orthogonal_vectors
            }
        
        return results
    
    def comprehensive_independence_test(self, vectors, tol=1e-10):
        """
        Perform comprehensive independence test using all methods.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to test
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        dict : Comprehensive test results
        """
        results = {}
        
        # Test with all methods
        for method in ['rank', 'gram_schmidt']:
            try:
                results[method] = self.test_linear_independence(vectors, method, tol)
            except Exception as e:
                results[method] = {'error': str(e)}
        
        # Add determinant test if matrix is square
        matrix = np.column_stack(vectors)
        if matrix.shape[0] == matrix.shape[1]:
            try:
                results['determinant'] = self.test_linear_independence(vectors, 'determinant', tol)
            except Exception as e:
                results['determinant'] = {'error': str(e)}
        
        # Overall result
        all_independent = all(
            result.get('independent', False) 
            for result in results.values() 
            if 'independent' in result
        )
        
        results['overall'] = {
            'independent': all_independent,
            'methods_agree': len([r for r in results.values() if 'independent' in r]) > 1
        }
        
        return results
    
    def find_basis(self, vectors, tol=1e-10):
        """
        Find a basis for the span of a set of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        tuple : (basis_vectors, dimension)
        """
        if not vectors:
            return [], 0
        
        # Convert to matrix
        matrix = np.column_stack(vectors)
        
        # Use QR decomposition with pivoting
        Q, R, P = np.linalg.qr(matrix, mode='full', pivoting=True)
        
        # Find rank
        rank = np.linalg.matrix_rank(matrix, tol=tol)
        
        # Return first 'rank' columns as basis
        basis = []
        for i in range(rank):
            basis.append(matrix[:, P[i]])
        
        return basis, rank
    
    def change_of_basis_matrix(self, basis_old, basis_new):
        """
        Find the change of basis matrix from basis_old to basis_new.
        
        Parameters:
        -----------
        basis_old : list of np.ndarray
            Old basis vectors
        basis_new : list of np.ndarray
            New basis vectors
            
        Returns:
        --------
        np.ndarray : Change of basis matrix P
        """
        # Convert bases to matrices
        B_old = np.column_stack(basis_old)
        B_new = np.column_stack(basis_new)
        
        # P satisfies: B_new = B_old * P
        # So P = B_old^(-1) * B_new
        P = np.linalg.solve(B_old, B_new)
        
        return P
    
    def transform_coordinates(self, vector, basis_old, basis_new):
        """
        Transform vector coordinates from basis_old to basis_new.
        
        Parameters:
        -----------
        vector : np.ndarray
            Vector in basis_old coordinates
        basis_old : list of np.ndarray
            Old basis
        basis_new : list of np.ndarray
            New basis
            
        Returns:
        --------
        np.ndarray : Vector in basis_new coordinates
        """
        P = self.change_of_basis_matrix(basis_old, basis_new)
        return np.linalg.solve(P, vector)
    
    def gram_schmidt(self, vectors, normalize=True):
        """
        Apply Gram-Schmidt orthogonalization to a set of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors
        normalize : bool
            Whether to normalize the resulting vectors
            
        Returns:
        --------
        list : Orthogonal (or orthonormal) vectors
        """
        if not vectors:
            return []
        
        orthogonal_vectors = []
        
        for i, v in enumerate(vectors):
            # Start with the original vector
            u = v.copy()
            
            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                proj = self._projection(v, orthogonal_vectors[j])
                u = u - proj
            
            # Add to orthogonal set (if not zero)
            if not np.allclose(u, 0):
                if normalize:
                    u = u / np.linalg.norm(u)
                orthogonal_vectors.append(u)
        
        return orthogonal_vectors
    
    def _projection(self, v, u):
        """
        Compute the projection of v onto u.
        
        Parameters:
        -----------
        v, u : np.ndarray
            Vectors
            
        Returns:
        --------
        np.ndarray : Projection of v onto u
        """
        return (np.dot(v, u) / np.dot(u, u)) * u
    
    def modified_gram_schmidt(self, vectors, normalize=True):
        """
        Apply modified Gram-Schmidt for better numerical stability.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors
        normalize : bool
            Whether to normalize the resulting vectors
            
        Returns:
        --------
        list : Orthogonal (or orthonormal) vectors
        """
        if not vectors:
            return []
        
        # Convert to matrix for easier manipulation
        A = np.column_stack(vectors)
        m, n = A.shape
        
        # Modified Gram-Schmidt
        for j in range(n):
            # Normalize the j-th column
            if normalize:
                A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
            
            # Orthogonalize remaining columns
            for k in range(j + 1, n):
                A[:, k] = A[:, k] - np.dot(A[:, j], A[:, k]) * A[:, j]
        
        # Convert back to list of vectors
        orthogonal_vectors = [A[:, i] for i in range(n)]
        
        return orthogonal_vectors
    
    def check_orthogonality(self, vectors, tol=1e-10):
        """
        Check orthogonality of a set of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to check
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        dict : Orthogonality analysis results
        """
        if not vectors:
            return {'orthogonal': True, 'max_error': 0}
        
        n = len(vectors)
        errors = []
        
        for i in range(n):
            for j in range(i + 1, n):
                dot_product = np.dot(vectors[i], vectors[j])
                errors.append(abs(dot_product))
        
        max_error = max(errors) if errors else 0
        orthogonal = max_error < tol
        
        return {
            'orthogonal': orthogonal,
            'max_error': max_error,
            'all_errors': errors
        }

class CoordinateSystem:
    """
    Class for working with coordinate systems and basis transformations.
    """
    
    def __init__(self):
        """Initialize the coordinate system toolkit."""
        self.li = LinearIndependence()
    
    def standard_basis(self, n):
        """
        Generate the standard basis for R^n.
        
        Parameters:
        -----------
        n : int
            Dimension of the space
            
        Returns:
        --------
        list : Standard basis vectors
        """
        basis = []
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            basis.append(e_i)
        return basis
    
    def vector_to_coordinates(self, vector, basis):
        """
        Find coordinates of a vector with respect to a basis.
        
        Parameters:
        -----------
        vector : np.ndarray
            Vector to find coordinates for
        basis : list of np.ndarray
            Basis vectors
            
        Returns:
        --------
        np.ndarray : Coordinates of the vector
        """
        # Solve the system: vector = c1*b1 + c2*b2 + ... + cn*bn
        basis_matrix = np.column_stack(basis)
        coordinates = np.linalg.solve(basis_matrix, vector)
        return coordinates
    
    def coordinates_to_vector(self, coordinates, basis):
        """
        Convert coordinates back to vector.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Coordinates of the vector
        basis : list of np.ndarray
            Basis vectors
            
        Returns:
        --------
        np.ndarray : Vector
        """
        return np.sum([c * b for c, b in zip(coordinates, basis)], axis=0)
    
    def visualize_basis_change(self, basis_old, basis_new, vectors=None):
        """
        Visualize change of basis in 2D or 3D.
        
        Parameters:
        -----------
        basis_old : list of np.ndarray
            Old basis
        basis_new : list of np.ndarray
            New basis
        vectors : list of np.ndarray, optional
            Vectors to transform and visualize
        """
        dim = len(basis_old[0])
        
        if dim == 2:
            self._visualize_2d_basis_change(basis_old, basis_new, vectors)
        elif dim == 3:
            self._visualize_3d_basis_change(basis_old, basis_new, vectors)
        else:
            print(f"Cannot visualize {dim}-dimensional basis change")
    
    def _visualize_2d_basis_change(self, basis_old, basis_new, vectors=None):
        """Visualize 2D basis change."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot old basis
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.grid(True)
        ax1.set_title("Old Basis")
        
        colors = ['red', 'blue']
        for i, (v, color) in enumerate(zip(basis_old, colors)):
            ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                      color=color, label=f'Old basis {i+1}')
        ax1.legend()
        
        # Plot new basis
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.grid(True)
        ax2.set_title("New Basis")
        
        for i, (v, color) in enumerate(zip(basis_new, colors)):
            ax2.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                      color=color, label=f'New basis {i+1}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

class MLApplications:
    """
    Class for machine learning applications of linear independence.
    """
    
    def __init__(self):
        """Initialize the ML applications toolkit."""
        self.li = LinearIndependence()
    
    def feature_selection_analysis(self, X, method='correlation', threshold=0.8):
        """
        Analyze feature independence and select features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        method : str
            Selection method ('correlation', 'variance', 'rank')
        threshold : float
            Threshold for selection
            
        Returns:
        --------
        dict : Feature selection results
        """
        n_samples, n_features = X.shape
        
        if method == 'correlation':
            # Correlation-based selection
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
            
            # Find highly correlated features
            high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
            redundant_features = set()
            
            for i, j in zip(*high_corr_pairs):
                if i < j:  # Avoid duplicates
                    redundant_features.add(j)
            
            selected_features = [i for i in range(n_features) if i not in redundant_features]
            
            return {
                'method': 'correlation',
                'selected_features': selected_features,
                'redundant_features': list(redundant_features),
                'correlation_matrix': corr_matrix
            }
        
        elif method == 'rank':
            # Rank-based selection
            rank = np.linalg.matrix_rank(X, tol=1e-10)
            basis, _ = self.li.find_basis(X.T)
            
            return {
                'method': 'rank',
                'rank': rank,
                'effective_features': rank,
                'redundant_features': n_features - rank,
                'basis': basis
            }
        
        elif method == 'variance':
            # Variance-based selection
            variances = np.var(X, axis=0)
            selected_features = np.where(variances > threshold)[0]
            
            return {
                'method': 'variance',
                'selected_features': selected_features,
                'variances': variances,
                'threshold': threshold
            }
    
    def pca_analysis(self, X, n_components=None):
        """
        Perform PCA analysis and find principal components.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix
        n_components : int, optional
            Number of components to keep
            
        Returns:
        --------
        dict : PCA analysis results
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        # Select number of components
        if n_components is None:
            # Keep components that explain 95% of variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Project data
        X_pca = X_centered @ eigenvectors[:, :n_components]
        
        return {
            'X_pca': X_pca,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': explained_variance_ratio,
            'n_components': n_components,
            'cumulative_variance': np.cumsum(explained_variance_ratio)
        }
    
    def neural_network_analysis(self, weight_matrices):
        """
        Analyze linear independence in neural network weight matrices.
        
        Parameters:
        -----------
        weight_matrices : list of np.ndarray
            List of weight matrices
            
        Returns:
        --------
        dict : Analysis results
        """
        results = {}
        
        for i, W in enumerate(weight_matrices):
            # Compute rank
            rank = np.linalg.matrix_rank(W)
            
            # Compute condition number
            condition_number = np.linalg.cond(W)
            
            # Check if matrix is well-conditioned
            well_conditioned = condition_number < 1000
            
            # Find singular values
            singular_values = np.linalg.svd(W, compute_uv=False)
            
            results[f'layer_{i}'] = {
                'shape': W.shape,
                'rank': rank,
                'full_rank': rank == min(W.shape),
                'condition_number': condition_number,
                'well_conditioned': well_conditioned,
                'singular_values': singular_values,
                'rank_deficiency': min(W.shape) - rank
            }
        
        return results

def demonstrate_linear_independence():
    """
    Demonstrate fundamental linear independence concepts.
    """
    print("=" * 60)
    print("LINEAR INDEPENDENCE CONCEPTS")
    print("=" * 60)
    
    li = LinearIndependence()
    
    # Example 1: Independent vectors
    print("\n1. Independent Vectors")
    print("-" * 40)
    
    independent_vectors = [
        np.array([1, 0]),
        np.array([0, 1])
    ]
    
    results = li.comprehensive_independence_test(independent_vectors)
    print(f"Vectors: {[v.tolist() for v in independent_vectors]}")
    print(f"Independent: {results['overall']['independent']}")
    print(f"Methods agree: {results['overall']['methods_agree']}")
    
    # Example 2: Dependent vectors
    print("\n\n2. Dependent Vectors")
    print("-" * 40)
    
    dependent_vectors = [
        np.array([1, 2]),
        np.array([2, 4])  # 2 * [1, 2]
    ]
    
    results = li.comprehensive_independence_test(dependent_vectors)
    print(f"Vectors: {[v.tolist() for v in dependent_vectors]}")
    print(f"Independent: {results['overall']['independent']}")
    print(f"Methods agree: {results['overall']['methods_agree']}")
    
    # Example 3: 3D vectors
    print("\n\n3. 3D Vectors")
    print("-" * 40)
    
    vectors_3d = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([1, 1, 0])  # Dependent on first two
    ]
    
    results = li.comprehensive_independence_test(vectors_3d)
    print(f"Vectors: {[v.tolist() for v in vectors_3d]}")
    print(f"Independent: {results['overall']['independent']}")
    print(f"Rank: {results['rank']['rank']}")
    print(f"Dimension of span: {results['rank']['rank']}")

def demonstrate_basis_and_coordinates():
    """
    Demonstrate basis finding and coordinate systems.
    """
    print("\n\n" + "=" * 60)
    print("BASIS AND COORDINATE SYSTEMS")
    print("=" * 60)
    
    li = LinearIndependence()
    cs = CoordinateSystem()
    
    # Example 1: Standard basis
    print("\n1. Standard Basis")
    print("-" * 40)
    
    standard_basis = cs.standard_basis(3)
    print(f"Standard basis for R³:")
    for i, e in enumerate(standard_basis):
        print(f"e{i+1} = {e}")
    
    # Example 2: Custom basis
    print("\n\n2. Custom Basis")
    print("-" * 40)
    
    custom_basis = [
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1])
    ]
    
    # Check if it's a basis
    results = li.comprehensive_independence_test(custom_basis)
    print(f"Custom basis independent: {results['overall']['independent']}")
    
    if results['overall']['independent']:
        print("This is a valid basis for R³")
        
        # Find coordinates of a vector
        vector = np.array([3, 2, 1])
        coordinates = cs.vector_to_coordinates(vector, custom_basis)
        print(f"Vector {vector} in custom basis coordinates: {coordinates}")
        
        # Verify reconstruction
        reconstructed = cs.coordinates_to_vector(coordinates, custom_basis)
        print(f"Reconstructed vector: {reconstructed}")
        print(f"Reconstruction error: {np.linalg.norm(vector - reconstructed):.2e}")
    
    # Example 3: Change of basis
    print("\n\n3. Change of Basis")
    print("-" * 40)
    
    basis_old = [np.array([1, 0]), np.array([0, 1])]
    basis_new = [np.array([1, 1]), np.array([1, -1])]
    
    # Find change of basis matrix
    P = li.change_of_basis_matrix(basis_old, basis_new)
    print(f"Change of basis matrix:\n{P}")
    
    # Transform a vector
    vector = np.array([3, 4])
    new_coordinates = li.transform_coordinates(vector, basis_old, basis_new)
    print(f"Vector {vector} in new basis: {new_coordinates}")
    
    # Verify transformation
    old_coordinates = li.transform_coordinates(new_coordinates, basis_new, basis_old)
    print(f"Back to old basis: {old_coordinates}")
    print(f"Transformation error: {np.linalg.norm(vector - old_coordinates):.2e}")

def demonstrate_gram_schmidt():
    """
    Demonstrate Gram-Schmidt orthogonalization.
    """
    print("\n\n" + "=" * 60)
    print("GRAM-SCHMIDT ORTHOGONALIZATION")
    print("=" * 60)
    
    li = LinearIndependence()
    
    # Example 1: Basic Gram-Schmidt
    print("\n1. Basic Gram-Schmidt")
    print("-" * 40)
    
    vectors = [
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1])
    ]
    
    print("Original vectors:")
    for i, v in enumerate(vectors):
        print(f"v{i+1} = {v}")
    
    # Apply Gram-Schmidt
    orthogonal_vectors = li.gram_schmidt(vectors)
    print(f"\nOrthogonal vectors:")
    for i, v in enumerate(orthogonal_vectors):
        print(f"u{i+1} = {v}")
    
    # Check orthogonality
    orthogonality = li.check_orthogonality(orthogonal_vectors)
    print(f"\nOrthogonality check:")
    print(f"Orthogonal: {orthogonality['orthogonal']}")
    print(f"Maximum error: {orthogonality['max_error']:.2e}")
    
    # Example 2: Modified Gram-Schmidt
    print("\n\n2. Modified Gram-Schmidt")
    print("-" * 40)
    
    modified_orthogonal = li.modified_gram_schmidt(vectors)
    modified_orthogonality = li.check_orthogonality(modified_orthogonal)
    
    print(f"Modified Gram-Schmidt orthogonal: {modified_orthogonality['orthogonal']}")
    print(f"Maximum error: {modified_orthogonality['max_error']:.2e}")
    
    # Compare methods
    print(f"\nComparison:")
    print(f"Standard Gram-Schmidt max error: {orthogonality['max_error']:.2e}")
    print(f"Modified Gram-Schmidt max error: {modified_orthogonality['max_error']:.2e}")
    
    # Example 3: Numerical stability test
    print("\n\n3. Numerical Stability Test")
    print("-" * 40)
    
    # Create nearly dependent vectors
    nearly_dependent = [
        np.array([1, 0]),
        np.array([1, 1e-10])  # Nearly parallel
    ]
    
    results = li.comprehensive_independence_test(nearly_dependent)
    print(f"Nearly dependent vectors independent: {results['overall']['independent']}")
    
    # Apply Gram-Schmidt
    orthogonal_nearly = li.gram_schmidt(nearly_dependent)
    orthogonality_nearly = li.check_orthogonality(orthogonal_nearly)
    
    print(f"Gram-Schmidt orthogonal: {orthogonality_nearly['orthogonal']}")
    print(f"Maximum error: {orthogonality_nearly['max_error']:.2e}")

def demonstrate_ml_applications():
    """
    Demonstrate machine learning applications.
    """
    print("\n\n" + "=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("=" * 60)
    
    ml = MLApplications()
    
    # Example 1: Feature selection
    print("\n1. Feature Selection Analysis")
    print("-" * 40)
    
    # Generate synthetic data with known dependencies
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Create linear dependency
    X[:, 2] = 2 * X[:, 0] + 3 * X[:, 1]
    # Create another dependency
    X[:, 4] = 0.5 * X[:, 0] - 0.5 * X[:, 1]
    
    print(f"Data shape: {X.shape}")
    print(f"Feature correlation matrix:\n{np.corrcoef(X.T)}")
    
    # Analyze feature independence
    rank_analysis = ml.feature_selection_analysis(X, method='rank')
    correlation_analysis = ml.feature_selection_analysis(X, method='correlation', threshold=0.8)
    
    print(f"\nRank analysis:")
    print(f"Matrix rank: {rank_analysis['rank']}")
    print(f"Effective features: {rank_analysis['effective_features']}")
    print(f"Redundant features: {rank_analysis['redundant_features']}")
    
    print(f"\nCorrelation analysis:")
    print(f"Selected features: {correlation_analysis['selected_features']}")
    print(f"Redundant features: {correlation_analysis['redundant_features']}")
    
    # Example 2: PCA analysis
    print("\n\n2. PCA Analysis")
    print("-" * 40)
    
    # Use iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    print(f"Iris data shape: {X_iris.shape}")
    
    # Apply PCA
    pca_results = ml.pca_analysis(X_iris, n_components=2)
    
    print(f"Explained variance ratio: {pca_results['explained_variance_ratio'][:5]}")
    print(f"Cumulative variance: {pca_results['cumulative_variance'][:5]}")
    print(f"Number of components for 95% variance: {pca_results['n_components']}")
    
    # Visualize PCA results
    plt.figure(figsize=(12, 5))
    
    # Explained variance plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
             pca_results['explained_variance_ratio'], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True)
    
    # Cumulative variance plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca_results['cumulative_variance']) + 1),
             pca_results['cumulative_variance'], 'ro-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Example 3: Neural network analysis
    print("\n\n3. Neural Network Weight Analysis")
    print("-" * 40)
    
    # Simulate neural network weight matrices
    input_dim = 10
    hidden_dim = 8
    output_dim = 3
    
    # Create weight matrices
    W1 = np.random.randn(input_dim, hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim)
    
    # Analyze weight matrices
    weight_matrices = [W1, W2]
    analysis = ml.neural_network_analysis(weight_matrices)
    
    for layer_name, layer_analysis in analysis.items():
        print(f"\n{layer_name}:")
        print(f"  Shape: {layer_analysis['shape']}")
        print(f"  Rank: {layer_analysis['rank']}")
        print(f"  Full rank: {layer_analysis['full_rank']}")
        print(f"  Condition number: {layer_analysis['condition_number']:.2f}")
        print(f"  Well-conditioned: {layer_analysis['well_conditioned']}")
        print(f"  Rank deficiency: {layer_analysis['rank_deficiency']}")

def create_exercises():
    """
    Create exercises for linear independence and basis.
    """
    print("\n\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    
    li = LinearIndependence()
    cs = CoordinateSystem()
    ml = MLApplications()
    
    # Exercise 1: Linear independence testing
    print("Exercise 1: Linear Independence Testing")
    print("-" * 50)
    
    test_vectors = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9])
    ]
    
    results = li.comprehensive_independence_test(test_vectors)
    print(f"Vectors: {[v.tolist() for v in test_vectors]}")
    print(f"Independent: {results['overall']['independent']}")
    print(f"Rank: {results['rank']['rank']}")
    print(f"Dimension of span: {results['rank']['rank']}")
    
    # Exercise 2: Change of basis
    print("\n\nExercise 2: Change of Basis")
    print("-" * 50)
    
    basis_old = [np.array([1, 0]), np.array([0, 1])]
    basis_new = [np.array([1, 1]), np.array([1, -1])]
    
    P = li.change_of_basis_matrix(basis_old, basis_new)
    print(f"Change of basis matrix:\n{P}")
    
    vector = np.array([3, 4])
    new_coords = li.transform_coordinates(vector, basis_old, basis_new)
    print(f"Vector {vector} in new basis: {new_coords}")
    
    # Exercise 3: Gram-Schmidt process
    print("\n\nExercise 3: Gram-Schmidt Process")
    print("-" * 50)
    
    vectors = [
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1])
    ]
    
    orthogonal = li.gram_schmidt(vectors)
    orthogonality = li.check_orthogonality(orthogonal)
    
    print(f"Original vectors: {len(vectors)}")
    print(f"Orthogonal vectors: {len(orthogonal)}")
    print(f"Orthogonal: {orthogonality['orthogonal']}")
    print(f"Max error: {orthogonality['max_error']:.2e}")
    
    # Exercise 4: Feature selection
    print("\n\nExercise 4: Feature Selection")
    print("-" * 50)
    
    # Generate data with known dependencies
    X = np.random.randn(50, 4)
    X[:, 2] = 2 * X[:, 0] + X[:, 1]  # Linear dependency
    X[:, 3] = 0.5 * X[:, 0]  # Another dependency
    
    rank_analysis = ml.feature_selection_analysis(X, method='rank')
    print(f"Data shape: {X.shape}")
    print(f"Matrix rank: {rank_analysis['rank']}")
    print(f"Effective features: {rank_analysis['effective_features']}")
    print(f"Redundant features: {rank_analysis['redundant_features']}")
    
    # Exercise 5: PCA implementation
    print("\n\nExercise 5: PCA Implementation")
    print("-" * 50)
    
    # Generate 2D data with clear structure
    X, _ = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)
    
    pca_results = ml.pca_analysis(X, n_components=2)
    print(f"Data shape: {X.shape}")
    print(f"Explained variance ratio: {pca_results['explained_variance_ratio'][:3]}")
    print(f"Number of components for 95% variance: {pca_results['n_components']}")
    
    # Exercise 6: Neural network analysis
    print("\n\nExercise 6: Neural Network Analysis")
    print("-" * 50)
    
    # Create weight matrices with different properties
    W1_good = np.random.randn(10, 8)
    W1_bad = np.random.randn(10, 8)
    W1_bad[:, 0] = 2 * W1_bad[:, 1]  # Create dependency
    
    weight_matrices = [W1_good, W1_bad]
    analysis = ml.neural_network_analysis(weight_matrices)
    
    for i, (layer_name, layer_analysis) in enumerate(analysis.items()):
        print(f"{layer_name}:")
        print(f"  Full rank: {layer_analysis['full_rank']}")
        print(f"  Condition number: {layer_analysis['condition_number']:.2f}")
        print(f"  Well-conditioned: {layer_analysis['well_conditioned']}")

def main():
    """
    Main function to run all demonstrations and exercises.
    """
    print("LINEAR INDEPENDENCE AND BASIS IMPLEMENTATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_linear_independence()
    demonstrate_basis_and_coordinates()
    demonstrate_gram_schmidt()
    demonstrate_ml_applications()
    
    # Run exercises
    create_exercises()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key Concepts Covered:
    1. Linear independence testing using multiple methods
    2. Basis finding and coordinate systems
    3. Change of basis transformations
    4. Gram-Schmidt orthogonalization process
    5. Machine learning applications (feature selection, PCA, neural networks)
    
    Key Takeaways:
    - Linear independence is fundamental to understanding vector space structure
    - Basis provides coordinate systems for vector spaces
    - Change of basis enables different perspectives on data
    - Gram-Schmidt creates orthogonal bases for numerical stability
    - Linear independence concepts are crucial for ML algorithms
    """)

if __name__ == "__main__":
    main() 