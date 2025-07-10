"""
Vector Spaces and Subspaces Implementation

This module provides comprehensive implementations of vector space concepts,
including subspaces, span, linear independence, basis, dimension, and
applications in machine learning.

Key Concepts:
- Vector space axioms and properties
- Subspaces and subspace criteria
- Span and linear combinations
- Linear independence and dependence
- Basis and dimension
- Null space and column space
- Rank-nullity theorem
- Applications in ML (feature spaces, kernel methods, PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, orth
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VectorSpace:
    """
    Comprehensive vector space analysis toolkit.
    
    This class provides methods for:
    - Vector space operations and verification
    - Subspace testing and analysis
    - Span computation and visualization
    - Linear independence testing
    - Basis finding and dimension computation
    - Null space and column space analysis
    - Machine learning applications
    """
    
    def __init__(self):
        """Initialize the vector space analysis toolkit."""
        self.vectors = None
        self.matrix = None
        self.basis = None
        self.dimension = None
    
    def vector_addition(self, v1, v2):
        """
        Perform vector addition.
        
        Parameters:
        -----------
        v1, v2 : np.ndarray
            Vectors to add
            
        Returns:
        --------
        np.ndarray : Sum of vectors
        """
        return v1 + v2
    
    def scalar_multiplication(self, v, c):
        """
        Perform scalar multiplication.
        
        Parameters:
        -----------
        v : np.ndarray
            Vector to multiply
        c : float
            Scalar multiplier
            
        Returns:
        --------
        np.ndarray : Scaled vector
        """
        return c * v
    
    def linear_combination(self, vectors, coefficients):
        """
        Compute linear combination of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            List of vectors
        coefficients : list of float
            Coefficients for each vector
            
        Returns:
        --------
        np.ndarray : Linear combination
        """
        if len(vectors) != len(coefficients):
            raise ValueError("Number of vectors must equal number of coefficients")
        
        result = np.zeros_like(vectors[0])
        for v, c in zip(vectors, coefficients):
            result += c * v
        return result
    
    def span(self, vectors):
        """
        Compute the span of a set of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors
            
        Returns:
        --------
        tuple : (basis_vectors, dimension)
        """
        if not vectors:
            return [], 0
        
        # Convert to matrix
        matrix = np.column_stack(vectors)
        
        # Find linearly independent vectors (basis for span)
        rank = np.linalg.matrix_rank(matrix)
        basis_vectors = []
        
        # Use QR decomposition to find basis
        Q, R, P = np.linalg.qr(matrix, mode='full', pivoting=True)
        
        # Take first 'rank' columns as basis
        for i in range(rank):
            basis_vectors.append(matrix[:, P[i]])
        
        return basis_vectors, rank
    
    def is_subspace(self, vectors, zero_vector=None):
        """
        Check if a set of vectors forms a subspace.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to test
        zero_vector : np.ndarray, optional
            Zero vector of the space
            
        Returns:
        --------
        bool : True if vectors form a subspace
        """
        if not vectors:
            return False
        
        # Check if zero vector is in the set
        if zero_vector is None:
            zero_vector = np.zeros_like(vectors[0])
        
        zero_in_set = any(np.allclose(v, zero_vector) for v in vectors)
        if not zero_in_set:
            return False
        
        # Check closure under addition
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                if i != j:
                    sum_vector = v1 + v2
                    if not any(np.allclose(sum_vector, v) for v in vectors):
                        return False
        
        # Check closure under scalar multiplication
        for v in vectors:
            for c in [-1, 2]:  # Test with -1 and 2
                scaled_vector = c * v
                if not any(np.allclose(scaled_vector, v_test) for v_test in vectors):
                    return False
        
        return True
    
    def is_linearly_independent(self, vectors, tol=1e-10):
        """
        Check if a set of vectors is linearly independent.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to test
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        bool : True if vectors are linearly independent
        """
        if not vectors:
            return True
        
        # Convert to matrix
        matrix = np.column_stack(vectors)
        
        # Check rank
        rank = np.linalg.matrix_rank(matrix, tol=tol)
        
        return rank == len(vectors)
    
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
        list : Basis vectors
        """
        if not vectors:
            return []
        
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
        
        return basis
    
    def compute_dimension(self, vectors):
        """
        Compute the dimension of the span of a set of vectors.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors
            
        Returns:
        --------
        int : Dimension of the span
        """
        if not vectors:
            return 0
        
        # Convert to matrix and compute rank
        matrix = np.column_stack(vectors)
        return np.linalg.matrix_rank(matrix)
    
    def null_space(self, A, tol=1e-10):
        """
        Compute the null space of a matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        np.ndarray : Basis for null space (columns are basis vectors)
        """
        return null_space(A, rcond=tol)
    
    def column_space(self, A, tol=1e-10):
        """
        Compute a basis for the column space of a matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix
        tol : float
            Tolerance for numerical comparisons
            
        Returns:
        --------
        np.ndarray : Basis for column space (columns are basis vectors)
        """
        return orth(A, rcond=tol)
    
    def rank_nullity_theorem(self, A):
        """
        Verify the rank-nullity theorem for a matrix.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix
            
        Returns:
        --------
        dict : Dictionary with rank, nullity, and verification
        """
        m, n = A.shape
        rank = np.linalg.matrix_rank(A)
        nullity = n - rank
        
        # Compute null space dimension
        null_basis = self.null_space(A)
        nullity_computed = null_basis.shape[1] if null_basis.size > 0 else 0
        
        return {
            'rank': rank,
            'nullity': nullity,
            'nullity_computed': nullity_computed,
            'n_columns': n,
            'theorem_holds': rank + nullity == n,
            'computed_agrees': nullity == nullity_computed
        }

class SubspaceAnalyzer:
    """
    Specialized class for analyzing subspaces and their properties.
    """
    
    def __init__(self):
        """Initialize the subspace analyzer."""
        self.vector_space = VectorSpace()
    
    def analyze_subspace(self, vectors, description=""):
        """
        Comprehensive analysis of a set of vectors as a potential subspace.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Set of vectors to analyze
        description : str
            Description of the subspace
            
        Returns:
        --------
        dict : Analysis results
        """
        print(f"\nSubspace Analysis: {description}")
        print("-" * 50)
        
        # Basic properties
        n_vectors = len(vectors)
        if n_vectors == 0:
            print("Empty set - not a subspace")
            return {}
        
        # Check if zero vector is included
        zero_vector = np.zeros_like(vectors[0])
        has_zero = any(np.allclose(v, zero_vector) for v in vectors)
        print(f"Contains zero vector: {has_zero}")
        
        # Check linear independence
        is_independent = self.vector_space.is_linearly_independent(vectors)
        print(f"Linearly independent: {is_independent}")
        
        # Compute span and dimension
        basis, dimension = self.vector_space.span(vectors)
        print(f"Dimension: {dimension}")
        print(f"Number of basis vectors: {len(basis)}")
        
        # Check subspace properties
        is_subspace = self.vector_space.is_subspace(vectors, zero_vector)
        print(f"Forms a subspace: {is_subspace}")
        
        return {
            'n_vectors': n_vectors,
            'has_zero': has_zero,
            'is_independent': is_independent,
            'dimension': dimension,
            'n_basis_vectors': len(basis),
            'is_subspace': is_subspace,
            'basis': basis
        }
    
    def visualize_subspace(self, vectors, title="Subspace Visualization"):
        """
        Visualize a subspace in 2D or 3D.
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            Vectors to visualize
        title : str
            Plot title
        """
        if not vectors:
            print("No vectors to visualize")
            return
        
        dim = len(vectors[0])
        
        if dim == 2:
            self._visualize_2d_subspace(vectors, title)
        elif dim == 3:
            self._visualize_3d_subspace(vectors, title)
        else:
            print(f"Cannot visualize {dim}-dimensional subspace")
    
    def _visualize_2d_subspace(self, vectors, title):
        """Visualize 2D subspace."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original vectors
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.grid(True)
        ax1.set_title("Original Vectors")
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(vectors)))
        for i, v in enumerate(vectors):
            ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                      color=colors[i], label=f'v{i+1}')
        ax1.legend()
        
        # Plot span (generate points in span)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.grid(True)
        ax2.set_title("Span of Vectors")
        
        # Generate points in span
        t = np.linspace(-1, 1, 20)
        for i, v in enumerate(vectors):
            for ti in t:
                point = ti * v
                ax2.plot(point[0], point[1], 'o', color=colors[i], alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_3d_subspace(self, vectors, title):
        """Visualize 3D subspace."""
        fig = plt.figure(figsize=(12, 5))
        
        # Original vectors
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_zlim(-2, 2)
        ax1.set_title("Original Vectors")
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(vectors)))
        for i, v in enumerate(vectors):
            ax1.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i], label=f'v{i+1}')
        ax1.legend()
        
        # Span
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_zlim(-2, 2)
        ax2.set_title("Span of Vectors")
        
        # Generate points in span
        t = np.linspace(-1, 1, 10)
        for i, v in enumerate(vectors):
            for ti in t:
                point = ti * v
                ax2.scatter(point[0], point[1], point[2], color=colors[i], alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demonstrate_vector_space_concepts():
    """
    Demonstrate fundamental vector space concepts.
    """
    print("=" * 60)
    print("VECTOR SPACE CONCEPTS")
    print("=" * 60)
    
    vs = VectorSpace()
    
    # Example 1: Basic vector operations
    print("\n1. Basic Vector Operations")
    print("-" * 40)
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    c = 2.5
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Scalar: {c}")
    print(f"Vector addition: {vs.vector_addition(v1, v2)}")
    print(f"Scalar multiplication: {vs.scalar_multiplication(v1, c)}")
    
    # Example 2: Linear combinations
    print("\n\n2. Linear Combinations")
    print("-" * 40)
    
    vectors = [v1, v2]
    coefficients = [2, -1]
    combination = vs.linear_combination(vectors, coefficients)
    print(f"Linear combination {coefficients[0]}*v1 + {coefficients[1]}*v2 = {combination}")
    
    # Example 3: Span computation
    print("\n\n3. Span Computation")
    print("-" * 40)
    
    vectors_span = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    basis, dimension = vs.span(vectors_span)
    print(f"Original vectors: {len(vectors_span)}")
    print(f"Basis vectors: {len(basis)}")
    print(f"Dimension: {dimension}")
    print(f"Basis: {[v.tolist() for v in basis]}")

def demonstrate_subspace_analysis():
    """
    Demonstrate subspace analysis and properties.
    """
    print("\n\n" + "=" * 60)
    print("SUBSPACE ANALYSIS")
    print("=" * 60)
    
    analyzer = SubspaceAnalyzer()
    
    # Example 1: Line through origin in R^2
    print("\n1. Line through origin in R²")
    print("-" * 40)
    
    line_vectors = [np.array([1, 2]), np.array([2, 4]), np.array([0, 0])]
    analyzer.analyze_subspace(line_vectors, "Line through origin")
    
    # Example 2: Plane through origin in R^3
    print("\n\n2. Plane through origin in R³")
    print("-" * 40)
    
    plane_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0])]
    analyzer.analyze_subspace(plane_vectors, "Plane through origin")
    
    # Example 3: Not a subspace
    print("\n\n3. Set that is not a subspace")
    print("-" * 40)
    
    not_subspace = [np.array([1, 1]), np.array([2, 2])]  # Missing zero vector
    analyzer.analyze_subspace(not_subspace, "Set missing zero vector")
    
    # Example 4: Linear independence
    print("\n\n4. Linear Independence Analysis")
    print("-" * 40)
    
    independent_vectors = [np.array([1, 0]), np.array([0, 1])]
    dependent_vectors = [np.array([1, 0]), np.array([2, 0])]
    
    print("Independent vectors:")
    print(f"Linearly independent: {vs.is_linearly_independent(independent_vectors)}")
    
    print("\nDependent vectors:")
    print(f"Linearly independent: {vs.is_linearly_independent(dependent_vectors)}")

def demonstrate_basis_and_dimension():
    """
    Demonstrate basis finding and dimension computation.
    """
    print("\n\n" + "=" * 60)
    print("BASIS AND DIMENSION")
    print("=" * 60)
    
    vs = VectorSpace()
    
    # Example 1: Standard basis for R^3
    print("\n1. Standard basis for R³")
    print("-" * 40)
    
    standard_basis = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]
    
    basis = vs.find_basis(standard_basis)
    dimension = vs.compute_dimension(standard_basis)
    
    print(f"Original vectors: {len(standard_basis)}")
    print(f"Basis vectors: {len(basis)}")
    print(f"Dimension: {dimension}")
    print(f"Linearly independent: {vs.is_linearly_independent(standard_basis)}")
    
    # Example 2: Redundant vectors
    print("\n\n2. Redundant vectors")
    print("-" * 40)
    
    redundant_vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),  # Redundant
        np.array([2, 2])   # Redundant
    ]
    
    basis = vs.find_basis(redundant_vectors)
    dimension = vs.compute_dimension(redundant_vectors)
    
    print(f"Original vectors: {len(redundant_vectors)}")
    print(f"Basis vectors: {len(basis)}")
    print(f"Dimension: {dimension}")
    print(f"Linearly independent: {vs.is_linearly_independent(redundant_vectors)}")
    
    # Example 3: Function space (polynomials)
    print("\n\n3. Function space (polynomials)")
    print("-" * 40)
    
    # Represent polynomials as coefficient vectors
    p1 = np.array([1, 0, 0])  # 1
    p2 = np.array([0, 1, 0])  # x
    p3 = np.array([0, 0, 1])  # x²
    
    polynomial_basis = [p1, p2, p3]
    basis = vs.find_basis(polynomial_basis)
    dimension = vs.compute_dimension(polynomial_basis)
    
    print(f"Polynomial basis vectors: {len(polynomial_basis)}")
    print(f"Dimension: {dimension}")
    print(f"Linearly independent: {vs.is_linearly_independent(polynomial_basis)}")

def demonstrate_null_and_column_spaces():
    """
    Demonstrate null space and column space analysis.
    """
    print("\n\n" + "=" * 60)
    print("NULL SPACE AND COLUMN SPACE")
    print("=" * 60)
    
    vs = VectorSpace()
    
    # Example matrix
    A = np.array([
        [1, 2, 3],
        [0, 1, 2],
        [1, 3, 5]
    ])
    
    print(f"Matrix A:\n{A}")
    
    # Null space
    null_basis = vs.null_space(A)
    print(f"\nNull space basis:\n{null_basis}")
    print(f"Null space dimension: {null_basis.shape[1] if null_basis.size > 0 else 0}")
    
    # Column space
    col_basis = vs.column_space(A)
    print(f"\nColumn space basis:\n{col_basis}")
    print(f"Column space dimension: {col_basis.shape[1] if col_basis.size > 0 else 0}")
    
    # Rank-nullity theorem
    theorem = vs.rank_nullity_theorem(A)
    print(f"\nRank-nullity theorem verification:")
    print(f"Rank: {theorem['rank']}")
    print(f"Nullity: {theorem['nullity']}")
    print(f"Number of columns: {theorem['n_columns']}")
    print(f"Theorem holds: {theorem['theorem_holds']}")
    print(f"Rank + Nullity = {theorem['rank'] + theorem['nullity']} = {theorem['n_columns']}")
    
    # Verify null space property
    if null_basis.size > 0:
        print(f"\nVerifying null space property (A * null_basis = 0):")
        result = A @ null_basis
        print(f"Result:\n{result}")
        print(f"All entries close to zero: {np.allclose(result, 0)}")

def demonstrate_ml_applications():
    """
    Demonstrate machine learning applications of vector spaces.
    """
    print("\n\n" + "=" * 60)
    print("MACHINE LEARNING APPLICATIONS")
    print("=" * 60)
    
    vs = VectorSpace()
    
    # Example 1: Feature space analysis
    print("\n1. Feature Space Analysis")
    print("-" * 40)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]  # Feature 1 correlated with feature 0
    X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 2]  # Feature 2 correlated with feature 0
    
    print(f"Data shape: {X.shape}")
    print(f"Feature correlation matrix:\n{np.corrcoef(X.T)}")
    
    # Analyze feature space
    feature_basis, feature_dimension = vs.span(X.T)
    print(f"\nFeature space dimension: {feature_dimension}")
    print(f"Effective number of independent features: {feature_dimension}")
    print(f"Redundant features: {n_features - feature_dimension}")
    
    # Example 2: PCA as change of basis
    print("\n\n2. PCA as Change of Basis")
    print("-" * 40)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original feature space dimension: {X_scaled.shape[1]}")
    print(f"PCA components: {pca.n_components_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Analyze PCA basis
    pca_basis = pca.components_.T
    pca_basis_rank = vs.compute_dimension(pca_basis)
    print(f"PCA basis dimension: {pca_basis_rank}")
    print(f"PCA basis is orthonormal: {np.allclose(pca_basis.T @ pca_basis, np.eye(pca_basis.shape[1]))}")
    
    # Example 3: Kernel methods
    print("\n\n3. Kernel Methods")
    print("-" * 40)
    
    # Polynomial kernel
    def polynomial_kernel(X, degree=2, c=1):
        return (X @ X.T + c) ** degree
    
    # RBF kernel
    def rbf_kernel(X, gamma=1.0):
        from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf
        return sklearn_rbf(X, gamma=gamma)
    
    # Compute kernel matrices
    K_poly = polynomial_kernel(X_scaled[:10, :])  # Use subset for demonstration
    K_rbf = rbf_kernel(X_scaled[:10, :])
    
    print(f"Polynomial kernel matrix shape: {K_poly.shape}")
    print(f"RBF kernel matrix shape: {K_rbf.shape}")
    
    # Analyze kernel feature space
    poly_eigenvals, poly_eigenvecs = np.linalg.eigh(K_poly)
    rbf_eigenvals, rbf_eigenvecs = np.linalg.eigh(K_rbf)
    
    print(f"Polynomial kernel effective dimension: {np.sum(poly_eigenvals > 1e-10)}")
    print(f"RBF kernel effective dimension: {np.sum(rbf_eigenvals > 1e-10)}")
    
    # Example 4: Neural network weight spaces
    print("\n\n4. Neural Network Weight Spaces")
    print("-" * 40)
    
    # Simulate neural network weights
    input_dim = 10
    hidden_dim = 5
    output_dim = 2
    
    # Weight matrices
    W1 = np.random.randn(input_dim, hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim)
    
    print(f"Weight matrix shapes:")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")
    
    # Analyze weight spaces
    W1_rank = vs.compute_dimension(W1.T)
    W2_rank = vs.compute_dimension(W2.T)
    
    print(f"W1 column space dimension: {W1_rank}")
    print(f"W2 column space dimension: {W2_rank}")
    
    # Null spaces
    W1_null = vs.null_space(W1)
    W2_null = vs.null_space(W2)
    
    print(f"W1 null space dimension: {W1_null.shape[1] if W1_null.size > 0 else 0}")
    print(f"W2 null space dimension: {W2_null.shape[1] if W2_null.size > 0 else 0}")

def create_exercises():
    """
    Create exercises for vector spaces and subspaces.
    """
    print("\n\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    
    vs = VectorSpace()
    analyzer = SubspaceAnalyzer()
    
    # Exercise 1: Subspace verification
    print("Exercise 1: Subspace Verification")
    print("-" * 50)
    
    # Test set W = {(x, y, z) : x + y + z = 0}
    # This is a plane through the origin
    W_vectors = [
        np.array([1, -1, 0]),
        np.array([1, 0, -1]),
        np.array([0, 0, 0])
    ]
    
    result = analyzer.analyze_subspace(W_vectors, "Plane x + y + z = 0")
    print(f"Forms a subspace: {result['is_subspace']}")
    print(f"Dimension: {result['dimension']}")
    
    # Exercise 2: Linear independence
    print("\n\nExercise 2: Linear Independence")
    print("-" * 50)
    
    test_sets = [
        [np.array([1, 2]), np.array([3, 4])],
        [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0])],
        [np.array([1, 0]), np.array([2, 0])]  # Dependent
    ]
    
    for i, vectors in enumerate(test_sets):
        is_independent = vs.is_linearly_independent(vectors)
        dimension = vs.compute_dimension(vectors)
        print(f"Set {i+1}: Independent = {is_independent}, Dimension = {dimension}")
    
    # Exercise 3: Matrix spaces
    print("\n\nExercise 3: Matrix Spaces")
    print("-" * 50)
    
    # Symmetric 2x2 matrices
    symmetric_matrices = [
        np.array([1, 0, 0, 0]).reshape(2, 2),  # [[1, 0], [0, 0]]
        np.array([0, 1, 1, 0]).reshape(2, 2),  # [[0, 1], [1, 0]]
        np.array([0, 0, 0, 1]).reshape(2, 2)   # [[0, 0], [0, 1]]
    ]
    
    # Flatten matrices for analysis
    flattened_matrices = [m.flatten() for m in symmetric_matrices]
    
    basis = vs.find_basis(flattened_matrices)
    dimension = vs.compute_dimension(flattened_matrices)
    
    print(f"Symmetric 2x2 matrices:")
    print(f"Basis vectors: {len(basis)}")
    print(f"Dimension: {dimension}")
    
    # Exercise 4: Null space and column space
    print("\n\nExercise 4: Null Space and Column Space")
    print("-" * 50)
    
    A = np.array([
        [1, 2, 3],
        [0, 1, 2],
        [1, 3, 5]
    ])
    
    null_basis = vs.null_space(A)
    col_basis = vs.column_space(A)
    theorem = vs.rank_nullity_theorem(A)
    
    print(f"Matrix A:\n{A}")
    print(f"Null space dimension: {null_basis.shape[1] if null_basis.size > 0 else 0}")
    print(f"Column space dimension: {col_basis.shape[1] if col_basis.size > 0 else 0}")
    print(f"Rank: {theorem['rank']}")
    print(f"Nullity: {theorem['nullity']}")
    print(f"Rank + Nullity = {theorem['rank'] + theorem['nullity']} = {theorem['n_columns']}")
    
    # Exercise 5: Function spaces
    print("\n\nExercise 5: Function Spaces")
    print("-" * 50)
    
    # Represent functions as vectors of coefficients
    # 1, sin(x), cos(x) as coefficient vectors for different powers
    # This is a simplified representation
    function_vectors = [
        np.array([1, 0, 0]),  # 1
        np.array([0, 1, 0]),  # sin(x) (simplified)
        np.array([0, 0, 1])   # cos(x) (simplified)
    ]
    
    is_independent = vs.is_linearly_independent(function_vectors)
    dimension = vs.compute_dimension(function_vectors)
    
    print(f"Function space {1, 'sin(x)', 'cos(x)'}:")
    print(f"Linearly independent: {is_independent}")
    print(f"Dimension: {dimension}")

def main():
    """
    Main function to run all demonstrations and exercises.
    """
    print("VECTOR SPACES AND SUBSPACES IMPLEMENTATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_vector_space_concepts()
    demonstrate_subspace_analysis()
    demonstrate_basis_and_dimension()
    demonstrate_null_and_column_spaces()
    demonstrate_ml_applications()
    
    # Run exercises
    create_exercises()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key Concepts Covered:
    1. Vector space axioms and properties
    2. Subspaces and subspace criteria
    3. Span and linear combinations
    4. Linear independence and dependence
    5. Basis and dimension computation
    6. Null space and column space analysis
    7. Rank-nullity theorem
    8. Machine learning applications (feature spaces, PCA, kernel methods)
    
    Key Takeaways:
    - Vector spaces provide the mathematical foundation for linear algebra
    - Subspaces represent solution sets and important geometric objects
    - Basis and dimension are fundamental for understanding coordinate systems
    - Null space and column space reveal matrix structure
    - Vector space concepts are essential for ML algorithms and data analysis
    """)

if __name__ == "__main__":
    main() 