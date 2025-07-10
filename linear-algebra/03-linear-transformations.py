"""
Linear Transformations
=====================

This module covers fundamental linear transformation concepts in linear algebra with
applications to AI/ML and data science. Linear transformations are functions that
preserve vector addition and scalar multiplication, forming the mathematical foundation
for neural networks, dimensionality reduction, and many other AI/ML techniques.

Key Topics:
- Linear transformation properties and matrix representation
- Common transformations (scaling, rotation, reflection, shear)
- Transformation composition and order effects
- Geometric properties (determinant, eigenvalues, eigenvectors)
- AI/ML applications (neural networks, PCA)
- Visualization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. LINEAR TRANSFORMATION FUNDAMENTALS
# =============================================================================

def test_linearity_properties(T_matrix, u, v, c):
    """
    Test if a matrix represents a linear transformation.
    
    A function T: ℝⁿ → ℝᵐ is linear if it satisfies:
    1. Additivity: T(u + v) = T(u) + T(v)
    2. Homogeneity: T(cu) = cT(u)
    
    These properties can be combined into:
    T(au + bv) = aT(u) + bT(v)
    """
    # Test additivity: T(u + v) = T(u) + T(v)
    left_side = T_matrix @ (u + v)
    right_side = (T_matrix @ u) + (T_matrix @ v)
    additivity_holds = np.allclose(left_side, right_side)
    
    # Test homogeneity: T(cu) = cT(u)
    left_side = T_matrix @ (c * u)
    right_side = c * (T_matrix @ u)
    homogeneity_holds = np.allclose(left_side, right_side)
    
    # Test zero preservation: T(0) = 0
    zero_vector = np.zeros_like(u)
    zero_preserved = np.allclose(T_matrix @ zero_vector, zero_vector)
    
    return additivity_holds, homogeneity_holds, zero_preserved

def demonstrate_linearity_properties():
    """
    Demonstrate linear transformation properties with examples.
    """
    print("=== Linear Transformation Properties ===")
    
    # Example test
    test_matrix = np.array([[2, 1], [1, 3]])
    u = np.array([1, 2])
    v = np.array([3, 4])
    c = 2.5

    additivity, homogeneity, zero_preserved = test_linearity_properties(test_matrix, u, v, c)
    print("Linear transformation properties:")
    print(f"Additivity: {additivity}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Zero preservation: {zero_preserved}")
    
    # Geometric interpretation
    print("\nGeometric properties of linear transformations:")
    print("- Preserve lines: Lines remain lines (though they may be rotated, scaled, or sheared)")
    print("- Preserve origin: T(0) = 0")
    print("- Preserve parallelism: Parallel lines remain parallel")
    print("- Preserve linear combinations: The image of a linear combination is the linear combination of the images")

# =============================================================================
# 2. COMMON LINEAR TRANSFORMATIONS
# =============================================================================

def create_scaling_matrix(scale_x, scale_y=None):
    """
    Create scaling matrix with different x and y scaling.
    
    Mathematical Definition:
    Scaling transforms vectors by multiplying each component by a scalar factor.
    
    Matrix Form: For uniform scaling by factor k:
    S = [k  0]
        [0  k]
    
    Properties:
    - Preserves angles between vectors
    - Changes lengths by factor |k|
    - If |k| > 1: dilation (expansion)
    - If |k| < 1: contraction (compression)
    - If k < 0: reflection through origin
    """
    if scale_y is None:
        scale_y = scale_x  # Uniform scaling
    return np.array([[scale_x, 0], [0, scale_y]])

def demonstrate_scaling_transformations():
    """
    Demonstrate different types of scaling transformations.
    """
    print("\n=== Scaling Transformations ===")
    
    # Different types of scaling
    uniform_scale = create_scaling_matrix(2)  # Uniform scaling by 2
    nonuniform_scale = create_scaling_matrix(2, 0.5)  # Scale x by 2, y by 0.5
    reflection = create_scaling_matrix(-1)  # Reflection through origin

    print("Uniform scaling matrix (scale by 2):")
    print(uniform_scale)
    print("\nNon-uniform scaling matrix (x×2, y×0.5):")
    print(nonuniform_scale)
    print("\nReflection matrix (scale by -1):")
    print(reflection)

    # Apply transformations
    v = np.array([1, 1])
    print(f"\nOriginal vector: {v}")
    print(f"Uniformly scaled: {uniform_scale @ v}")
    print(f"Non-uniformly scaled: {nonuniform_scale @ v}")
    print(f"Reflected: {reflection @ v}")

    # Verify properties
    print(f"\nDeterminant of uniform scaling: {np.linalg.det(uniform_scale)}")
    print(f"Area scaling factor: {np.linalg.det(uniform_scale)}")

def create_rotation_matrix(angle_rad):
    """
    Create 2D rotation matrix.
    
    Mathematical Definition:
    Rotation transforms vectors by rotating them around the origin by a specified angle.
    
    Matrix Form: For rotation by angle θ (counterclockwise):
    R(θ) = [cos(θ)  -sin(θ)]
           [sin(θ)   cos(θ)]
    
    Properties:
    - Preserves lengths: |R(θ)v| = |v|
    - Preserves angles between vectors
    - Determinant is 1: det(R(θ)) = 1
    - Inverse is transpose: R(θ)⁻¹ = R(θ)ᵀ = R(-θ)
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

def demonstrate_rotation_transformations():
    """
    Demonstrate rotation transformations and their properties.
    """
    print("\n=== Rotation Transformations ===")
    
    # Different rotation angles
    angles_deg = [0, 45, 90, 180, 270]
    angles_rad = [np.radians(angle) for angle in angles_deg]

    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        R = create_rotation_matrix(angle_rad)
        print(f"\nRotation matrix ({angle_deg}°):")
        print(R)
        print(f"Determinant: {np.linalg.det(R):.6f}")
        print(f"Rᵀ × R = I: {np.allclose(R.T @ R, np.eye(2))}")

    # Test rotation properties
    v = np.array([1, 0])
    original_length = np.linalg.norm(v)

    for angle_deg, angle_rad in zip([0, 45, 90], angles_rad[:3]):
        R = create_rotation_matrix(angle_rad)
        v_rotated = R @ v
        rotated_length = np.linalg.norm(v_rotated)
        print(f"\n{angle_deg}° rotation:")
        print(f"Original: {v}, length: {original_length:.4f}")
        print(f"Rotated: {v_rotated}, length: {rotated_length:.4f}")
        print(f"Length preserved: {np.isclose(original_length, rotated_length)}")

    # Composition of rotations
    R1 = create_rotation_matrix(np.pi/4)  # 45°
    R2 = create_rotation_matrix(np.pi/6)  # 30°
    R_combined = R1 @ R2  # Rotate by 30° then by 45°
    R_direct = create_rotation_matrix(np.pi/4 + np.pi/6)  # Direct 75° rotation

    print(f"\nComposition of rotations:")
    print(f"R(45°) × R(30°) = R(75°): {np.allclose(R_combined, R_direct)}")

def create_reflection_matrices():
    """
    Create various reflection matrices.
    
    Mathematical Definition:
    Reflection flips vectors across a specified line or plane.
    
    Matrix Forms:
    - Reflection across x-axis: Rₓ = [1  0]
                                    [0 -1]
    - Reflection across y-axis: Rᵧ = [-1  0]
                                     [0   1]
    - Reflection across line y = x: R_{y=x} = [0  1]
                                              [1  0]
    
    Properties:
    - Determinant is -1: det(R) = -1
    - R² = I (applying reflection twice returns to original)
    - Preserves lengths but changes orientation
    """
    # Reflection across x-axis
    R_x = np.array([[1, 0], [0, -1]])
    
    # Reflection across y-axis
    R_y = np.array([[-1, 0], [0, 1]])
    
    # Reflection across line y = x
    R_y_eq_x = np.array([[0, 1], [1, 0]])
    
    # Reflection across line y = -x
    R_y_eq_neg_x = np.array([[0, -1], [-1, 0]])
    
    return R_x, R_y, R_y_eq_x, R_y_eq_neg_x

def demonstrate_reflection_transformations():
    """
    Demonstrate reflection transformations and their properties.
    """
    print("\n=== Reflection Transformations ===")
    
    R_x, R_y, R_y_eq_x, R_y_eq_neg_x = create_reflection_matrices()

    reflections = {
        "Across x-axis": R_x,
        "Across y-axis": R_y,
        "Across y = x": R_y_eq_x,
        "Across y = -x": R_y_eq_neg_x
    }

    for name, matrix in reflections.items():
        print(f"\n{name}:")
        print(matrix)
        print(f"Determinant: {np.linalg.det(matrix)}")
        print(f"Matrix² = I: {np.allclose(matrix @ matrix, np.eye(2))}")

    # Test reflection properties
    v = np.array([1, 1])
    print(f"\nOriginal vector: {v}")

    for name, matrix in reflections.items():
        v_reflected = matrix @ v
        print(f"{name}: {v_reflected}")
        print(f"Length preserved: {np.isclose(np.linalg.norm(v), np.linalg.norm(v_reflected))}")

def create_shear_matrices(shear_factor):
    """
    Create horizontal and vertical shear matrices.
    
    Mathematical Definition:
    Shear transforms vectors by adding a multiple of one component to another,
    creating a "sliding" effect.
    
    Matrix Forms:
    - Horizontal shear: Hₓ = [1  k]
                            [0  1]
    - Vertical shear: Hᵧ = [1  0]
                           [k  1]
    
    Properties:
    - Determinant is 1: det(H) = 1 (preserves area)
    - Preserves lines parallel to the shear direction
    - Changes angles between vectors
    """
    # Horizontal shear (shear in x-direction)
    H_x = np.array([[1, shear_factor], [0, 1]])
    
    # Vertical shear (shear in y-direction)
    H_y = np.array([[1, 0], [shear_factor, 1]])
    
    return H_x, H_y

def demonstrate_shear_transformations():
    """
    Demonstrate shear transformations and their properties.
    """
    print("\n=== Shear Transformations ===")
    
    shear_factor = 0.5
    H_x, H_y = create_shear_matrices(shear_factor)

    print("Horizontal shear matrix:")
    print(H_x)
    print(f"Determinant: {np.linalg.det(H_x)}")

    print("\nVertical shear matrix:")
    print(H_y)
    print(f"Determinant: {np.linalg.det(H_y)}")

    # Apply shear transformations
    v = np.array([1, 1])
    print(f"\nOriginal vector: {v}")
    print(f"Horizontal shear: {H_x @ v}")
    print(f"Vertical shear: {H_y @ v}")

    # Verify area preservation
    original_area = 1  # Unit square
    sheared_area = np.linalg.det(H_x) * original_area
    print(f"\nOriginal area: {original_area}")
    print(f"Sheared area: {sheared_area}")
    print(f"Area preserved: {np.isclose(original_area, sheared_area)}")

# =============================================================================
# 3. COMPOSITION OF TRANSFORMATIONS
# =============================================================================

def compose_transformations(*matrices):
    """
    Compose multiple transformations by matrix multiplication.
    
    Mathematical Principle:
    Linear transformations can be combined by matrix multiplication.
    The composition of transformations T₁ and T₂ is given by:
    (T₂ ∘ T₁)(x) = T₂(T₁(x)) = A₂(A₁x) = (A₂A₁)x
    
    Important Note:
    Matrix multiplication is not commutative, so the order of transformations matters!
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = matrix @ result
    return result

def demonstrate_transformation_composition():
    """
    Demonstrate composition of transformations and order effects.
    """
    print("\n=== Transformation Composition ===")
    
    # Example: Scale then rotate
    scale_matrix = create_scaling_matrix(2)
    rotation_matrix = create_rotation_matrix(np.pi/4)

    # Scale then rotate
    scale_then_rotate = rotation_matrix @ scale_matrix
    # Rotate then scale
    rotate_then_scale = scale_matrix @ rotation_matrix

    print("Scale then rotate:")
    print(scale_then_rotate)
    print("\nRotate then scale:")
    print(rotate_then_scale)
    print(f"\nOrder matters: {not np.allclose(scale_then_rotate, rotate_then_scale)}")

    # Test on a vector
    v = np.array([1, 0])
    print(f"\nOriginal vector: {v}")
    print(f"Scale then rotate: {scale_then_rotate @ v}")
    print(f"Rotate then scale: {rotate_then_scale @ v}")

    # Multiple transformations
    shear_matrix = create_shear_matrices(0.3)[0]  # Horizontal shear
    combined = compose_transformations(scale_matrix, rotation_matrix, shear_matrix)
    print(f"\nCombined transformation (scale → rotate → shear):")
    print(combined)

# =============================================================================
# 4. PROPERTIES OF LINEAR TRANSFORMATIONS
# =============================================================================

def analyze_transformation_properties(matrix, name):
    """
    Analyze properties of a transformation matrix.
    
    Key Property:
    The determinant of a transformation matrix tells us how it affects
    area (in 2D) or volume (in 3D).
    
    - |det(A)| = 1: Preserves area/volume
    - |det(A)| > 1: Expands area/volume
    - |det(A)| < 1: Contracts area/volume
    - det(A) < 0: Changes orientation (reflection)
    """
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)
    
    print(f"\n{name}:")
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {det:.4f}")
    print(f"Trace: {trace:.4f}")
    
    if abs(det) == 1:
        print("Area/volume preserving")
    elif abs(det) > 1:
        print(f"Expands area/volume by factor {abs(det):.2f}")
    else:
        print(f"Contracts area/volume by factor {abs(det):.2f}")
    
    if det < 0:
        print("Changes orientation (reflection)")
    else:
        print("Preserves orientation")
    
    # Check if orthogonal (rotation/reflection)
    is_orthogonal = np.allclose(matrix.T @ matrix, np.eye(matrix.shape[0]))
    if is_orthogonal:
        print("Orthogonal transformation (rotation or reflection)")
    
    return det, trace

def demonstrate_transformation_properties():
    """
    Demonstrate analysis of different transformation properties.
    """
    print("\n=== Transformation Properties Analysis ===")
    
    # Analyze different transformations
    transformations = {
        "Identity": np.eye(2),
        "Scaling (2x)": create_scaling_matrix(2),
        "Rotation (45°)": create_rotation_matrix(np.pi/4),
        "Reflection (x-axis)": np.array([[1, 0], [0, -1]]),
        "Shear (0.5)": np.array([[1, 0.5], [0, 1]]),
        "Non-uniform scaling": create_scaling_matrix(2, 0.5)
    }

    for name, matrix in transformations.items():
        analyze_transformation_properties(matrix, name)

def analyze_eigenproperties(matrix, name):
    """
    Analyze eigenvalues and eigenvectors of a transformation.
    
    Definition:
    For a linear transformation T represented by matrix A, a non-zero vector v
    is an eigenvector with eigenvalue λ if:
    Av = λv
    
    Geometric Interpretation:
    - Eigenvectors are vectors that don't change direction under the transformation
    - Eigenvalues tell us how much these vectors are scaled
    - The eigenvectors form a basis that diagonalizes the transformation
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    print(f"\n{name} - Eigenanalysis:")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:")
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"  λ{i+1} = {eigenval:.4f}: {eigenvec}")
    
    # Verify eigenvalue equation
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        left_side = matrix @ eigenvec
        right_side = eigenval * eigenvec
        verification = np.allclose(left_side, right_side)
        print(f"  Verification for λ{i+1}: {verification}")
    
    return eigenvalues, eigenvectors

def demonstrate_eigenvalue_analysis():
    """
    Demonstrate eigenvalue analysis of different transformations.
    """
    print("\n=== Eigenvalue Analysis ===")
    
    transformations = {
        "Scaling (2x)": create_scaling_matrix(2),
        "Rotation (45°)": create_rotation_matrix(np.pi/4),
        "Reflection": np.array([[1, 0], [0, -1]]),
        "Shear": np.array([[1, 0.5], [0, 1]])
    }

    for name, matrix in transformations.items():
        analyze_eigenproperties(matrix, name)

# =============================================================================
# 5. AI/ML APPLICATIONS
# =============================================================================

def neural_network_layer_analysis():
    """
    Analyze linear transformations in neural networks.
    """
    print("\n=== Neural Network Layer Analysis ===")
    
    # Simulate a neural network layer
    input_size = 4
    output_size = 3
    batch_size = 5
    
    # Weight matrix (linear transformation)
    W = np.random.randn(input_size, output_size) * 0.1
    b = np.zeros(output_size)  # Bias vector
    
    # Input data
    X = np.random.randn(batch_size, input_size)
    
    # Forward pass (linear transformation + bias)
    Z = X @ W + b
    
    print("Neural Network Layer Analysis:")
    print(f"Input shape: {X.shape}")
    print(f"Weight matrix shape: {W.shape}")
    print(f"Output shape: {Z.shape}")
    
    # Analyze the linear transformation
    print(f"\nWeight matrix properties:")
    print(f"Rank: {np.linalg.matrix_rank(W)}")
    print(f"Condition number: {np.linalg.cond(W):.2f}")
    
    # Singular value decomposition
    U, S, Vt = np.linalg.svd(W)
    print(f"Singular values: {S}")
    print(f"Effective rank: {np.sum(S > 1e-10)}")
    
    return W, X, Z

def pca_analysis():
    """
    Demonstrate PCA as a linear transformation.
    """
    print("\n=== Principal Component Analysis (PCA) ===")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create correlated data
    data = np.random.multivariate_normal([0, 0, 0], 
                                       [[1, 0.8, 0.6], 
                                        [0.8, 1, 0.7], 
                                        [0.6, 0.7, 1]], 
                                       n_samples)
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print("PCA Analysis:")
    print(f"Original data shape: {data.shape}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Explained variance ratio: {eigenvalues / np.sum(eigenvalues)}")
    
    # Project data onto principal components
    data_pca = data_centered @ eigenvectors
    
    print(f"\nTransformed data shape: {data_pca.shape}")
    print(f"Variance in each component: {np.var(data_pca, axis=0)}")
    
    return data, data_pca, eigenvectors, eigenvalues

# =============================================================================
# 6. VISUALIZATION OF TRANSFORMATIONS
# =============================================================================

def visualize_transformations():
    """
    Comprehensive visualization of linear transformations.
    """
    # Create unit square and unit vectors
    unit_square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    unit_vectors = [np.array([1, 0]), np.array([0, 1])]
    
    # Define transformations
    transformations = {
        "Identity": np.eye(2),
        "Scaling (2x)": create_scaling_matrix(2),
        "Rotation (45°)": create_rotation_matrix(np.pi/4),
        "Reflection (x-axis)": np.array([[1, 0], [0, -1]]),
        "Shear (0.5)": np.array([[1, 0.5], [0, 1]]),
        "Non-uniform scaling": create_scaling_matrix(2, 0.5)
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, matrix) in enumerate(transformations.items()):
        ax = axes[i]
        
        # Original unit square
        ax.plot(unit_square[0], unit_square[1], 'b-', linewidth=2, label='Original')
        
        # Transformed unit square
        transformed_square = matrix @ unit_square
        ax.plot(transformed_square[0], transformed_square[1], 'r-', linewidth=2, label='Transformed')
        
        # Unit vectors
        for j, vector in enumerate(unit_vectors):
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1,
                     color='blue', alpha=0.5, width=0.02)
            transformed_vector = matrix @ vector
            ax.quiver(0, 0, transformed_vector[0], transformed_vector[1], angles='xy', scale_units='xy', scale=1,
                     color='red', alpha=0.7, width=0.02)
        
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nDet: {np.linalg.det(matrix):.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. EXERCISES
# =============================================================================

def verify_linearity():
    """
    Verify linear transformation properties.
    """
    print("\n=== Exercise 1: Linear Transformation Properties ===")
    
    A = np.array([[2, 1], [1, 3]])
    u = np.array([1, 2])
    v = np.array([3, 4])
    c = 2.5
    
    # Test additivity
    additivity = np.allclose(A @ (u + v), A @ u + A @ v)
    print(f"Additivity: {additivity}")
    
    # Test homogeneity
    homogeneity = np.allclose(A @ (c * u), c * (A @ u))
    print(f"Homogeneity: {homogeneity}")
    
    # Test zero preservation
    zero_preservation = np.allclose(A @ np.zeros(2), np.zeros(2))
    print(f"Zero preservation: {zero_preservation}")
    
    return additivity and homogeneity and zero_preservation

def composition_order_effects():
    """
    Study the effect of transformation order.
    """
    print("\n=== Exercise 2: Transformation Composition ===")
    
    # Define transformations
    scale = create_scaling_matrix(2)
    rotate = create_rotation_matrix(np.pi/6)  # 30 degrees
    shear = np.array([[1, 0.5], [0, 1]])
    
    # Test vector
    v = np.array([1, 1])
    
    # Different compositions
    T1 = scale @ rotate @ shear  # Scale → Rotate → Shear
    T2 = rotate @ scale @ shear  # Rotate → Scale → Shear
    T3 = shear @ rotate @ scale  # Shear → Rotate → Scale
    
    print("Transformation compositions:")
    print(f"Scale → Rotate → Shear: {T1 @ v}")
    print(f"Rotate → Scale → Shear: {T2 @ v}")
    print(f"Shear → Rotate → Scale: {T3 @ v}")
    
    # Check if results are different
    results_different = not (np.allclose(T1 @ v, T2 @ v) and np.allclose(T2 @ v, T3 @ v))
    print(f"Order matters: {results_different}")

def eigenvalue_analysis():
    """
    Analyze eigenvalues of different transformations.
    """
    print("\n=== Exercise 3: Eigenvalue Analysis ===")
    
    transformations = {
        "Scaling (2x)": create_scaling_matrix(2),
        "Rotation (45°)": create_rotation_matrix(np.pi/4),
        "Reflection": np.array([[1, 0], [0, -1]]),
        "Shear": np.array([[1, 0.5], [0, 1]])
    }
    
    for name, matrix in transformations.items():
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        print(f"\n{name}:")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Product of eigenvalues: {np.prod(eigenvalues):.4f}")
        print(f"Determinant: {np.linalg.det(matrix):.4f}")
        print(f"Sum of eigenvalues: {np.sum(eigenvalues):.4f}")
        print(f"Trace: {np.trace(matrix):.4f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to demonstrate all linear transformation concepts and operations.
    """
    print("LINEAR TRANSFORMATIONS")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Linear Transformation Fundamentals
    demonstrate_linearity_properties()
    
    # 2. Common Linear Transformations
    demonstrate_scaling_transformations()
    demonstrate_rotation_transformations()
    demonstrate_reflection_transformations()
    demonstrate_shear_transformations()
    
    # 3. Transformation Composition
    demonstrate_transformation_composition()
    
    # 4. Transformation Properties
    demonstrate_transformation_properties()
    demonstrate_eigenvalue_analysis()
    
    # 5. AI/ML Applications
    neural_network_layer_analysis()
    pca_analysis()
    
    # 6. Exercises
    verify_linearity()
    composition_order_effects()
    eigenvalue_analysis()
    
    # 7. Visualizations (uncomment to show plots)
    print("\n=== Visualizations ===")
    print("Uncomment the following line to show transformation visualizations:")
    print("# visualize_transformations()")
    
    print("\n" + "=" * 50)
    print("LINEAR TRANSFORMATIONS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 