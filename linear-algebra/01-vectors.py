"""
Vectors and Vector Operations
============================

This module covers fundamental vector concepts in linear algebra with applications
to AI/ML and data science. Vectors are essential for representing data points,
features, and parameters in mathematical models.

Key Topics:
- Vector representation and operations
- Dot product and cross product
- Vector properties (magnitude, unit vectors)
- Geometric interpretations
- AI/ML applications
- Visualization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. VECTOR CREATION AND REPRESENTATION
# =============================================================================

def demonstrate_vector_creation():
    """
    Demonstrate different ways to create vectors in NumPy.
    
    Vectors are ordered lists of numbers that can represent:
    - Points in space (coordinates)
    - Directions and magnitudes
    - Features of data points
    - Parameters in machine learning models
    """
    print("=== Vector Creation Examples ===")
    
    # Creating vectors with different methods
    v1 = np.array([1, 2, 3])  # 3D vector
    v2 = np.array([4, 5, 6])  # 3D vector
    v3 = np.array([1, 2])     # 2D vector
    
    print("Vector v1:", v1)
    print("Vector v2:", v2)
    print("Vector v3:", v3)
    print("Shape of v1:", v1.shape)
    print("Dimension of v1:", v1.ndim)
    
    # Creating special vectors
    zeros_vector = np.zeros(5)  # Zero vector
    ones_vector = np.ones(4)    # Vector of ones
    random_vector = np.random.randn(3)  # Random vector
    linspace_vector = np.linspace(0, 10, 5)  # Equally spaced values
    
    print("\nSpecial vectors:")
    print("Zero vector:", zeros_vector)
    print("Ones vector:", ones_vector)
    print("Random vector:", random_vector)
    print("Linspace vector:", linspace_vector)
    
    # Vector properties
    print("\nVector properties:")
    print("Data type of v1:", v1.dtype)
    print("Is v1 a 1D array?", v1.ndim == 1)
    print("Length of v1:", len(v1))
    
    return v1, v2, v3

# =============================================================================
# 2. VECTOR OPERATIONS
# =============================================================================

def demonstrate_vector_addition(v1, v2, v3):
    """
    Demonstrate vector addition and its properties.
    
    Mathematical Definition:
    Vector addition is performed component-wise:
    a + b = [a₁ + b₁, a₂ + b₂, ..., aₙ + bₙ]
    
    Properties:
    - Commutative: a + b = b + a
    - Associative: (a + b) + c = a + (b + c)
    - Identity: a + 0 = a
    - Inverse: a + (-a) = 0
    """
    print("\n=== Vector Addition ===")
    
    # Vector addition
    result = v1 + v2
    print("v1 + v2 =", result)
    
    # Element-wise addition (manual calculation)
    result_manual = np.array([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
    print("Manual addition:", result_manual)
    
    # Broadcasting with scalars
    result_broadcast = v1 + 5  # Adds 5 to each component
    print("v1 + 5 =", result_broadcast)
    
    # Adding multiple vectors
    v4 = np.array([7, 8, 9])
    sum_all = v1 + v2 + v4
    print("v1 + v2 + v4 =", sum_all)
    
    # Verification of properties
    print("\nVerification of properties:")
    print("Commutative property (v1 + v2 == v2 + v1):", 
          np.array_equal(v1 + v2, v2 + v1))
    print("Associative property check:", 
          np.array_equal((v1 + v2) + v4, v1 + (v2 + v4)))
    
    return v4

def demonstrate_scalar_multiplication(v1, v2):
    """
    Demonstrate scalar multiplication and its properties.
    
    Mathematical Definition:
    Scalar multiplication multiplies each component by a scalar:
    c·v = [cv₁, cv₂, ..., cvₙ]
    
    Geometric Interpretation:
    - If c > 0: Vector is scaled by factor c in same direction
    - If c < 0: Vector is scaled by factor |c| in opposite direction
    - If c = 0: Result is the zero vector
    """
    print("\n=== Scalar Multiplication ===")
    
    # Scalar multiplication
    scalar = 2.5
    result = scalar * v1
    print(f"{scalar} * v1 =", result)
    
    # Negative vector
    negative_v1 = -v1
    print("-v1 =", negative_v1)
    
    # Multiple scalar operations
    result_chain = 2 * 3 * v1
    print("2 * 3 * v1 =", result_chain)
    
    # Verification of properties
    print("\nVerification of properties:")
    print("Distributive over vector addition:", 
          np.array_equal(2 * (v1 + v2), 2 * v1 + 2 * v2))
    print("Associative property:", 
          np.array_equal((2 * 3) * v1, 2 * (3 * v1)))
    
    # Scaling effects demonstration
    print("\nScaling effects:")
    scales = [0.5, 1, 2, -1]
    for scale in scales:
        scaled = scale * v1
        print(f"{scale} * v1 = {scaled}, magnitude = {np.linalg.norm(scaled):.2f}")

def demonstrate_vector_subtraction(v1, v2):
    """
    Demonstrate vector subtraction.
    
    Mathematical Definition:
    Vector subtraction is defined as addition with the negative:
    a - b = a + (-b) = [a₁ - b₁, a₂ - b₂, ..., aₙ - bₙ]
    
    Geometric Interpretation:
    a - b represents the vector from the tip of b to the tip of a
    """
    print("\n=== Vector Subtraction ===")
    
    # Vector subtraction
    result = v1 - v2
    print("v1 - v2 =", result)
    
    # Equivalent to addition with negative
    result_equivalent = v1 + (-v2)
    print("v1 + (-v2) =", result_equivalent)
    
    # Subtraction properties
    print("v1 - v1 =", v1 - v1)  # Should be zero vector
    print("v1 - v2 = -(v2 - v1):", np.array_equal(v1 - v2, -(v2 - v1)))
    
    # Distance between points (represented as vectors)
    distance = np.linalg.norm(v1 - v2)
    print("Distance between v1 and v2:", distance)

# =============================================================================
# 3. VECTOR PROPERTIES
# =============================================================================

def demonstrate_vector_magnitude(v1, v2):
    """
    Demonstrate vector magnitude (length) and different norm types.
    
    Mathematical Definition:
    The magnitude (or length) of a vector is given by the Euclidean norm:
    |v| = √(v₁² + v₂² + ... + vₙ²)
    
    Properties:
    - |v| ≥ 0 (non-negative)
    - |v| = 0 if and only if v = 0
    - |c·v| = |c|·|v| (scalar multiplication)
    - Triangle inequality: |a + b| ≤ |a| + |b|
    """
    print("\n=== Vector Magnitude ===")
    
    # Vector magnitude
    magnitude_v1 = np.linalg.norm(v1)
    print("Magnitude of v1:", magnitude_v1)
    
    # Manual calculation
    magnitude_manual = np.sqrt(np.sum(v1**2))
    print("Manual magnitude calculation:", magnitude_manual)
    
    # Different norm types
    l1_norm = np.linalg.norm(v1, ord=1)  # Manhattan norm
    l2_norm = np.linalg.norm(v1, ord=2)  # Euclidean norm (default)
    l_inf_norm = np.linalg.norm(v1, ord=np.inf)  # Maximum norm
    
    print(f"\nDifferent norm types:")
    print(f"L1 norm (Manhattan): {l1_norm:.2f}")
    print(f"L2 norm (Euclidean): {l2_norm:.2f}")
    print(f"L∞ norm (Maximum): {l_inf_norm:.2f}")
    
    # Verification of properties
    print("\nVerification of properties:")
    print("Non-negative:", magnitude_v1 >= 0)
    print("Scalar multiplication:", np.linalg.norm(2 * v1) == 2 * magnitude_v1)
    print("Triangle inequality:", np.linalg.norm(v1 + v2) <= magnitude_v1 + np.linalg.norm(v2))

def demonstrate_unit_vectors(v1):
    """
    Demonstrate unit vectors and their properties.
    
    Mathematical Definition:
    A unit vector has magnitude 1 and points in the same direction:
    v̂ = v / |v|
    
    Properties:
    - |v̂| = 1
    - v̂ points in the same direction as v
    - Any vector can be written as v = |v| · v̂
    """
    print("\n=== Unit Vectors ===")
    
    # Unit vector
    unit_v1 = v1 / np.linalg.norm(v1)
    print("Unit vector of v1:", unit_v1)
    print("Magnitude of unit vector:", np.linalg.norm(unit_v1))
    
    # Verification
    print("Is unit vector magnitude 1?", np.isclose(np.linalg.norm(unit_v1), 1))
    
    # Reconstructing original vector
    reconstructed = np.linalg.norm(v1) * unit_v1
    print("Reconstructed v1:", reconstructed)
    print("Reconstruction successful?", np.array_equal(v1, reconstructed))
    
    # Standard unit vectors (basis vectors)
    e1 = np.array([1, 0, 0])  # i-hat
    e2 = np.array([0, 1, 0])  # j-hat
    e3 = np.array([0, 0, 1])  # k-hat
    
    print("\nStandard unit vectors:")
    print("e1 (i-hat):", e1)
    print("e2 (j-hat):", e2)
    print("e3 (k-hat):", e3)
    print("Magnitudes:", [np.linalg.norm(e) for e in [e1, e2, e3]])
    
    return e1, e2, e3

# =============================================================================
# 4. DOT PRODUCT (INNER PRODUCT)
# =============================================================================

def demonstrate_dot_product(v1, v2, e1, e2, e3):
    """
    Demonstrate dot product and its applications.
    
    Mathematical Definition:
    The dot product of two vectors is a scalar defined as:
    a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = |a||b|cos(θ)
    
    Where θ is the angle between the vectors.
    
    Geometric Interpretation:
    - Measures the projection of one vector onto another
    - If vectors are perpendicular: a · b = 0
    - If vectors are parallel: a · b = ±|a||b|
    """
    print("\n=== Dot Product ===")
    
    # Dot product
    dot_product = np.dot(v1, v2)
    print("Dot product v1 · v2 =", dot_product)
    
    # Alternative notation
    dot_product_alt = v1 @ v2
    print("Dot product using @ operator:", dot_product_alt)
    
    # Manual calculation
    dot_manual = sum(v1[i] * v2[i] for i in range(len(v1)))
    print("Manual dot product:", dot_manual)
    
    # Angle between vectors
    cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)
    print(f"Angle between v1 and v2: {angle_deg:.2f} degrees")
    
    # Verification of properties
    print("\nVerification of properties:")
    print("Commutative property:", np.dot(v1, v2) == np.dot(v2, v1))
    print("Self dot product equals magnitude squared:", 
          np.isclose(np.dot(v1, v1), np.linalg.norm(v1)**2))
    
    # Projection calculation
    projection_length = dot_product / np.linalg.norm(v2)
    projection_vector = projection_length * (v2 / np.linalg.norm(v2))
    print("Projection of v1 onto v2:", projection_vector)
    
    # Orthogonality test
    orthogonal_threshold = 1e-10
    is_orthogonal = abs(np.dot(v1, v2)) < orthogonal_threshold
    print("Are v1 and v2 orthogonal?", is_orthogonal)
    
    # Dot product with unit vectors
    print("\nDot product with unit vectors:")
    print("v1 · e1 =", np.dot(v1, e1))  # x-component
    print("v1 · e2 =", np.dot(v1, e2))  # y-component
    print("v1 · e3 =", np.dot(v1, e3))  # z-component

# =============================================================================
# 5. CROSS PRODUCT (3D VECTORS ONLY)
# =============================================================================

def demonstrate_cross_product(v1, v2, v4):
    """
    Demonstrate cross product and its applications.
    
    Mathematical Definition:
    The cross product of two 3D vectors is a vector perpendicular to both:
    a × b = [a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁]
    
    Geometric Interpretation:
    - Produces a vector perpendicular to both input vectors
    - Direction follows the right-hand rule
    - Magnitude is |a × b| = |a||b|sin(θ)
    - Used to find normal vectors, areas, and volumes
    """
    print("\n=== Cross Product ===")
    
    # Cross product (only for 3D vectors)
    cross_product = np.cross(v1, v2)
    print("Cross product v1 × v2 =", cross_product)
    
    # Verify perpendicularity
    perpendicular_to_v1 = np.dot(cross_product, v1)
    perpendicular_to_v2 = np.dot(cross_product, v2)
    print("Cross product · v1 =", perpendicular_to_v1)
    print("Cross product · v2 =", perpendicular_to_v2)
    
    # Magnitude of cross product
    cross_magnitude = np.linalg.norm(cross_product)
    angle_rad = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
    expected_magnitude = np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(angle_rad)
    print("Cross product magnitude:", cross_magnitude)
    print("Expected magnitude:", expected_magnitude)
    
    # Verification of properties
    print("\nVerification of properties:")
    print("Anti-commutative property:", np.array_equal(cross_product, -np.cross(v2, v1)))
    
    # Area of parallelogram
    area = np.linalg.norm(cross_product)
    print("Area of parallelogram formed by v1 and v2:", area)
    
    # Unit normal vector
    normal_unit = cross_product / np.linalg.norm(cross_product)
    print("Unit normal vector:", normal_unit)
    
    # Triple scalar product (volume of parallelepiped)
    triple_scalar = np.dot(v1, np.cross(v2, v4))
    print("Triple scalar product (volume):", triple_scalar)

# =============================================================================
# 6. VECTOR VISUALIZATION
# =============================================================================

def plot_2d_vectors(v3):
    """
    Create 2D vector visualization showing vector addition and parallelogram law.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Origin
    origin = np.array([0, 0])
    
    # Plot vectors
    ax.quiver(origin[0], origin[1], v3[0], v3[1], 
              angles='xy', scale_units='xy', scale=1, color='blue', label='v3')
    
    # Add another vector
    v4 = np.array([3, 1])
    ax.quiver(origin[0], origin[1], v4[0], v4[1], 
              angles='xy', scale_units='xy', scale=1, color='red', label='v4')
    
    # Vector addition
    v_sum = v3 + v4
    ax.quiver(origin[0], origin[1], v_sum[0], v_sum[1], 
              angles='xy', scale_units='xy', scale=1, color='green', label='v3 + v4')
    
    # Parallelogram construction
    ax.quiver(v3[0], v3[1], v4[0], v4[1], 
              angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5)
    ax.quiver(v4[0], v4[1], v3[0], v3[1], 
              angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)
    
    # Set limits and labels
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Vector Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.show()

def plot_3d_vectors(v1, v2, cross_product):
    """
    Create 3D vector visualization showing vectors and their cross product.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin
    origin = np.array([0, 0, 0])
    
    # Plot vectors
    ax.quiver(origin[0], origin[1], origin[2], 
              v1[0], v1[1], v1[2], color='blue', label='v1', arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 
              v2[0], v2[1], v2[2], color='red', label='v2', arrow_length_ratio=0.1)
    
    # Cross product
    ax.quiver(origin[0], origin[1], origin[2], 
              cross_product[0], cross_product[1], cross_product[2], 
              color='green', label='v1 × v2', arrow_length_ratio=0.1)
    
    # Set limits and labels
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_zlim(-1, 7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vector Visualization')
    ax.legend()
    
    plt.show()

# =============================================================================
# 7. AI/ML APPLICATIONS
# =============================================================================

def demonstrate_ml_applications():
    """
    Demonstrate vector applications in machine learning and data science.
    """
    print("\n=== AI/ML Applications ===")
    
    # 1. Feature Vectors
    print("\n1. Feature Vectors:")
    house_features = np.array([
        2000,    # square footage
        3,       # bedrooms
        2,       # bathrooms
        2010,    # year built
        250000   # price
    ])
    
    # Normalize features for machine learning
    normalized_features = (house_features - np.mean(house_features)) / np.std(house_features)
    print("Original house features:", house_features)
    print("Normalized house features:", normalized_features)
    
    # 2. Similarity Measures
    print("\n2. Similarity Measures:")
    # Cosine similarity between two documents (represented as word frequency vectors)
    doc1 = np.array([1, 0, 2, 1, 0])  # word frequencies
    doc2 = np.array([0, 1, 1, 0, 2])
    
    cosine_similarity = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
    print("Document 1 (word frequencies):", doc1)
    print("Document 2 (word frequencies):", doc2)
    print("Cosine similarity between documents:", cosine_similarity)
    
    # 3. Gradient Vectors
    print("\n3. Gradient Vectors:")
    # Example: Gradient of a simple function f(x,y) = x² + y²
    def gradient_2d(x, y):
        return np.array([2*x, 2*y])
    
    # Gradient at point (1, 2)
    grad = gradient_2d(1, 2)
    print("Gradient of f(x,y) = x² + y² at (1, 2):", grad)
    print("Gradient magnitude:", np.linalg.norm(grad))

# =============================================================================
# 8. EXERCISES
# =============================================================================

def vector_exercises():
    """
    Demonstrate vector exercises and problem-solving techniques.
    """
    print("\n=== Vector Exercises ===")
    
    # Exercise 1: Vector Operations
    print("\nExercise 1: Vector Operations")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    
    # Calculate: 2a + 3b - c
    result = 2*a + 3*b - c
    print("Given: a =", a, ", b =", b, ", c =", c)
    print("Calculate: 2a + 3b - c")
    print("Result:", result)
    
    # Exercise 2: Vector Properties
    print("\nExercise 2: Vector Properties")
    # Verify the Cauchy-Schwarz inequality: |a·b| ≤ |a|·|b|
    dot_product = np.dot(a, b)
    magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
    
    print("Cauchy-Schwarz inequality verification:")
    print("|a·b| =", abs(dot_product))
    print("|a|·|b| =", magnitude_product)
    print("Inequality holds:", abs(dot_product) <= magnitude_product)
    
    # Exercise 3: Vector Decomposition
    print("\nExercise 3: Vector Decomposition")
    # Decompose vector v into components parallel and perpendicular to u
    v = np.array([3, 4, 5])
    u = np.array([1, 0, 0])
    
    # Parallel component
    u_unit = u / np.linalg.norm(u)
    v_parallel = np.dot(v, u_unit) * u_unit
    
    # Perpendicular component
    v_perpendicular = v - v_parallel
    
    print("Original vector v:", v)
    print("Direction vector u:", u)
    print("Parallel component:", v_parallel)
    print("Perpendicular component:", v_perpendicular)
    print("Sum of components:", v_parallel + v_perpendicular)
    print("Verification (should be zero):", np.dot(v_parallel, v_perpendicular))

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to demonstrate all vector concepts and operations.
    """
    print("VECTORS AND VECTOR OPERATIONS")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Vector Creation
    v1, v2, v3 = demonstrate_vector_creation()
    
    # 2. Vector Operations
    v4 = demonstrate_vector_addition(v1, v2, v3)
    demonstrate_scalar_multiplication(v1, v2)
    demonstrate_vector_subtraction(v1, v2)
    
    # 3. Vector Properties
    demonstrate_vector_magnitude(v1, v2)
    e1, e2, e3 = demonstrate_unit_vectors(v1)
    
    # 4. Dot Product
    demonstrate_dot_product(v1, v2, e1, e2, e3)
    
    # 5. Cross Product
    demonstrate_cross_product(v1, v2, v4)
    
    # 6. AI/ML Applications
    demonstrate_ml_applications()
    
    # 7. Exercises
    vector_exercises()
    
    # 8. Visualizations (uncomment to show plots)
    print("\n=== Visualizations ===")
    print("Uncomment the following lines to show vector visualizations:")
    print("# plot_2d_vectors(v3)")
    print("# plot_3d_vectors(v1, v2, np.cross(v1, v2))")
    
    print("\n" + "=" * 50)
    print("VECTOR OPERATIONS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 