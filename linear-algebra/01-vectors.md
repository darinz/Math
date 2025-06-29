# Vectors and Vector Operations

[![Chapter](https://img.shields.io/badge/Chapter-1-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Vectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-brightgreen.svg)]()

## Introduction

Vectors are fundamental objects in linear algebra that represent both magnitude and direction. In AI/ML and data science, vectors are used to represent data points, features, and parameters in mathematical models. Understanding vectors is crucial for grasping more advanced concepts like matrices, linear transformations, and machine learning algorithms.

### Why Vectors Matter in AI/ML

1. **Feature Representation**: Each data point in a dataset can be represented as a vector where each component corresponds to a feature
2. **Model Parameters**: Neural network weights, regression coefficients, and other model parameters are stored as vectors
3. **Embeddings**: Word embeddings, image embeddings, and other learned representations are vectors
4. **Optimization**: Gradient descent and other optimization algorithms work with vectors of gradients

## What is a Vector?

A vector is an ordered list of numbers (scalars) that can represent:
- **Points in space**: Each component represents a coordinate in n-dimensional space
- **Directions and magnitudes**: Vectors can point in specific directions with specific lengths
- **Features of data points**: Each component represents a different feature or attribute
- **Parameters in machine learning models**: Weights, biases, and other learnable parameters

### Geometric Interpretation

In 2D space, a vector $\vec{v} = [x, y]$ represents:
- A point $(x, y)$ in the plane
- An arrow from the origin $(0, 0)$ to the point $(x, y)$
- A displacement with magnitude $\sqrt{x^2 + y^2}$ and direction $\tan^{-1}(y/x)$

In 3D space, a vector $\vec{v} = [x, y, z]$ represents:
- A point $(x, y, z)$ in 3D space
- An arrow from the origin $(0, 0, 0)$ to the point $(x, y, z)$
- A displacement with magnitude $\sqrt{x^2 + y^2 + z^2}$

## Vector Representation

### Mathematical Notation

A vector $\vec{v}$ in $\mathbb{R}^n$ (n-dimensional real space) is written as:
$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

Where:
- $v_i$ is the $i$-th component of the vector
- $n$ is the dimension of the vector space
- $\mathbb{R}^n$ denotes the set of all n-tuples of real numbers

### Vector Types

1. **Row Vector**: $\vec{v} = [v_1, v_2, \ldots, v_n]$
2. **Column Vector**: $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$

In linear algebra, we typically work with column vectors, but row vectors are useful for certain operations.

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating vectors
v1 = np.array([1, 2, 3])  # 3D vector
v2 = np.array([4, 5, 6])
v3 = np.array([1, 2])     # 2D vector

print("Vector v1:", v1)
print("Vector v2:", v2)
print("Vector v3:", v3)
print("Shape of v1:", v1.shape)
print("Dimension of v1:", v1.ndim)

# Creating vectors with different methods
zeros_vector = np.zeros(5)  # Zero vector
ones_vector = np.ones(4)    # Vector of ones
random_vector = np.random.randn(3)  # Random vector
linspace_vector = np.linspace(0, 10, 5)  # Equally spaced values

print("Zero vector:", zeros_vector)
print("Ones vector:", ones_vector)
print("Random vector:", random_vector)
print("Linspace vector:", linspace_vector)

# Vector types and data types
print("Data type of v1:", v1.dtype)
print("Is v1 a 1D array?", v1.ndim == 1)
print("Length of v1:", len(v1))
```

## Vector Operations

### 1. Vector Addition

**Mathematical Definition:**
Vector addition is performed component-wise:
$$\vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Geometric Interpretation:**
- In 2D/3D: Vector addition follows the parallelogram law
- The sum vector represents the diagonal of the parallelogram formed by the two vectors
- This is equivalent to placing the tail of the second vector at the head of the first

**Properties:**
- **Commutative**: $\vec{a} + \vec{b} = \vec{b} + \vec{a}$
- **Associative**: $(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c})$
- **Identity**: $\vec{a} + \vec{0} = \vec{a}$ (where $\vec{0}$ is the zero vector)
- **Inverse**: $\vec{a} + (-\vec{a}) = \vec{0}$

**Python Implementation:**
```python
# Vector addition
result = v1 + v2
print("v1 + v2 =", result)

# Element-wise addition
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
print("Commutative property (v1 + v2 == v2 + v1):", np.array_equal(v1 + v2, v2 + v1))
print("Associative property check:", np.array_equal((v1 + v2) + v4, v1 + (v2 + v4)))
```

### 2. Scalar Multiplication

**Mathematical Definition:**
Scalar multiplication multiplies each component by a scalar:
$$c\vec{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

**Geometric Interpretation:**
- If $c > 0$: The vector is scaled by factor $c$ in the same direction
- If $c < 0$: The vector is scaled by factor $|c|$ in the opposite direction
- If $c = 0$: The result is the zero vector
- If $|c| > 1$: The vector is stretched
- If $|c| < 1$: The vector is compressed

**Properties:**
- **Distributive over vector addition**: $c(\vec{a} + \vec{b}) = c\vec{a} + c\vec{b}$
- **Distributive over scalar addition**: $(c + d)\vec{a} = c\vec{a} + d\vec{a}$
- **Associative**: $(cd)\vec{a} = c(d\vec{a})$
- **Identity**: $1\vec{a} = \vec{a}$

**Python Implementation:**
```python
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
print("Distributive over vector addition:", 
      np.array_equal(2 * (v1 + v2), 2 * v1 + 2 * v2))
print("Associative property:", 
      np.array_equal((2 * 3) * v1, 2 * (3 * v1)))

# Scaling effects
scales = [0.5, 1, 2, -1]
for scale in scales:
    scaled = scale * v1
    print(f"{scale} * v1 = {scaled}, magnitude = {np.linalg.norm(scaled):.2f}")
```

### 3. Vector Subtraction

**Mathematical Definition:**
Vector subtraction is defined as addition with the negative:
$$\vec{a} - \vec{b} = \vec{a} + (-\vec{b}) = \begin{bmatrix} a_1 - b_1 \\ a_2 - b_2 \\ \vdots \\ a_n - b_n \end{bmatrix}$$

**Geometric Interpretation:**
- $\vec{a} - \vec{b}$ represents the vector from the tip of $\vec{b}$ to the tip of $\vec{a}$
- This is useful for finding displacement vectors and differences between points

**Python Implementation:**
```python
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
```

## Vector Properties

### 1. Magnitude (Length)

**Mathematical Definition:**
The magnitude (or length) of a vector is given by the Euclidean norm:
$$|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Geometric Interpretation:**
- In 2D: $|\vec{v}| = \sqrt{x^2 + y^2}$ (Pythagorean theorem)
- In 3D: $|\vec{v}| = \sqrt{x^2 + y^2 + z^2}$
- The magnitude represents the "size" or "length" of the vector

**Properties:**
- $|\vec{v}| \geq 0$ (non-negative)
- $|\vec{v}| = 0$ if and only if $\vec{v} = \vec{0}$
- $|c\vec{v}| = |c| \cdot |\vec{v}|$ (scalar multiplication)
- Triangle inequality: $|\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$

**Python Implementation:**
```python
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

print(f"L1 norm: {l1_norm:.2f}")
print(f"L2 norm: {l2_norm:.2f}")
print(f"L∞ norm: {l_inf_norm:.2f}")

# Verification of properties
print("Non-negative:", magnitude_v1 >= 0)
print("Scalar multiplication:", np.linalg.norm(2 * v1) == 2 * magnitude_v1)
print("Triangle inequality:", np.linalg.norm(v1 + v2) <= magnitude_v1 + np.linalg.norm(v2))
```

### 2. Unit Vector

A unit vector has magnitude 1 and points in the same direction as the original vector.

**Mathematical Definition:**
$$\hat{v} = \frac{\vec{v}}{|\vec{v}|}$$

**Geometric Interpretation:**
- Unit vectors are used to represent pure directions
- Any vector can be written as $\vec{v} = |\vec{v}| \cdot \hat{v}$
- Unit vectors are fundamental in coordinate systems and basis representations

**Properties:**
- $|\hat{v}| = 1$
- $\hat{v}$ points in the same direction as $\vec{v}$
- If $\vec{v} = \vec{0}$, the unit vector is undefined

**Python Implementation:**
```python
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

print("Standard unit vectors:")
print("e1 (i-hat):", e1)
print("e2 (j-hat):", e2)
print("e3 (k-hat):", e3)
print("Magnitudes:", [np.linalg.norm(e) for e in [e1, e2, e3]])
```

## Dot Product (Inner Product)

### Mathematical Definition
The dot product of two vectors is a scalar defined as:
$$\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + \cdots + a_nb_n = |\vec{a}||\vec{b}|\cos\theta$$

Where $\theta$ is the angle between the vectors.

### Geometric Interpretation
- The dot product measures the projection of one vector onto another
- $\vec{a} \cdot \vec{b} = |\vec{a}| \cdot |\vec{b}| \cos\theta$
- If vectors are perpendicular: $\vec{a} \cdot \vec{b} = 0$
- If vectors are parallel: $\vec{a} \cdot \vec{b} = \pm|\vec{a}||\vec{b}|$

### Properties
- **Commutative**: $\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$
- **Distributive**: $\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}$
- **Scalar multiplication**: $(c\vec{a}) \cdot \vec{b} = c(\vec{a} \cdot \vec{b})$
- **Positive definite**: $\vec{a} \cdot \vec{a} = |\vec{a}|^2 \geq 0$

### Python Implementation
```python
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
print("v1 · e1 =", np.dot(v1, e1))  # x-component
print("v1 · e2 =", np.dot(v1, e2))  # y-component
print("v1 · e3 =", np.dot(v1, e3))  # z-component
```

## Cross Product (3D Vectors Only)

### Mathematical Definition
The cross product of two 3D vectors is a vector perpendicular to both:
$$\vec{a} \times \vec{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}$$

### Geometric Interpretation
- The cross product produces a vector perpendicular to both input vectors
- The direction follows the right-hand rule
- The magnitude is $|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta$
- The cross product is used to find normal vectors, areas, and volumes

### Properties
- **Anti-commutative**: $\vec{a} \times \vec{b} = -(\vec{b} \times \vec{a})$
- **Distributive**: $\vec{a} \times (\vec{b} + \vec{c}) = \vec{a} \times \vec{b} + \vec{a} \times \vec{c}$
- **Scalar multiplication**: $(c\vec{a}) \times \vec{b} = c(\vec{a} \times \vec{b})$
- **Perpendicular to both vectors**: $\vec{a} \cdot (\vec{a} \times \vec{b}) = 0$ and $\vec{b} \cdot (\vec{a} \times \vec{b}) = 0$

### Python Implementation
```python
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
expected_magnitude = np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(angle_rad)
print("Cross product magnitude:", cross_magnitude)
print("Expected magnitude:", expected_magnitude)

# Verification of properties
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
```

## Vector Visualization

```python
# 2D Vector Visualization
def plot_2d_vectors():
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

# 3D Vector Visualization
def plot_3d_vectors():
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

# Run visualizations
plot_2d_vectors()
plot_3d_vectors()
```

## Applications in AI/ML

### 1. Feature Vectors
```python
# Example: Feature vector for a house
house_features = np.array([
    2000,    # square footage
    3,       # bedrooms
    2,       # bathrooms
    2010,    # year built
    250000   # price
])

# Normalize features for machine learning
normalized_features = (house_features - np.mean(house_features)) / np.std(house_features)
print("Normalized house features:", normalized_features)
```

### 2. Similarity Measures
```python
# Cosine similarity between two documents (represented as word frequency vectors)
doc1 = np.array([1, 0, 2, 1, 0])  # word frequencies
doc2 = np.array([0, 1, 1, 0, 2])

cosine_similarity = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
print("Cosine similarity between documents:", cosine_similarity)
```

### 3. Gradient Vectors
```python
# Example: Gradient of a simple function f(x,y) = x^2 + y^2
def gradient_2d(x, y):
    return np.array([2*x, 2*y])

# Gradient at point (1, 2)
grad = gradient_2d(1, 2)
print("Gradient at (1, 2):", grad)
print("Gradient magnitude:", np.linalg.norm(grad))
```

## Exercises

### Exercise 1: Vector Operations
```python
# Given vectors a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9]
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

# Calculate: 2a + 3b - c
result = 2*a + 3*b - c
print("2a + 3b - c =", result)
```

### Exercise 2: Vector Properties
```python
# Verify the Cauchy-Schwarz inequality: |a·b| ≤ |a|·|b|
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)
magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)

print("|a·b| =", abs(dot_product))
print("|a|·|b| =", magnitude_product)
print("Cauchy-Schwarz inequality holds:", abs(dot_product) <= magnitude_product)
```

### Exercise 3: Vector Decomposition
```python
# Decompose vector v into components parallel and perpendicular to u
v = np.array([3, 4, 5])
u = np.array([1, 0, 0])

# Parallel component
u_unit = u / np.linalg.norm(u)
v_parallel = np.dot(v, u_unit) * u_unit

# Perpendicular component
v_perpendicular = v - v_parallel

print("Original vector v:", v)
print("Parallel component:", v_parallel)
print("Perpendicular component:", v_perpendicular)
print("Sum of components:", v_parallel + v_perpendicular)
print("Verification (should be zero):", np.dot(v_parallel, v_perpendicular))
```

## Summary

In this chapter, we've covered:

1. **Vector Fundamentals**: Definition, representation, and geometric interpretation
2. **Basic Operations**: Addition, scalar multiplication, and subtraction with detailed properties
3. **Vector Properties**: Magnitude, unit vectors, and their geometric meaning
4. **Dot Product**: Definition, properties, and applications in similarity and projection
5. **Cross Product**: 3D vector operation for finding perpendicular vectors and areas
6. **Visualization**: 2D and 3D plotting of vectors and their operations
7. **AI/ML Applications**: Feature vectors, similarity measures, and gradients

### Key Takeaways:
- Vectors are fundamental for representing data and mathematical objects
- Vector operations have both algebraic and geometric interpretations
- The dot product measures similarity and projection
- The cross product (in 3D) produces perpendicular vectors
- Understanding vectors is essential for linear algebra and machine learning

### Next Steps:
In the next chapter, we'll explore matrices, which are collections of vectors that enable more complex linear transformations and operations. 