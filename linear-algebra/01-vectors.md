# Vectors and Vector Operations

[![Chapter](https://img.shields.io/badge/Chapter-1-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Vectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-brightgreen.svg)]()

## Introduction

Vectors are fundamental objects in linear algebra that represent both magnitude and direction. In AI/ML and data science, vectors are used to represent data points, features, and parameters in mathematical models.

## What is a Vector?

A vector is an ordered list of numbers (scalars) that can represent:
- Points in space
- Directions and magnitudes
- Features of data points
- Parameters in machine learning models

## Vector Representation

### Mathematical Notation
A vector $\vec{v}$ in $\mathbb{R}^n$ is written as:
$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

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
```

## Vector Operations

### 1. Vector Addition

**Mathematical Definition:**
$$\vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Python Implementation:**
```python
# Vector addition
result = v1 + v2
print("v1 + v2 =", result)

# Element-wise addition
result_manual = np.array([v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]])
print("Manual addition:", result_manual)
```

### 2. Scalar Multiplication

**Mathematical Definition:**
$$c\vec{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

**Python Implementation:**
```python
# Scalar multiplication
scalar = 2.5
result = scalar * v1
print(f"{scalar} * v1 =", result)

# Negative vector
negative_v1 = -v1
print("-v1 =", negative_v1)
```

### 3. Vector Subtraction

**Mathematical Definition:**
$$\vec{a} - \vec{b} = \vec{a} + (-\vec{b})$$

**Python Implementation:**
```python
# Vector subtraction
result = v1 - v2
print("v1 - v2 =", result)

# Equivalent to addition with negative
result_equivalent = v1 + (-v2)
print("v1 + (-v2) =", result_equivalent)
```

## Vector Properties

### 1. Magnitude (Length)

**Mathematical Definition:**
$$|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Python Implementation:**
```python
# Vector magnitude
magnitude_v1 = np.linalg.norm(v1)
print("Magnitude of v1:", magnitude_v1)

# Manual calculation
magnitude_manual = np.sqrt(np.sum(v1**2))
print("Manual magnitude calculation:", magnitude_manual)
```

### 2. Unit Vector

A unit vector has magnitude 1 and points in the same direction as the original vector.

**Mathematical Definition:**
$$\hat{v} = \frac{\vec{v}}{|\vec{v}|}$$

**Python Implementation:**
```python
# Unit vector
unit_v1 = v1 / np.linalg.norm(v1)
print("Unit vector of v1:", unit_v1)
print("Magnitude of unit vector:", np.linalg.norm(unit_v1))
```

## Dot Product (Inner Product)

### Mathematical Definition
$$\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + \cdots + a_nb_n = |\vec{a}||\vec{b}|\cos\theta$$

### Properties
- Commutative: $\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$
- Distributive: $\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}$
- Scalar multiplication: $(c\vec{a}) \cdot \vec{b} = c(\vec{a} \cdot \vec{b})$

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
```

## Cross Product (3D Vectors Only)

### Mathematical Definition
$$\vec{a} \times \vec{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}$$

### Properties
- Anti-commutative: $\vec{a} \times \vec{b} = -(\vec{b} \times \vec{a})$
- Perpendicular to both vectors
- Magnitude: $|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta$

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
    
    # Vector sum
    v_sum = v3 + v4
    ax.quiver(origin[0], origin[1], v_sum[0], v_sum[1], 
              angles='xy', scale_units='xy', scale=1, color='green', label='v3 + v4')
    
    # Set limits and grid
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('2D Vector Visualization')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

# 3D Vector Visualization
def plot_3d_vectors():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin
    origin = np.array([0, 0, 0])
    
    # Plot vectors
    ax.quiver(origin[0], origin[1], origin[2], 
              v1[0], v1[1], v1[2], color='blue', label='v1')
    ax.quiver(origin[0], origin[1], origin[2], 
              v2[0], v2[1], v2[2], color='red', label='v2')
    
    # Vector sum
    v_sum = v1 + v2
    ax.quiver(origin[0], origin[1], origin[2], 
              v_sum[0], v_sum[1], v_sum[2], color='green', label='v1 + v2')
    
    # Set limits
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_zlim(-1, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('3D Vector Visualization')
    plt.show()

# Run visualizations
plot_2d_vectors()
plot_3d_vectors()
```

## Applications in Data Science

### 1. Feature Vectors
```python
# Example: Feature vector for a house
house_features = np.array([2000,  # square feet
                          3,      # bedrooms
                          2,      # bathrooms
                          2010,   # year built
                          250000]) # price

print("House feature vector:", house_features)
```

### 2. Data Normalization
```python
# Normalize feature vector
normalized_features = house_features / np.linalg.norm(house_features)
print("Normalized features:", normalized_features)
print("Magnitude of normalized vector:", np.linalg.norm(normalized_features))
```

### 3. Similarity Calculation
```python
# Calculate similarity between two data points
house1 = np.array([2000, 3, 2, 2010, 250000])
house2 = np.array([1800, 3, 2, 2008, 230000])

# Cosine similarity
cosine_sim = np.dot(house1, house2) / (np.linalg.norm(house1) * np.linalg.norm(house2))
print(f"Cosine similarity between houses: {cosine_sim:.4f}")
```

## Exercises

1. **Vector Operations**: Create two 4D vectors and compute their sum, difference, dot product, and magnitudes.

2. **Unit Vectors**: Find the unit vector in the direction of $\vec{v} = [3, 4, 0, 1]$.

3. **Angle Calculation**: Calculate the angle between vectors $\vec{a} = [1, 2, 3]$ and $\vec{b} = [4, 5, 6]$.

4. **Data Similarity**: Create feature vectors for three different products and calculate their pairwise similarities.

5. **Visualization**: Plot the vectors $\vec{v} = [2, 3]$ and $\vec{w} = [1, 4]$ and their sum.

## Solutions

```python
# Exercise 1: Vector Operations
v1_4d = np.array([1, 2, 3, 4])
v2_4d = np.array([5, 6, 7, 8])

print("Sum:", v1_4d + v2_4d)
print("Difference:", v1_4d - v2_4d)
print("Dot product:", np.dot(v1_4d, v2_4d))
print("Magnitude v1:", np.linalg.norm(v1_4d))
print("Magnitude v2:", np.linalg.norm(v2_4d))

# Exercise 2: Unit Vector
v = np.array([3, 4, 0, 1])
unit_v = v / np.linalg.norm(v)
print("Unit vector:", unit_v)

# Exercise 3: Angle Calculation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
print(f"Angle: {angle:.2f} degrees")
```

## Key Takeaways

- Vectors represent both magnitude and direction
- Vector operations (addition, scalar multiplication) are fundamental
- Dot product measures similarity and angle between vectors
- Cross product (3D) creates perpendicular vectors
- Vectors are essential for representing data in machine learning
- Normalization and similarity calculations are common in data science

## Next Chapter

In the next chapter, we'll explore matrices and matrix operations, which extend vector concepts to 2D arrays and are crucial for linear transformations and solving systems of equations. 