# Linear Transformations

[![Chapter](https://img.shields.io/badge/Chapter-3-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Linear_Transformations-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)]()

## Introduction

Linear transformations are functions that preserve vector addition and scalar multiplication. They are fundamental in understanding how matrices transform vectors and spaces, which is crucial for machine learning algorithms.

## What is a Linear Transformation?

A function T: ℝⁿ → ℝᵐ is linear if it satisfies:
1. T(u + v) = T(u) + T(v) (additivity)
2. T(cu) = cT(u) (homogeneity)

### Matrix Representation
Every linear transformation can be represented by a matrix A:
T(x) = Ax

## Common Linear Transformations

### Scaling (Dilation/Contraction)
Scaling transforms vectors by multiplying each component by a scalar.

```python
import numpy as np
import matplotlib.pyplot as plt

# Scaling matrix
scale_factor = 2
S = np.array([[scale_factor, 0],
              [0, scale_factor]])

print("Scaling matrix (scale by 2):")
print(S)

# Apply transformation
v = np.array([1, 1])
v_scaled = S @ v
print(f"\nOriginal vector: {v}")
print(f"Scaled vector: {v_scaled}")
```

### Rotation
Rotation transforms vectors by rotating them around the origin.

```python
# Rotation matrix (45 degrees)
theta = np.pi/4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

print("Rotation matrix (45°):")
print(R)

# Apply rotation
v = np.array([1, 0])
v_rotated = R @ v
print(f"\nOriginal vector: {v}")
print(f"Rotated vector: {v_rotated}")
```

### Reflection
Reflection flips vectors across a line or plane.

```python
# Reflection across x-axis
R_x = np.array([[1, 0],
                [0, -1]])

# Reflection across y-axis
R_y = np.array([[-1, 0],
                [0, 1]])

print("Reflection across x-axis:")
print(R_x)
print("\nReflection across y-axis:")
print(R_y)

# Apply reflections
v = np.array([1, 1])
v_reflected_x = R_x @ v
v_reflected_y = R_y @ v

print(f"\nOriginal vector: {v}")
print(f"Reflected across x-axis: {v_reflected_x}")
print(f"Reflected across y-axis: {v_reflected_y}")
```

### Shear
Shear transforms vectors by adding a multiple of one component to another.

```python
# Shear transformation
shear_factor = 0.5
H = np.array([[1, shear_factor],
              [0, 1]])

print("Shear matrix:")
print(H)

# Apply shear
v = np.array([1, 1])
v_sheared = H @ v
print(f"\nOriginal vector: {v}")
print(f"Sheared vector: {v_sheared}")
```

## Visualization of Transformations

```python
def plot_transformation(matrix, title, vectors=None):
    if vectors is None:
        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    
    plt.figure(figsize=(8, 6))
    
    # Original vectors
    for i, v in enumerate(vectors):
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                  color='blue', alpha=0.5, label=f'Original {i+1}' if i == 0 else "")
    
    # Transformed vectors
    for i, v in enumerate(vectors):
        v_transformed = matrix @ v
        plt.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1,
                  color='red', alpha=0.7, label=f'Transformed {i+1}' if i == 0 else "")
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

# Visualize different transformations
plot_transformation(S, 'Scaling Transformation')
plot_transformation(R, 'Rotation Transformation')
plot_transformation(R_x, 'Reflection Across X-axis')
plot_transformation(H, 'Shear Transformation')
```

## Composition of Transformations

Linear transformations can be combined by matrix multiplication.

```python
# Combine rotation and scaling
combined = S @ R  # Scale then rotate
print("Combined transformation (scale then rotate):")
print(combined)

# Apply combined transformation
v = np.array([1, 0])
v_combined = combined @ v
print(f"\nOriginal vector: {v}")
print(f"Combined transformation: {v_combined}")

# Note: order matters!
combined_reverse = R @ S  # Rotate then scale
print(f"\nReverse order (rotate then scale): {combined_reverse @ v}")
```

## Properties of Linear Transformations

### Preserving Linear Combinations
```python
# Test linearity properties
def test_linearity(matrix, u, v, c):
    # Test additivity: T(u + v) = T(u) + T(v)
    additivity = matrix @ (u + v) == (matrix @ u) + (matrix @ v)
    
    # Test homogeneity: T(cu) = cT(u)
    homogeneity = matrix @ (c * u) == c * (matrix @ u)
    
    return additivity.all(), homogeneity.all()

# Test with rotation matrix
u = np.array([1, 2])
v = np.array([3, 4])
c = 2

additivity, homogeneity = test_linearity(R, u, v, c)
print(f"Additivity preserved: {additivity}")
print(f"Homogeneity preserved: {homogeneity}")
```

### Determinant and Area/Volume
The determinant of a transformation matrix tells us how it affects area/volume.

```python
# Calculate determinants
det_S = np.linalg.det(S)
det_R = np.linalg.det(R)
det_H = np.linalg.det(H)

print(f"Determinant of scaling matrix: {det_S}")
print(f"Determinant of rotation matrix: {det_R}")
print(f"Determinant of shear matrix: {det_H}")

# Area scaling factor
print(f"\nArea scaling factor for scaling: {det_S}")
print(f"Area scaling factor for rotation: {det_R}")
print(f"Area scaling factor for shear: {det_H}")
```

## Applications in Machine Learning

### Feature Transformations
```python
# Example: Feature scaling and rotation
features = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

# Scale features
scaled_features = features @ S
print("Original features:")
print(features)
print("\nScaled features:")
print(scaled_features)

# Rotate features
rotated_features = features @ R
print("\nRotated features:")
print(rotated_features)
```

### Data Augmentation
```python
# Data augmentation with transformations
original_data = np.array([[1, 1], [2, 2], [3, 3]])

# Create augmented dataset
augmented_data = []
for point in original_data:
    # Original point
    augmented_data.append(point)
    
    # Rotated point
    augmented_data.append(R @ point)
    
    # Scaled point
    augmented_data.append(S @ point)

augmented_data = np.array(augmented_data)
print("Original data:")
print(original_data)
print("\nAugmented data:")
print(augmented_data)
```

### Principal Component Analysis (PCA)
PCA involves finding the principal components (eigenvectors) and transforming data.

```python
from sklearn.decomposition import PCA

# Create sample data
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 1], [1, 1]])

# Apply PCA
pca = PCA(n_components=2)
data_transformed = pca.fit_transform(data)

print("PCA transformation matrix:")
print(pca.components_)
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")

# Visualize PCA transformation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(data_transformed[:, 0], data_transformed[:, 1], alpha=0.6)
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
```

## Exercises

### Exercise 1: Basic Transformations
```python
# Create transformation matrices and apply them
# Your code here:
# 1. Create a rotation matrix for 30 degrees
# 2. Create a scaling matrix that scales by 1.5
# 3. Apply both transformations to vector [2, 1]
# 4. Compare the results of different orders
```

### Exercise 2: Transformation Properties
```python
# Test linearity properties
# Your code here:
# 1. Create a 2×2 matrix of your choice
# 2. Test if it preserves additivity
# 3. Test if it preserves homogeneity
# 4. Calculate its determinant
```

### Exercise 3: Data Transformation
```python
# Transform a dataset
data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

# Your code here:
# 1. Apply a rotation transformation
# 2. Apply a scaling transformation
# 3. Combine both transformations
# 4. Visualize the results
```

## Solutions

### Solution 1: Basic Transformations
```python
# 1. Rotation matrix (30 degrees)
theta_30 = np.pi/6
R_30 = np.array([[np.cos(theta_30), -np.sin(theta_30)],
                 [np.sin(theta_30), np.cos(theta_30)]])
print("Rotation matrix (30°):")
print(R_30)

# 2. Scaling matrix (1.5)
S_15 = np.array([[1.5, 0], [0, 1.5]])
print("\nScaling matrix (1.5):")
print(S_15)

# 3. Apply transformations
v = np.array([2, 1])
v_rotated = R_30 @ v
v_scaled = S_15 @ v
v_combined = S_15 @ R_30 @ v

print(f"\nOriginal vector: {v}")
print(f"Rotated: {v_rotated}")
print(f"Scaled: {v_scaled}")
print(f"Combined (scale then rotate): {v_combined}")

# 4. Different order
v_combined_reverse = R_30 @ S_15 @ v
print(f"Combined (rotate then scale): {v_combined_reverse}")
```

### Solution 2: Transformation Properties
```python
# 1. Create matrix
A = np.array([[2, 1], [1, 3]])
print("Matrix A:")
print(A)

# 2. Test additivity
u = np.array([1, 2])
v = np.array([3, 4])
additivity = np.allclose(A @ (u + v), A @ u + A @ v)
print(f"\nAdditivity preserved: {additivity}")

# 3. Test homogeneity
c = 2
homogeneity = np.allclose(A @ (c * u), c * (A @ u))
print(f"Homogeneity preserved: {homogeneity}")

# 4. Determinant
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")
```

### Solution 3: Data Transformation
```python
# 1. Rotation transformation
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
data_rotated = data @ R.T

# 2. Scaling transformation
S = np.array([[2, 0], [0, 1]])
data_scaled = data @ S

# 3. Combined transformation
combined = S @ R
data_combined = data @ combined.T

# 4. Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.axis('equal')

plt.subplot(1, 4, 2)
plt.scatter(data_rotated[:, 0], data_rotated[:, 1])
plt.title('Rotated Data')
plt.axis('equal')

plt.subplot(1, 4, 3)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1])
plt.title('Scaled Data')
plt.axis('equal')

plt.subplot(1, 4, 4)
plt.scatter(data_combined[:, 0], data_combined[:, 1])
plt.title('Combined Transformation')
plt.axis('equal')

plt.tight_layout()
plt.show()
```

## Summary

In this chapter, we covered:
- Definition and properties of linear transformations
- Common transformations (scaling, rotation, reflection, shear)
- Matrix representation of transformations
- Composition of transformations
- Applications in machine learning
- Visualization techniques

Linear transformations are fundamental for understanding how data is processed and transformed in machine learning algorithms.

## Next Steps

In the next chapter, we'll explore eigenvalues and eigenvectors, which are crucial for understanding matrix behavior and applications like PCA and spectral clustering. 