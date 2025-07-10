# Vectors and Vector Operations

[![Chapter](https://img.shields.io/badge/Chapter-1-blue.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Vectors-green.svg)]()
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-brightgreen.svg)]()

## Introduction

Vectors are fundamental objects in linear algebra that represent both magnitude and direction. In AI/ML and data science, vectors are used to represent data points, features, and parameters in mathematical models. Understanding vectors is crucial for grasping more advanced concepts like matrices, linear transformations, and machine learning algorithms.

### Why Vectors Matter in AI/ML

1. **Feature Representation**: Each data point in a dataset can be represented as a vector where each component corresponds to a feature.
2. **Model Parameters**: Neural network weights, regression coefficients, and other model parameters are stored as vectors.
3. **Embeddings**: Word embeddings, image embeddings, and other learned representations are vectors.
4. **Optimization**: Gradient descent and other optimization algorithms work with vectors of gradients.

## What is a Vector?

A vector is an ordered list of numbers (scalars) that can represent:
- **Points in space**: Each component represents a coordinate in $`n`$-dimensional space.
- **Directions and magnitudes**: Vectors can point in specific directions with specific lengths.
- **Features of data points**: Each component represents a different feature or attribute.
- **Parameters in machine learning models**: Weights, biases, and other learnable parameters.

### Geometric Interpretation

In 2D space, a vector $`\vec{v} = [x, y]`$ represents:
- A point $`(x, y)`$ in the plane.
- An arrow from the origin $`(0, 0)`$ to the point $`(x, y)`$.
- A displacement with magnitude $`\sqrt{x^2 + y^2}`$ and direction $`\tan^{-1}(y/x)`$.

In 3D space, a vector $`\vec{v} = [x, y, z]`$ represents:
- A point $`(x, y, z)`$ in 3D space.
- An arrow from the origin $`(0, 0, 0)`$ to the point $`(x, y, z)`$.
- A displacement with magnitude $`\sqrt{x^2 + y^2 + z^2}`$.

## Vector Representation

### Mathematical Notation

A vector $`\vec{v}`$ in $`\mathbb{R}^n`$ (n-dimensional real space) is written as:
```math
\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```
Where:
- $`v_i`$ is the $`i`$-th component of the vector.
- $`n`$ is the dimension of the vector space.
- $`\mathbb{R}^n`$ denotes the set of all n-tuples of real numbers.

### Vector Types

1. **Row Vector**: $`\vec{v} = [v_1, v_2, \ldots, v_n]`$
2. **Column Vector**: 
```math
\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```
In linear algebra, we typically work with column vectors, but row vectors are useful for certain operations.

## Vector Operations

### 1. Vector Addition

**Mathematical Definition:**
Vector addition is performed component-wise:
```math
\vec{a} + \vec{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}
```
**Geometric Interpretation:**
- In 2D/3D: Vector addition follows the parallelogram law.
- The sum vector represents the diagonal of the parallelogram formed by the two vectors.
- This is equivalent to placing the tail of the second vector at the head of the first.

**Properties:**
- **Commutative**: $`\vec{a} + \vec{b} = \vec{b} + \vec{a}`$
- **Associative**: $`(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c})`$
- **Identity**: $`\vec{a} + \vec{0} = \vec{a}`$ (where $`\vec{0}`$ is the zero vector)
- **Inverse**: $`\vec{a} + (-\vec{a}) = \vec{0}`$

### 2. Scalar Multiplication

**Mathematical Definition:**
Scalar multiplication multiplies each component by a scalar:
```math
c\vec{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}
```
**Geometric Interpretation:**
- If $`c > 0`$: The vector is scaled by factor $`c`$ in the same direction.
- If $`c < 0`$: The vector is scaled by factor $`|c|`$ in the opposite direction.
- If $`c = 0`$: The result is the zero vector.
- If $`|c| > 1`$: The vector is stretched.
- If $`|c| < 1`$: The vector is compressed.

**Properties:**
- **Distributive over vector addition**: $`c(\vec{a} + \vec{b}) = c\vec{a} + c\vec{b}`$
- **Distributive over scalar addition**: $`(c + d)\vec{a} = c\vec{a} + d\vec{a}`$
- **Associative**: $`(cd)\vec{a} = c(d\vec{a})`$
- **Identity**: $`1\vec{a} = \vec{a}`$

### 3. Vector Subtraction

**Mathematical Definition:**
Vector subtraction is defined as addition with the negative:
```math
\vec{a} - \vec{b} = \vec{a} + (-\vec{b}) = \begin{bmatrix} a_1 - b_1 \\ a_2 - b_2 \\ \vdots \\ a_n - b_n \end{bmatrix}
```
**Geometric Interpretation:**
- $`\vec{a} - \vec{b}`$ represents the vector from the tip of $`\vec{b}`$ to the tip of $`\vec{a}`$.
- This is useful for finding displacement vectors and differences between points.

## Vector Properties

### 1. Magnitude (Length)

**Mathematical Definition:**
The magnitude (or length) of a vector is given by the Euclidean norm:
```math
|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
```
**Geometric Interpretation:**
- In 2D: $`|\vec{v}| = \sqrt{x^2 + y^2}`$ (Pythagorean theorem)
- In 3D: $`|\vec{v}| = \sqrt{x^2 + y^2 + z^2}`$
- The magnitude represents the "size" or "length" of the vector.

**Properties:**
- $`|\vec{v}| \geq 0`$ (non-negative)
- $`|\vec{v}| = 0`$ if and only if $`\vec{v} = \vec{0}`$
- $`|c\vec{v}| = |c| \cdot |\vec{v}|`$ (scalar multiplication)
- Triangle inequality: $`|\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|`$

### 2. Unit Vector

A unit vector has magnitude 1 and points in the same direction as the original vector.

**Mathematical Definition:**
```math
\hat{v} = \frac{\vec{v}}{|\vec{v}|}
```
**Geometric Interpretation:**
- Unit vectors are used to represent pure directions.
- Any vector can be written as $`\vec{v} = |\vec{v}| \cdot \hat{v}`$.
- Unit vectors are fundamental in coordinate systems and basis representations.

**Properties:**
- $`|\hat{v}| = 1`$
- $`\hat{v}`$ points in the same direction as $`\vec{v}`$
- If $`\vec{v} = \vec{0}`$, the unit vector is undefined.

## Dot Product (Inner Product)

### Mathematical Definition
The dot product of two vectors is a scalar defined as:
```math
\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + \cdots + a_nb_n = |\vec{a}||\vec{b}|\cos\theta
```
Where $`\theta`$ is the angle between the vectors.

### Geometric Interpretation
- The dot product measures the projection of one vector onto another.
- $`\vec{a} \cdot \vec{b} = |\vec{a}| \cdot |\vec{b}| \cos\theta`$
- If vectors are perpendicular: $`\vec{a} \cdot \vec{b} = 0`$
- If vectors are parallel: $`\vec{a} \cdot \vec{b} = \pm|\vec{a}||\vec{b}|`$

### Properties
- **Commutative**: $`\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}`$
- **Distributive**: $`\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}`$
- **Scalar multiplication**: $`(c\vec{a}) \cdot \vec{b} = c(\vec{a} \cdot \vec{b})`$
- **Positive definite**: $`\vec{a} \cdot \vec{a} = |\vec{a}|^2 \geq 0`$

## Cross Product (3D Vectors Only)

### Mathematical Definition
The cross product of two 3D vectors is a vector perpendicular to both:
```math
\vec{a} \times \vec{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}
```
### Geometric Interpretation
- The cross product produces a vector perpendicular to both input vectors.
- The direction follows the right-hand rule.
- The magnitude is $`|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta`$
- The cross product is used to find normal vectors, areas, and volumes.

### Properties
- **Anti-commutative**: $`\vec{a} \times \vec{b} = -(\vec{b} \times \vec{a})`$
- **Distributive**: $`\vec{a} \times (\vec{b} + \vec{c}) = \vec{a} \times \vec{b} + \vec{a} \times \vec{c}`$
- **Scalar multiplication**: $`(c\vec{a}) \times \vec{b} = c(\vec{a} \times \vec{b})`$
- **Perpendicular to both vectors**: $`\vec{a} \cdot (\vec{a} \times \vec{b}) = 0`$ and $`\vec{b} \cdot (\vec{a} \times \vec{b}) = 0`$

## Vector Visualization

*Visualizing vectors can help build intuition. In 2D, vectors are arrows in the plane. In 3D, they are arrows in space. Try sketching vectors and their sums, differences, and scalar multiples to see these operations geometrically.*

## Applications in AI/ML

### 1. Feature Vectors

Feature vectors represent data points in machine learning, where each component corresponds to a different feature or attribute.

### 2. Similarity Measures

Cosine similarity measures the angle between vectors, providing a normalized measure of similarity between data points.

### 3. Gradient Vectors

Gradient vectors point in the direction of steepest ascent of a function and are fundamental in optimization algorithms.

## Exercises

### Exercise 1: Vector Operations

Given vectors $`\vec{a} = [1, 2, 3]`$, $`\vec{b} = [4, 5, 6]`$, $`\vec{c} = [7, 8, 9]`$, calculate $`2\vec{a} + 3\vec{b} - \vec{c}`$.

### Exercise 2: Vector Properties

Verify the Cauchy-Schwarz inequality: $`|\vec{a} \cdot \vec{b}| \leq |\vec{a}| \cdot |\vec{b}|`$ for vectors $`\vec{a} = [1, 2, 3]`$ and $`\vec{b} = [4, 5, 6]`$.

### Exercise 3: Vector Decomposition

Decompose vector $`\vec{v} = [3, 4, 5]`$ into components parallel and perpendicular to $`\vec{u} = [1, 0, 0]`$.

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