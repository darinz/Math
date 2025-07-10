# Multivariate Statistics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![FactorAnalyzer](https://img.shields.io/badge/FactorAnalyzer-0.4+-blue.svg)](https://factor-analyzer.readthedocs.io/)

## Introduction

Multivariate statistics deals with the analysis of data with multiple variables. This chapter covers dimensionality reduction, clustering, and multivariate analysis techniques essential for AI/ML.

### Why Multivariate Statistics Matters

Multivariate statistics is crucial for understanding complex relationships in high-dimensional data. It helps us:

1. **Reduce Dimensionality**: Simplify complex datasets while preserving important information
2. **Discover Patterns**: Identify underlying structures and relationships
3. **Group Similar Objects**: Cluster observations based on multiple characteristics
4. **Feature Engineering**: Create new features from existing variables
5. **Data Visualization**: Represent high-dimensional data in lower dimensions

### Challenges of Multivariate Data

Working with multiple variables introduces unique challenges:

1. **Curse of Dimensionality**: Performance degrades as dimensions increase
2. **Correlation**: Variables may be highly correlated, creating redundancy
3. **Complexity**: Relationships become harder to visualize and understand
4. **Computational Cost**: Analysis becomes more computationally intensive
5. **Interpretation**: Results become more difficult to interpret

### Types of Multivariate Analysis

1. **Dimensionality Reduction**: PCA, Factor Analysis, MDS
2. **Clustering**: K-means, Hierarchical, DBSCAN
3. **Classification**: Discriminant Analysis, Canonical Correlation
4. **Association**: Correlation analysis, Canonical correlation
5. **Visualization**: Multidimensional scaling, Biplots

## Table of Contents
- [Principal Component Analysis](#principal-component-analysis)
- [Factor Analysis](#factor-analysis)
- [Cluster Analysis](#cluster-analysis)
- [Discriminant Analysis](#discriminant-analysis)
- [Canonical Correlation](#canonical-correlation)
- [Multidimensional Scaling](#multidimensional-scaling)
- [Practical Applications](#practical-applications)

## Setup

The examples in this chapter use Python libraries for multivariate analysis. We'll work with both theoretical concepts and practical implementations to build intuition and computational skills.

## Principal Component Analysis

Principal Component Analysis (PCA) is a fundamental technique for dimensionality reduction that transforms correlated variables into uncorrelated principal components.

### Understanding PCA

Think of PCA as finding the "best" directions to view your data. Just as you might rotate a 3D object to see it from different angles, PCA finds the directions that show the most variation in your data.

#### Intuitive Example: Student Grades

Consider student grades in multiple subjects:
- **Math, Physics, Chemistry**: Highly correlated (quantitative skills)
- **English, History**: Correlated (verbal skills)
- **Art, Music**: Correlated (creative skills)

PCA might find:
- **PC1**: Overall academic performance
- **PC2**: Quantitative vs. verbal skills
- **PC3**: Creative vs. analytical skills

### Basic PCA

#### Mathematical Foundation

PCA finds linear combinations of original variables that maximize variance:

```math
\text{PC}_i = \mathbf{w}_i^T \mathbf{X} = w_{i1}X_1 + w_{i2}X_2 + \cdots + w_{ip}X_p
```

Where:
- $`\mathbf{w}_i`$ is the i-th eigenvector (loading vector)
- $`\mathbf{X}`$ is the centered data matrix
- $`\text{PC}_i`$ is the i-th principal component

#### Optimization Problem

Find $`\mathbf{w}_1`$ that maximizes:

```math
\text{Var}(\text{PC}_1) = \mathbf{w}_1^T \mathbf{\Sigma} \mathbf{w}_1
```

Subject to $`\mathbf{w}_1^T \mathbf{w}_1 = 1`$ (unit length constraint).

#### Solution: Eigenvalue Decomposition

The solution is found by solving:

```math
\mathbf{\Sigma} \mathbf{w}_i = \lambda_i \mathbf{w}_i
```

Where:
- $`\mathbf{\Sigma}`$ is the covariance matrix
- $`\lambda_i`$ is the i-th eigenvalue
- $`\mathbf{w}_i`$ is the i-th eigenvector

#### Example: 2D Data

For centered data with covariance matrix:
```math
\mathbf{\Sigma} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}
```

**Eigenvalues**: $`\lambda_1 = 5.24`$, $`\lambda_2 = 1.76`$
**Eigenvectors**: $`\mathbf{w}_1 = (0.85, 0.53)`$, $`\mathbf{w}_2 = (-0.53, 0.85)`$

#### Variance Explained

The proportion of variance explained by each PC:

```math
\text{Proportion} = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}
```

#### Example: Variance Explanation

For the 2D example:
- **PC1**: $`\frac{5.24}{5.24 + 1.76} = 74.9\%`$ of variance
- **PC2**: $`\frac{1.76}{5.24 + 1.76} = 25.1\%`$ of variance

### PCA Algorithm

#### Step-by-Step Process

1. **Center the data**: $`\mathbf{X}_{centered} = \mathbf{X} - \bar{\mathbf{X}}`$
2. **Compute covariance matrix**: $`\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}_{centered}^T \mathbf{X}_{centered}`$
3. **Find eigenvalues and eigenvectors**: $`\mathbf{\Sigma} \mathbf{W} = \mathbf{W} \mathbf{\Lambda}`$
4. **Sort by eigenvalues**: Order PCs by decreasing variance
5. **Project data**: $`\mathbf{Z} = \mathbf{X}_{centered} \mathbf{W}`$

#### Example: Student Data

**Original variables**: Math, Physics, Chemistry, English, History
**Centered data**: Mean = 0 for each subject
**Covariance matrix**: 5×5 matrix of subject correlations
**Eigenvalues**: [2.8, 1.5, 0.4, 0.2, 0.1]
**Variance explained**: [56%, 30%, 8%, 4%, 2%]

### Choosing Number of Components

#### Methods for Component Selection

**1. Kaiser Criterion**: Keep components with eigenvalues > 1
**2. Scree Plot**: Look for "elbow" in eigenvalue plot
**3. Cumulative Variance**: Keep enough components to explain desired variance
**4. Cross-validation**: Use reconstruction error

#### Example: Component Selection

For student data:
- **Kaiser**: Keep first 2 components (eigenvalues > 1)
- **Scree plot**: Sharp drop after PC2
- **90% variance**: Keep first 3 components
- **Cross-validation**: Minimum error with 2 components

### PCA Properties

#### Mathematical Properties

1. **Orthogonality**: $`\mathbf{w}_i^T \mathbf{w}_j = 0`$ for $`i \neq j`$
2. **Uncorrelated**: $`\text{Cov}(\text{PC}_i, \text{PC}_j) = 0`$ for $`i \neq j`$
3. **Variance ordering**: $`\text{Var}(\text{PC}_1) \geq \text{Var}(\text{PC}_2) \geq \cdots`$
4. **Linear transformation**: $`\mathbf{Z} = \mathbf{X} \mathbf{W}`$

#### Geometric Interpretation

PCA finds the directions of maximum variance:
- **PC1**: Direction of greatest spread
- **PC2**: Direction of second greatest spread (orthogonal to PC1)
- **PC3**: Direction of third greatest spread (orthogonal to PC1 and PC2)

### PCA Applications

#### Dimensionality Reduction

**Example**: Image compression
- **Original**: 100×100 pixel image (10,000 dimensions)
- **PCA**: Keep top 50 components
- **Compression**: 99.5% variance explained with 0.5% of dimensions

#### Feature Engineering

**Example**: Financial data
- **Original**: 20 stock price variables
- **PCA**: 5 principal components
- **Interpretation**: Market factors (growth, value, size, etc.)

#### Visualization

**Example**: High-dimensional data
- **Original**: 50 variables
- **PCA**: 2 principal components
- **Plot**: 2D scatter plot showing data structure

## Factor Analysis

Factor Analysis identifies latent variables (factors) that explain the correlations among observed variables.

### Understanding Factor Analysis

Factor Analysis assumes that observed variables are linear combinations of unobserved factors plus error.

#### Intuitive Example: Intelligence Testing

Consider test scores in different subjects:
- **Math, Physics, Chemistry**: Load on "Quantitative Ability"
- **English, History, Literature**: Load on "Verbal Ability"
- **Art, Music, Drama**: Load on "Creative Ability"

The underlying factors are the latent abilities that explain test performance.

### Exploratory Factor Analysis

#### Mathematical Model

The factor analysis model is:

```math
\mathbf{X} = \mathbf{L} \mathbf{F} + \mathbf{\epsilon}
```

Where:
- $`\mathbf{X}`$ is the n×p observed data matrix
- $`\mathbf{L}`$ is the p×m factor loading matrix
- $`\mathbf{F}`$ is the m×n factor score matrix
- $`\mathbf{\epsilon}`$ is the n×p error matrix

#### Assumptions

1. **Factors are uncorrelated**: $`\text{Cov}(\mathbf{F}) = \mathbf{I}`$
2. **Errors are uncorrelated**: $`\text{Cov}(\mathbf{\epsilon}) = \mathbf{\Psi}`$ (diagonal)
3. **Factors and errors are uncorrelated**: $`\text{Cov}(\mathbf{F}, \mathbf{\epsilon}) = \mathbf{0}`$

#### Covariance Structure

The covariance matrix of observed variables:

```math
\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^T + \mathbf{\Psi}
```

#### Example: 3-Variable Model

For variables X₁, X₂, X₃ with 2 factors:

```math
\begin{pmatrix} X_1 \\ X_2 \\ X_3 \end{pmatrix} = \begin{pmatrix} l_{11} & l_{12} \\ l_{21} & l_{22} \\ l_{31} & l_{32} \end{pmatrix} \begin{pmatrix} F_1 \\ F_2 \end{pmatrix} + \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \epsilon_3 \end{pmatrix}
```

### Factor Loading Interpretation

#### Loading Matrix

The loading matrix $`\mathbf{L}`$ shows how each variable relates to each factor:

```math
\mathbf{L} = \begin{pmatrix} l_{11} & l_{12} & \cdots & l_{1m} \\ l_{21} & l_{22} & \cdots & l_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ l_{p1} & l_{p2} & \cdots & l_{pm} \end{pmatrix}
```

#### Example: Student Data

**Loading Matrix**:
```math
\mathbf{L} = \begin{pmatrix} 0.8 & 0.1 & 0.2 \\ 0.7 & 0.2 & 0.1 \\ 0.6 & 0.3 & 0.2 \\ 0.1 & 0.8 & 0.1 \\ 0.2 & 0.7 & 0.2 \\ 0.1 & 0.1 & 0.9 \end{pmatrix}
```

**Interpretation**:
- **Factor 1**: Quantitative subjects (Math, Physics, Chemistry)
- **Factor 2**: Verbal subjects (English, History)
- **Factor 3**: Creative subjects (Art)

### Factor Extraction Methods

#### Principal Component Method

1. **Extract eigenvalues/eigenvectors** of correlation matrix
2. **Keep m largest eigenvalues**
3. **Loadings**: $`\mathbf{L} = \mathbf{W} \sqrt{\mathbf{\Lambda}}`$

#### Maximum Likelihood Method

Maximize the likelihood function:

```math
L(\mathbf{L}, \mathbf{\Psi}) = \prod_{i=1}^{n} \frac{1}{(2\pi)^{p/2} |\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}_i - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \mathbf{\mu})\right)
```

#### Example: Extraction Results

**Correlation Matrix**:
```math
\mathbf{R} = \begin{pmatrix} 1.0 & 0.8 & 0.6 & 0.2 & 0.1 & 0.1 \\ 0.8 & 1.0 & 0.7 & 0.2 & 0.1 & 0.1 \\ 0.6 & 0.7 & 1.0 & 0.2 & 0.1 & 0.1 \\ 0.2 & 0.2 & 0.2 & 1.0 & 0.8 & 0.1 \\ 0.1 & 0.1 & 0.1 & 0.8 & 1.0 & 0.1 \\ 0.1 & 0.1 & 0.1 & 0.1 & 0.1 & 1.0 \end{pmatrix}
```

**Eigenvalues**: [2.8, 1.5, 0.4, 0.2, 0.1, 0.0]
**Factors**: 3 factors explain 78% of variance

### Factor Rotation

#### Purpose of Rotation

Rotation simplifies the factor structure by making loadings more interpretable.

#### Orthogonal Rotation (Varimax)

Maximize the variance of squared loadings:

```math
V = \sum_{j=1}^{m} \left[\frac{1}{p} \sum_{i=1}^{p} l_{ij}^4 - \left(\frac{1}{p} \sum_{i=1}^{p} l_{ij}^2\right)^2\right]
```

#### Example: Before and After Rotation

**Before Rotation**:
```math
\mathbf{L} = \begin{pmatrix} 0.7 & 0.5 & 0.3 \\ 0.6 & 0.6 & 0.2 \\ 0.5 & 0.7 & 0.1 \\ 0.4 & 0.3 & 0.8 \\ 0.3 & 0.4 & 0.7 \\ 0.2 & 0.2 & 0.9 \end{pmatrix}
```

**After Varimax Rotation**:
```math
\mathbf{L} = \begin{pmatrix} 0.9 & 0.1 & 0.1 \\ 0.8 & 0.2 & 0.1 \\ 0.7 & 0.3 & 0.1 \\ 0.1 & 0.9 & 0.1 \\ 0.2 & 0.8 & 0.1 \\ 0.1 & 0.1 & 0.9 \end{pmatrix}
```

### Model Evaluation

#### Goodness of Fit

**Chi-square test**:
```math
\chi^2 = (n-1) \ln \frac{|\mathbf{S}|}{|\mathbf{\hat{\Sigma}}|}
```

Where $`\mathbf{S}`$ is sample covariance and $`\mathbf{\hat{\Sigma}}`$ is model covariance.

#### Residual Analysis

**Residual matrix**:
```math
\mathbf{R} = \mathbf{S} - \mathbf{\hat{\Sigma}}
```

**Root Mean Square Residual (RMSR)**:
```math
\text{RMSR} = \sqrt{\frac{1}{p(p+1)/2} \sum_{i \leq j} r_{ij}^2}
```

## Cluster Analysis

Cluster Analysis groups similar observations together without predefined labels.

### Understanding Clustering

Clustering is like organizing a library by topic - books about similar subjects are placed together, even though we don't have predefined categories.

#### Intuitive Example: Customer Segmentation

Consider customer data with multiple variables:
- **Age, Income, Spending**: Demographic and behavioral variables
- **Clusters**: Young professionals, families, retirees, students
- **Purpose**: Targeted marketing strategies

### K-Means Clustering

#### Mathematical Foundation

K-means minimizes the within-cluster sum of squares:

```math
\min_{\{S_1, S_2, \ldots, S_k\}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \mathbf{\mu}_i\|^2
```

Where:
- $`S_i`$ is the i-th cluster
- $`\mathbf{\mu}_i`$ is the centroid of cluster i
- $`\|\mathbf{x} - \mathbf{\mu}_i\|^2`$ is the squared Euclidean distance

#### Algorithm

1. **Initialize**: Randomly assign k centroids
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as cluster means
4. **Repeat**: Until convergence

#### Example: 2D Clustering

**Data**: 100 points in 2D space
**K = 3**: Three clusters
**Initial centroids**: Random points
**Final centroids**: Mean of cluster members

#### Convergence

The algorithm converges because:
1. **Assignment step**: Reduces total distance
2. **Update step**: Minimizes within-cluster distance
3. **Monotonic**: Objective function never increases

### Hierarchical Clustering

#### Agglomerative Approach

Start with n singleton clusters and merge closest pairs.

#### Distance Measures

**Single Linkage**:
```math
d(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
```

**Complete Linkage**:
```math
d(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
```

**Average Linkage**:
```math
d(C_i, C_j) = \frac{1}{|C_i| |C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} \|\mathbf{x} - \mathbf{y}\|
```

**Ward's Method**:
```math
d(C_i, C_j) = \frac{|C_i| |C_j|}{|C_i| + |C_j|} \|\mathbf{\mu}_i - \mathbf{\mu}_j\|^2
```

#### Example: Dendrogram

**Data**: 6 points in 2D
**Distance matrix**: 6×6 symmetric matrix
**Merging sequence**: (1,2), (3,4), (5,6), ((1,2),(3,4)), (((1,2),(3,4)),(5,6))

### DBSCAN Clustering

#### Density-Based Approach

DBSCAN groups points based on density connectivity.

#### Parameters

- **ε (eps)**: Maximum distance for neighborhood
- **MinPts**: Minimum points for core point

#### Point Types

1. **Core Point**: At least MinPts points within ε
2. **Border Point**: Within ε of core point, but not core
3. **Noise Point**: Neither core nor border

#### Algorithm

1. **Find core points**: Points with ≥ MinPts neighbors
2. **Form clusters**: Connect core points within ε
3. **Assign border points**: To nearest core point
4. **Label noise**: Remaining points

#### Example: Parameter Selection

**Data**: 1000 points with varying density
**ε = 0.5**: Captures local density
**MinPts = 5**: Ensures meaningful clusters
**Result**: 3 clusters + noise points

### Clustering Evaluation

#### Internal Measures

**Silhouette Coefficient**:
```math
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
```

Where:
- $`a(i)`$ = average distance to points in same cluster
- $`b(i)`$ = minimum average distance to points in other clusters

**Calinski-Harabasz Index**:
```math
CH = \frac{\text{tr}(\mathbf{B}_k)}{\text{tr}(\mathbf{W}_k)} \times \frac{n-k}{k-1}
```

Where $`\mathbf{B}_k`$ and $`\mathbf{W}_k`$ are between and within cluster scatter matrices.

#### External Measures

**Adjusted Rand Index (ARI)**:
```math
ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \frac{\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}}{\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - \frac{\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}}
```

## Discriminant Analysis

Discriminant Analysis finds optimal linear combinations of variables for classification.

### Understanding Discriminant Analysis

Discriminant Analysis is like finding the best "viewing angle" to separate different groups in your data.

#### Intuitive Example: Iris Classification

Consider iris flower data:
- **Sepal length, Sepal width, Petal length, Petal width**
- **Groups**: Setosa, Versicolor, Virginica
- **Goal**: Find linear combination that best separates species

### Linear Discriminant Analysis

#### Mathematical Foundation

LDA finds projection that maximizes between-group variance relative to within-group variance:

```math
J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{B} \mathbf{w}}{\mathbf{w}^T \mathbf{W} \mathbf{w}}
```

Where:
- $`\mathbf{B}`$ = between-group scatter matrix
- $`\mathbf{W}`$ = within-group scatter matrix

#### Solution

The optimal projection is the eigenvector of $`\mathbf{W}^{-1} \mathbf{B}`$:

```math
\mathbf{W}^{-1} \mathbf{B} \mathbf{w} = \lambda \mathbf{w}
```

#### Example: 2-Group Case

For 2 groups with means $`\mathbf{\mu}_1`$, $`\mathbf{\mu}_2`$:

```math
\mathbf{w} = \mathbf{W}^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)
```

#### Classification Rule

Assign observation $`\mathbf{x}`$ to group with highest discriminant score:

```math
g_i(\mathbf{x}) = \mathbf{x}^T \mathbf{W}^{-1} \mathbf{\mu}_i - \frac{1}{2} \mathbf{\mu}_i^T \mathbf{W}^{-1} \mathbf{\mu}_i + \ln P(G_i)
```

### Quadratic Discriminant Analysis

#### Mathematical Foundation

QDA allows different covariance matrices for each group:

```math
g_i(\mathbf{x}) = -\frac{1}{2} \ln |\mathbf{\Sigma}_i| - \frac{1}{2} (\mathbf{x} - \mathbf{\mu}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x} - \mathbf{\mu}_i) + \ln P(G_i)
```

#### Example: QDA vs LDA

**LDA**: Assumes equal covariance matrices
**QDA**: Allows different covariance matrices
**Choice**: LDA if covariances are similar, QDA if they differ

## Canonical Correlation

Canonical Correlation Analysis finds relationships between two sets of variables.

### Understanding Canonical Correlation

Canonical correlation is like finding the "best" correlation between two sets of variables.

#### Intuitive Example: Academic Performance

Consider two sets of variables:
- **Set 1**: Math, Physics, Chemistry scores
- **Set 2**: English, History, Literature scores
- **Goal**: Find linear combinations that are maximally correlated

### Canonical Correlation Analysis

#### Mathematical Foundation

Find linear combinations $`U = \mathbf{a}^T \mathbf{X}`$ and $`V = \mathbf{b}^T \mathbf{Y}`$ that maximize correlation:

```math
\rho = \frac{\text{Cov}(U, V)}{\sqrt{\text{Var}(U) \text{Var}(V)}} = \frac{\mathbf{a}^T \mathbf{\Sigma}_{XY} \mathbf{b}}{\sqrt{\mathbf{a}^T \mathbf{\Sigma}_{XX} \mathbf{a} \mathbf{b}^T \mathbf{\Sigma}_{YY} \mathbf{b}}}
```

#### Solution

The canonical correlations are the eigenvalues of:

```math
\mathbf{\Sigma}_{XX}^{-1} \mathbf{\Sigma}_{XY} \mathbf{\Sigma}_{YY}^{-1} \mathbf{\Sigma}_{YX}
```

#### Example: Academic Data

**Set 1**: Math, Physics, Chemistry
**Set 2**: English, History, Literature
**Canonical correlations**: [0.85, 0.45, 0.12]
**Interpretation**: Strong relationship between quantitative and verbal abilities

## Multidimensional Scaling

Multidimensional Scaling (MDS) represents high-dimensional data in lower dimensions while preserving distances.

### Understanding MDS

MDS is like creating a map from distance information - given distances between cities, reconstruct their geographic positions.

#### Intuitive Example: Brand Perception

Consider brand similarity data:
- **Distance matrix**: How similar/different brands are perceived
- **MDS solution**: 2D map showing brand positions
- **Interpretation**: Dimensions represent brand attributes

### Classical MDS

#### Mathematical Foundation

Given distance matrix $`\mathbf{D}`$, find coordinates $`\mathbf{X}`$ that minimize:

```math
\text{Stress} = \sqrt{\frac{\sum_{i,j} (d_{ij} - \hat{d}_{ij})^2}{\sum_{i,j} d_{ij}^2}}
```

Where $`\hat{d}_{ij}`$ is the Euclidean distance between points in the MDS solution.

#### Algorithm

1. **Convert distances to similarities**: $`s_{ij} = -\frac{1}{2} d_{ij}^2`$
2. **Double-center**: $`b_{ij} = s_{ij} - \bar{s}_{i.} - \bar{s}_{.j} + \bar{s}_{..}`$
3. **Eigenvalue decomposition**: $`\mathbf{B} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T`$
4. **Coordinates**: $`\mathbf{X} = \mathbf{V} \mathbf{\Lambda}^{1/2}`$

#### Example: Brand Positioning

**Distance matrix**: 5×5 brand similarity matrix
**MDS solution**: 2D coordinates for each brand
**Interpretation**: Horizontal axis = "premium vs. budget", Vertical axis = "traditional vs. modern"

## Multivariate Normal Distribution

The **multivariate normal distribution** is the most important distribution in multivariate statistics, serving as the foundation for many statistical methods.

### Understanding the Multivariate Normal

The multivariate normal extends the univariate normal to multiple dimensions, maintaining the bell-shaped curve in higher dimensions.

#### Intuitive Example: Height and Weight

Consider height and weight data:
- **Univariate**: Each variable follows normal distribution
- **Bivariate**: Joint distribution forms elliptical contours
- **Properties**: Linear combinations are normal, conditional distributions are normal

### Mathematical Definition

A random vector $`\mathbf{X} = (X_1, X_2, \ldots, X_p)^T`$ follows a **p-dimensional multivariate normal distribution** if its joint probability density function is:

```math
f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})\right)
```

Where:
- $`\mathbf{\mu} = (\mu_1, \mu_2, \ldots, \mu_p)^T`$ is the **mean vector**
- $`\mathbf{\Sigma}`$ is the **covariance matrix** (p × p symmetric positive definite)
- $`|\mathbf{\Sigma}|`$ is the determinant of $`\mathbf{\Sigma}`$

**Notation:** $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$

#### Example: Bivariate Normal

For p = 2:
```math
f(x_1, x_2) = \frac{1}{2\pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)} \left[\frac{(x_1-\mu_1)^2}{\sigma_1^2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2} - \frac{2\rho(x_1-\mu_1)(x_2-\mu_2)}{\sigma_1\sigma_2}\right]\right)
```

### Properties of Multivariate Normal Distribution

#### 1. Marginal Distributions

If $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$, then any subset of components follows a multivariate normal distribution:
- $`X_i \sim N(\mu_i, \sigma_i^2)`$ for individual components
- $`\mathbf{X}_{(1,2)} \sim N_2(\mathbf{\mu}_{(1,2)}, \mathbf{\Sigma}_{(1,2)})`$ for any 2-dimensional subset

#### Example: Height-Weight Data

**Joint distribution**: Bivariate normal
**Marginal distributions**: Height ~ N(70, 16), Weight ~ N(160, 400)
**Correlation**: ρ = 0.7

#### 2. Linear Transformations

If $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$ and $`\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}`$, then:
```math
\mathbf{Y} \sim N_m(\mathbf{A}\mathbf{\mu} + \mathbf{b}, \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T)
```

Where $`\mathbf{A}`$ is an m × p matrix and $`\mathbf{b}`$ is an m-dimensional vector.

#### Example: Standardization

**Original**: $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$
**Standardized**: $`\mathbf{Z} = \mathbf{\Sigma}^{-1/2}(\mathbf{X} - \mathbf{\mu}) \sim N_p(\mathbf{0}, \mathbf{I})`$

#### 3. Independence

For multivariate normal, uncorrelated components are independent:
- If $`\text{Cov}(X_i, X_j) = 0`$ for all i ≠ j, then $`X_i`$ and $`X_j`$ are independent
- This is a unique property of the normal distribution

#### Example: Independent Components

**Diagonal covariance**: $`\mathbf{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_p^2)`$
**Result**: All components are independent

#### 4. Conditional Distributions

If $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$, partitioned as:

```math
\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}, \quad \mathbf{\mu} = \begin{pmatrix} \mathbf{\mu}_1 \\ \mathbf{\mu}_2 \end{pmatrix}, \quad \mathbf{\Sigma} = \begin{pmatrix} \mathbf{\Sigma}_{11} & \mathbf{\Sigma}_{12} \\ \mathbf{\Sigma}_{21} & \mathbf{\Sigma}_{22} \end{pmatrix}
```

Then the conditional distribution is:

```math
\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim N_{p_1}(\mathbf{\mu}_{1|2}, \mathbf{\Sigma}_{1|2})
```

Where:

```math
\mathbf{\mu}_{1|2} = \mathbf{\mu}_1 + \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \mathbf{\mu}_2)
```

```math
\mathbf{\Sigma}_{1|2} = \mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}
```

#### Example: Height Given Weight

**Joint distribution**: Height and weight ~ N₂(μ, Σ)
**Conditional**: Height|Weight = 180 ~ N(μ₁|₂, σ₁|₂²)
**Interpretation**: Height distribution for people weighing 180 lbs

#### 5. Maximum Likelihood Estimation

For a sample $`\mathbf{X}_1, \mathbf{X}_2, \ldots, \mathbf{X}_n \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$, the MLEs are:

```math
\hat{\mathbf{\mu}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{X}_i = \bar{\mathbf{X}}
```

```math
\hat{\mathbf{\Sigma}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{X}_i - \bar{\mathbf{X}})(\mathbf{X}_i - \bar{\mathbf{X}})^T
```

#### Example: Sample Estimation

**Sample size**: n = 100
**Sample mean**: $`\bar{\mathbf{X}} = (70, 160)^T`$
**Sample covariance**: $`\mathbf{S} = \begin{pmatrix} 16 & 56 \\ 56 & 400 \end{pmatrix}`$

#### 6. Mahalanobis Distance

The squared Mahalanobis distance is:

```math
D^2(\mathbf{x}) = (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})
```

This follows a $`\chi^2(p)`$ distribution under the null hypothesis.

#### Example: Outlier Detection

**Data point**: $`\mathbf{x} = (75, 200)^T`$
**Mahalanobis distance**: $`D^2 = 4.2`$
**Critical value**: $`\chi^2_{0.95}(2) = 5.99`$
**Conclusion**: Not an outlier (D² < critical value)

### Characteristic Function

The characteristic function of $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$ is:

```math
\phi_{\mathbf{X}}(\mathbf{t}) = E[e^{i\mathbf{t}^T\mathbf{X}}] = \exp\left(i\mathbf{t}^T\mathbf{\mu} - \frac{1}{2}\mathbf{t}^T\mathbf{\Sigma}\mathbf{t}\right)
```

### Moment Generating Function

The moment generating function is:

```math
M_{\mathbf{X}}(\mathbf{t}) = E[e^{\mathbf{t}^T\mathbf{X}}] = \exp\left(\mathbf{t}^T\mathbf{\mu} + \frac{1}{2}\mathbf{t}^T\mathbf{\Sigma}\mathbf{t}\right)
```

### Central Limit Theorem (Multivariate)

If $`\mathbf{X}_1, \mathbf{X}_2, \ldots, \mathbf{X}_n`$ are independent random vectors with $`E[\mathbf{X}_i] = \mathbf{\mu}`$ and $`\text{Cov}(\mathbf{X}_i) = \mathbf{\Sigma}`$, then:

```math
\sqrt{n}(\bar{\mathbf{X}} - \mathbf{\mu}) \xrightarrow{d} N_p(\mathbf{0}, \mathbf{\Sigma})
```

#### Example: Sample Mean Distribution

**Population**: $`\mathbf{X} \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$
**Sample size**: n = 50
**Sample mean**: $`\bar{\mathbf{X}} \sim N_p(\mathbf{\mu}, \frac{1}{50}\mathbf{\Sigma})`$

## Practical Applications

### Customer Segmentation

#### Business Context

Customer segmentation helps businesses understand their customer base and develop targeted marketing strategies.

#### Data Description

**Variables**: Age, Income, Spending, Frequency, Recency
**Sample**: 10,000 customers
**Goal**: Identify distinct customer segments

#### Analysis Process

1. **Data preprocessing**: Standardize variables
2. **Dimensionality reduction**: PCA to 3 components
3. **Clustering**: K-means with 5 clusters
4. **Profiling**: Analyze cluster characteristics

#### Example Results

**Cluster 1**: Young professionals (high income, moderate spending)
**Cluster 2**: Families (moderate income, high spending)
**Cluster 3**: Students (low income, low spending)
**Cluster 4**: Retirees (moderate income, low spending)
**Cluster 5**: High-net-worth (very high income, very high spending)

#### Business Impact

- **Targeted marketing**: Different strategies for each segment
- **Product development**: Features for specific segments
- **Pricing strategies**: Segment-specific pricing
- **Customer retention**: Segment-specific retention programs

### Feature Engineering

#### Machine Learning Context

Feature engineering creates new variables that improve model performance.

#### PCA for Feature Engineering

**Original features**: 50 variables
**PCA components**: 10 principal components
**Variance explained**: 85% of total variance
**New features**: Linear combinations of original variables

#### Example: Image Processing

**Original**: 100×100 pixel images (10,000 features)
**PCA**: 50 principal components
**Compression**: 99% variance with 0.5% of features
**Application**: Face recognition, image classification

### Anomaly Detection

#### Multivariate Approach

Anomaly detection identifies unusual observations in high-dimensional data.

#### Mahalanobis Distance Method

1. **Estimate parameters**: $`\hat{\mathbf{\mu}}`$, $`\hat{\mathbf{\Sigma}}`$
2. **Calculate distances**: $`D^2(\mathbf{x}_i)`$ for each observation
3. **Identify outliers**: Points with $`D^2 > \chi^2_{0.95}(p)`$

#### Example: Credit Card Fraud

**Variables**: Transaction amount, time, location, merchant type
**Normal transactions**: Follow multivariate normal distribution
**Fraudulent transactions**: Deviate from normal pattern
**Detection**: High Mahalanobis distance indicates fraud

## Practice Problems

### Problem 1: Dimensionality Reduction

**Objective**: Implement PCA with automatic component selection and visualization.

**Tasks**:
1. Create PCA implementation with eigenvalue decomposition
2. Add automatic component selection methods
3. Implement scree plot and cumulative variance visualization
4. Add biplot functionality for component interpretation
5. Include reconstruction error analysis

**Example Implementation**:
```python
def pca_analysis(data, n_components=None, method='eigenvalue'):
    """
    Perform PCA with automatic component selection.
    
    Returns: components, loadings, variance_explained, reconstruction_error
    """
    # Implementation here
```

### Problem 2: Clustering Evaluation

**Objective**: Create comprehensive clustering evaluation frameworks.

**Tasks**:
1. Implement internal evaluation metrics (silhouette, Calinski-Harabasz)
2. Add external evaluation metrics (ARI, AMI, homogeneity)
3. Create clustering visualization tools
4. Include stability analysis methods
5. Add automated cluster number selection

### Problem 3: Feature Engineering

**Objective**: Build automated feature engineering pipelines using multivariate techniques.

**Tasks**:
1. Implement PCA-based feature creation
2. Add factor analysis for latent variable extraction
3. Create canonical correlation features
4. Include clustering-based features
5. Add feature selection methods

### Problem 4: Anomaly Detection

**Objective**: Develop multivariate anomaly detection methods.

**Tasks**:
1. Implement Mahalanobis distance-based detection
2. Add clustering-based outlier detection
3. Create PCA-based reconstruction error methods
4. Include robust estimation techniques
5. Add visualization tools for anomaly detection

### Problem 5: Real-World Multivariate Analysis

**Objective**: Apply multivariate techniques to real datasets.

**Tasks**:
1. Choose a dataset (customer, financial, biological)
2. Perform comprehensive exploratory analysis
3. Apply multiple multivariate techniques
4. Compare and evaluate different approaches
5. Write comprehensive analysis report

## Further Reading

### Books
- **"Multivariate Data Analysis"** by Hair, Black, Babin, and Anderson
- **"Applied Multivariate Statistical Analysis"** by Johnson and Wichern
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
- **"Multivariate Analysis"** by Mardia, Kent, and Bibby

### Online Resources
- **StatQuest**: YouTube channel with clear multivariate explanations
- **Khan Academy**: Linear algebra and statistics courses
- **Coursera**: Machine Learning course by Andrew Ng
- **edX**: Statistical Learning course

### Advanced Topics
- **Independent Component Analysis**: Finding independent sources
- **Non-negative Matrix Factorization**: Constrained dimensionality reduction
- **Manifold Learning**: Non-linear dimensionality reduction
- **Spectral Clustering**: Graph-based clustering methods
- **Bayesian Multivariate Methods**: Probabilistic approaches

## Key Takeaways

### Fundamental Concepts
- **PCA** reduces dimensionality while preserving variance
- **Factor Analysis** identifies latent variables underlying observed features
- **Clustering** groups similar observations without predefined labels
- **Discriminant Analysis** finds optimal projections for classification
- **Canonical Correlation** analyzes relationships between two sets of variables
- **Multivariate techniques** are essential for high-dimensional data analysis
- **Feature engineering** benefits greatly from multivariate statistical methods
- **Real-world applications** include customer segmentation, feature selection, and data exploration

### Mathematical Tools
- **Eigenvalue decomposition** provides solutions for many multivariate problems
- **Distance measures** are fundamental to clustering and classification
- **Covariance matrices** capture relationships between variables
- **Linear transformations** preserve multivariate normal properties
- **Optimization methods** find optimal solutions for complex problems

### Applications
- **Customer segmentation** uses clustering to identify market segments
- **Feature engineering** creates new variables for machine learning
- **Anomaly detection** identifies unusual observations in high-dimensional data
- **Data visualization** represents complex relationships in lower dimensions
- **Quality control** monitors multivariate processes

### Best Practices
- **Always scale variables** before multivariate analysis
- **Check assumptions** for each method (normality, linearity, etc.)
- **Use multiple evaluation metrics** for comprehensive assessment
- **Validate results** with domain knowledge and cross-validation
- **Interpret results** in context of the specific application

### Next Steps
In the following chapters, we'll build on multivariate foundations to explore:
- **Bayesian Statistics**: Probabilistic approaches to inference
- **Analysis of Variance**: Comparing means across multiple groups
- **Nonparametric Methods**: When assumptions are violated
- **Advanced Topics**: Specialized methods for complex data structures

Remember that multivariate statistics is not just about applying mathematical techniques—it's about understanding complex relationships, discovering patterns, and extracting meaningful insights from high-dimensional data. The methods and concepts covered in this chapter provide the foundation for sophisticated data analysis and evidence-based decision making. 