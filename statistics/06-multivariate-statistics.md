# Multivariate Statistics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![FactorAnalyzer](https://img.shields.io/badge/FactorAnalyzer-0.4+-blue.svg)](https://factor-analyzer.readthedocs.io/)

Multivariate statistics deals with the analysis of data with multiple variables. This chapter covers dimensionality reduction, clustering, and multivariate analysis techniques essential for AI/ML.

## Table of Contents
- [Principal Component Analysis](#principal-component-analysis)
- [Factor Analysis](#factor-analysis)
- [Cluster Analysis](#cluster-analysis)
- [Discriminant Analysis](#discriminant-analysis)
- [Canonical Correlation](#canonical-correlation)
- [Multidimensional Scaling](#multidimensional-scaling)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import make_blobs, make_classification
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Principal Component Analysis

### Basic PCA

```python
def generate_multivariate_data(n_samples=300, n_features=10):
    """Generate multivariate data with known structure"""
    # Generate correlated features
    np.random.seed(42)
    
    # Create correlation matrix
    corr_matrix = np.eye(n_features)
    corr_matrix[0:3, 0:3] = 0.8  # First 3 features highly correlated
    corr_matrix[3:6, 3:6] = 0.7  # Next 3 features moderately correlated
    corr_matrix[6:8, 6:8] = 0.6  # Features 6-7 correlated
    
    # Generate data
    data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=corr_matrix,
        size=n_samples
    )
    
    # Add some noise features
    noise_features = np.random.normal(0, 1, (n_samples, 2))
    data = np.column_stack([data, noise_features])
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(data.shape[1])]
    
    return pd.DataFrame(data, columns=feature_names)

# Generate data
df_pca = generate_multivariate_data()
print("Multivariate Data Overview")
print(f"Shape: {df_pca.shape}")
print(f"Features: {list(df_pca.columns)}")

# Correlation matrix
correlation_matrix = df_pca.corr()
print(f"\nCorrelation Matrix Shape: {correlation_matrix.shape}")

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Perform PCA
def perform_pca_analysis(data, n_components=None):
    """Perform comprehensive PCA analysis"""
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    if n_components is None:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    
    pca_result = pca.fit_transform(data_scaled)
    
    return pca, pca_result, data_scaled

pca, pca_result, data_scaled = perform_pca_analysis(df_pca)

print("PCA Results")
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Visualize PCA results
plt.figure(figsize=(15, 10))

# Scree plot
plt.subplot(2, 3, 1)
n_components = len(pca.explained_variance_ratio_)
plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.xticks(range(1, n_components + 1))

# Cumulative explained variance
plt.subplot(2, 3, 2)
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), cumulative_var, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

# First two principal components
plt.subplot(2, 3, 3)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, color='lightgreen')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
plt.title('First Two Principal Components')

# Component loadings
plt.subplot(2, 3, 4)
loadings = pca.components_.T
plt.bar(range(len(df_pca.columns)), loadings[:, 0], alpha=0.7, color='orange')
plt.xlabel('Features')
plt.ylabel('Loading')
plt.title('PC1 Loadings')
plt.xticks(range(len(df_pca.columns)), df_pca.columns, rotation=45)

plt.subplot(2, 3, 5)
plt.bar(range(len(df_pca.columns)), loadings[:, 1], alpha=0.7, color='purple')
plt.xlabel('Features')
plt.ylabel('Loading')
plt.title('PC2 Loadings')
plt.xticks(range(len(df_pca.columns)), df_pca.columns, rotation=45)

# Biplot (simplified)
plt.subplot(2, 3, 6)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, color='lightgreen')
for i, feature in enumerate(df_pca.columns):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
              head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, fontsize=8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot (PC1 vs PC2)')

plt.tight_layout()
plt.show()

# Determine optimal number of components
def find_optimal_components(data, max_components=None):
    """Find optimal number of components using various criteria"""
    
    if max_components is None:
        max_components = min(data.shape)
    
    # Perform PCA with maximum components
    pca_full = PCA()
    pca_full.fit(StandardScaler().fit_transform(data))
    
    # Calculate metrics
    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find components for different thresholds
    n_80 = np.argmax(cumulative_var >= 0.8) + 1
    n_90 = np.argmax(cumulative_var >= 0.9) + 1
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    # Kaiser criterion (eigenvalues > 1)
    eigenvalues = pca_full.explained_variance_
    n_kaiser = np.sum(eigenvalues > 1)
    
    # Elbow method (simplified)
    second_derivative = np.diff(np.diff(explained_var))
    n_elbow = np.argmax(second_derivative) + 2
    
    return {
        'n_80': n_80,
        'n_90': n_90,
        'n_95': n_95,
        'n_kaiser': n_kaiser,
        'n_elbow': n_elbow,
        'explained_var': explained_var,
        'cumulative_var': cumulative_var
    }

optimal_components = find_optimal_components(df_pca)

print("Optimal Number of Components")
print(f"80% variance: {optimal_components['n_80']} components")
print(f"90% variance: {optimal_components['n_90']} components")
print(f"95% variance: {optimal_components['n_95']} components")
print(f"Kaiser criterion: {optimal_components['n_kaiser']} components")
print(f"Elbow method: {optimal_components['n_elbow']} components")
```

## Factor Analysis

### Exploratory Factor Analysis

```python
def perform_factor_analysis(data, n_factors=None):
    """Perform factor analysis"""
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform factor analysis
    if n_factors is None:
        n_factors = data.shape[1] // 2  # Default to half the features
    
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_result = fa.fit_transform(data_scaled)
    
    return fa, fa_result, data_scaled

fa_model, fa_result, data_scaled_fa = perform_factor_analysis(df_pca, n_factors=4)

print("Factor Analysis Results")
print(f"Number of factors: {fa_model.n_components_}")
print(f"Factor loadings shape: {fa_model.components_.shape}")

# Visualize factor analysis results
plt.figure(figsize=(15, 10))

# Factor loadings heatmap
plt.subplot(2, 3, 1)
loadings_df = pd.DataFrame(fa_model.components_.T, 
                          index=df_pca.columns,
                          columns=[f'Factor_{i+1}' for i in range(fa_model.n_components_)])
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Factor Loadings')

# Factor scores
plt.subplot(2, 3, 2)
plt.scatter(fa_result[:, 0], fa_result[:, 1], alpha=0.7, color='skyblue')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Scores (Factor 1 vs Factor 2)')

# Factor loadings bar plot
plt.subplot(2, 3, 3)
loadings_abs = np.abs(fa_model.components_.T)
plt.bar(range(len(df_pca.columns)), loadings_abs[:, 0], alpha=0.7, color='lightgreen')
plt.xlabel('Features')
plt.ylabel('Absolute Loading')
plt.title('Factor 1 Loadings')
plt.xticks(range(len(df_pca.columns)), df_pca.columns, rotation=45)

plt.subplot(2, 3, 4)
plt.bar(range(len(df_pca.columns)), loadings_abs[:, 1], alpha=0.7, color='orange')
plt.xlabel('Features')
plt.ylabel('Absolute Loading')
plt.title('Factor 2 Loadings')
plt.xticks(range(len(df_pca.columns)), df_pca.columns, rotation=45)

# Factor correlation
plt.subplot(2, 3, 5)
factor_corr = np.corrcoef(fa_result.T)
sns.heatmap(factor_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Factor Correlation Matrix')

# Reconstruction error
reconstructed = fa_model.inverse_transform(fa_result)
reconstruction_error = np.mean((data_scaled_fa - reconstructed) ** 2)
plt.subplot(2, 3, 6)
plt.text(0.5, 0.5, f'Reconstruction Error:\n{reconstruction_error:.4f}', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.title('Reconstruction Error')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Reconstruction Error: {reconstruction_error:.4f}")
```

## Cluster Analysis

### K-Means Clustering

```python
def generate_clustering_data(n_samples=300):
    """Generate data for clustering analysis"""
    # Generate clusters
    centers = [[2, 2], [8, 3], [3, 6], [7, 8]]
    cluster_std = [1.5, 1.2, 1.8, 1.3]
    
    X, y_true = make_blobs(n_samples=n_samples, centers=centers, 
                          cluster_std=cluster_std, random_state=42)
    
    # Add some noise
    noise = np.random.normal(0, 0.5, (n_samples, 2))
    X += noise
    
    return X, y_true

X_cluster, y_true = generate_clustering_data()

print("Clustering Data Overview")
print(f"Data shape: {X_cluster.shape}")
print(f"True clusters: {len(np.unique(y_true))}")

# K-Means clustering
def perform_kmeans_analysis(X, max_k=10):
    """Perform K-means clustering with different k values"""
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    
    return inertias, silhouette_scores, calinski_scores

inertias, silhouette_scores, calinski_scores = perform_kmeans_analysis(X_cluster)

# Visualize clustering results
plt.figure(figsize=(15, 10))

# Elbow plot
plt.subplot(2, 3, 1)
k_values = range(2, len(inertias) + 2)
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Plot')
plt.grid(True, alpha=0.3)

# Silhouette score
plt.subplot(2, 3, 2)
plt.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True, alpha=0.3)

# Calinski-Harabasz score
plt.subplot(2, 3, 3)
plt.plot(k_values, calinski_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Score')
plt.grid(True, alpha=0.3)

# Optimal K-means clustering
optimal_k = k_values[np.argmax(silhouette_scores)]
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_optimal.fit(X_cluster)

plt.subplot(2, 3, 4)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=kmeans_optimal.labels_, 
           cmap='viridis', alpha=0.7)
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.legend()

# True clusters
plt.subplot(2, 3, 5)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=y_true, 
           cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('True Clusters')

# Comparison
plt.subplot(2, 3, 6)
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(y_true, kmeans_optimal.labels_)
plt.text(0.5, 0.5, f'Adjusted Rand Index:\n{ari_score:.3f}', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
plt.title('Clustering Quality')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Optimal number of clusters: {optimal_k}")
print(f"Adjusted Rand Index: {ari_score:.3f}")
```

### Hierarchical Clustering

```python
def perform_hierarchical_clustering(X, n_clusters=4):
    """Perform hierarchical clustering"""
    
    # Perform clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    return hierarchical, hierarchical_labels

hierarchical_model, hierarchical_labels = perform_hierarchical_clustering(X_cluster)

# Visualize hierarchical clustering
plt.figure(figsize=(15, 5))

# Dendrogram (simplified visualization)
from scipy.cluster.hierarchy import dendrogram, linkage

plt.subplot(1, 3, 1)
linkage_matrix = linkage(X_cluster, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')

# Hierarchical clustering results
plt.subplot(1, 3, 2)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=hierarchical_labels, 
           cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering')

# Comparison with K-means
plt.subplot(1, 3, 3)
from sklearn.metrics import adjusted_rand_score
hierarchical_ari = adjusted_rand_score(y_true, hierarchical_labels)
kmeans_ari = adjusted_rand_score(y_true, kmeans_optimal.labels_)

plt.bar(['K-Means', 'Hierarchical'], [kmeans_ari, hierarchical_ari], 
        alpha=0.7, color=['skyblue', 'lightgreen'])
plt.ylabel('Adjusted Rand Index')
plt.title('Clustering Method Comparison')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"Hierarchical Clustering ARI: {hierarchical_ari:.3f}")
print(f"K-Means ARI: {kmeans_ari:.3f}")
```

### DBSCAN Clustering

```python
def perform_dbscan_clustering(X):
    """Perform DBSCAN clustering"""
    
    # Perform DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    return dbscan, dbscan_labels

dbscan_model, dbscan_labels = perform_dbscan_clustering(X_cluster)

# Visualize DBSCAN results
plt.figure(figsize=(15, 5))

# DBSCAN clustering
plt.subplot(1, 3, 1)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=dbscan_labels, 
           cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'DBSCAN Clustering (eps=1.0, min_samples=5)')

# Number of clusters found
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

plt.subplot(1, 3, 2)
cluster_counts = [list(dbscan_labels).count(i) for i in set(dbscan_labels)]
plt.bar(range(len(cluster_counts)), cluster_counts, alpha=0.7, color='orange')
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.title(f'Cluster Sizes (Noise: {n_noise})')

# Comparison with other methods
plt.subplot(1, 3, 3)
dbscan_ari = adjusted_rand_score(y_true, dbscan_labels)
methods = ['K-Means', 'Hierarchical', 'DBSCAN']
ari_scores = [kmeans_ari, hierarchical_ari, dbscan_ari]
colors = ['skyblue', 'lightgreen', 'orange']

plt.bar(methods, ari_scores, alpha=0.7, color=colors)
plt.ylabel('Adjusted Rand Index')
plt.title('Clustering Method Comparison')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"DBSCAN Results:")
print(f"Number of clusters: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise}")
print(f"DBSCAN ARI: {dbscan_ari:.3f}")
```

## Discriminant Analysis

### Linear Discriminant Analysis

```python
def generate_classification_data(n_samples=300):
    """Generate data for discriminant analysis"""
    
    # Generate classification data
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=6,
                             n_redundant=2, n_classes=3, n_clusters_per_class=1,
                             random_state=42)
    
    return X, y

X_disc, y_disc = generate_classification_data()

print("Classification Data Overview")
print(f"Data shape: {X_disc.shape}")
print(f"Number of classes: {len(np.unique(y_disc))}")

# Perform LDA
def perform_lda_analysis(X, y):
    """Perform Linear Discriminant Analysis"""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform LDA
    lda = LinearDiscriminantAnalysis()
    lda_result = lda.fit_transform(X_scaled, y)
    
    return lda, lda_result, X_scaled

lda_model, lda_result, X_scaled_disc = perform_lda_analysis(X_disc, y_disc)

# Visualize LDA results
plt.figure(figsize=(15, 10))

# LDA projection
plt.subplot(2, 3, 1)
plt.scatter(lda_result[:, 0], lda_result[:, 1], c=y_disc, cmap='viridis', alpha=0.7)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection')
plt.colorbar()

# Explained variance ratio
plt.subplot(2, 3, 2)
explained_var = lda_model.explained_variance_ratio_
plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='skyblue')
plt.xlabel('Linear Discriminant')
plt.ylabel('Explained Variance Ratio')
plt.title('LDA Explained Variance')

# Feature importance
plt.subplot(2, 3, 3)
feature_importance = np.abs(lda_model.coef_[0])
plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7, color='lightgreen')
plt.xlabel('Features')
plt.ylabel('Coefficient Magnitude')
plt.title('LDA Feature Importance')

# Classification accuracy
from sklearn.model_selection import cross_val_score
lda_scores = cross_val_score(lda_model, X_scaled_disc, y_disc, cv=5)

plt.subplot(2, 3, 4)
plt.bar(range(1, len(lda_scores) + 1), lda_scores, alpha=0.7, color='orange')
plt.axhline(lda_scores.mean(), color='red', linestyle='--', label=f'Mean: {lda_scores.mean():.3f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('LDA Cross-Validation Scores')
plt.legend()

# Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred_lda = lda_model.predict(X_scaled_disc)
cm = confusion_matrix(y_disc, y_pred_lda)

plt.subplot(2, 3, 5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LDA Confusion Matrix')

# Comparison with PCA
pca_disc = PCA(n_components=2)
pca_result_disc = pca_disc.fit_transform(X_scaled_disc)

plt.subplot(2, 3, 6)
plt.scatter(pca_result_disc[:, 0], pca_result_disc[:, 1], c=y_disc, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"LDA Results:")
print(f"Explained variance ratio: {explained_var}")
print(f"Cross-validation accuracy: {lda_scores.mean():.3f} (+/- {lda_scores.std() * 2:.3f})")
```

## Canonical Correlation

### Canonical Correlation Analysis

```python
def generate_canonical_data(n_samples=200):
    """Generate data for canonical correlation analysis"""
    
    # Generate two sets of variables
    np.random.seed(42)
    
    # First set of variables
    X1 = np.random.multivariate_normal([0, 0, 0], [[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]], n_samples)
    
    # Second set of variables (correlated with first set)
    X2 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n_samples)
    
    # Add correlation between sets
    X2[:, 0] = 0.6 * X1[:, 0] + 0.4 * X2[:, 0]
    X2[:, 1] = 0.5 * X1[:, 1] + 0.5 * X2[:, 1]
    
    return X1, X2

X1_canon, X2_canon = generate_canonical_data()

print("Canonical Correlation Data Overview")
print(f"X1 shape: {X1_canon.shape}")
print(f"X2 shape: {X2_canon.shape}")

# Perform canonical correlation analysis
def perform_canonical_correlation(X1, X2):
    """Perform canonical correlation analysis"""
    
    from sklearn.cross_decomposition import CCA
    
    # Standardize data
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    X2_scaled = scaler2.fit_transform(X2)
    
    # Perform CCA
    cca = CCA(n_components=min(X1.shape[1], X2.shape[1]))
    cca_result = cca.fit_transform(X1_scaled, X2_scaled)
    
    return cca, cca_result, X1_scaled, X2_scaled

cca_model, cca_result, X1_scaled_canon, X2_scaled_canon = perform_canonical_correlation(X1_canon, X2_canon)

# Visualize canonical correlation results
plt.figure(figsize=(15, 10))

# Canonical variates
plt.subplot(2, 3, 1)
plt.scatter(cca_result[0][:, 0], cca_result[1][:, 0], alpha=0.7, color='skyblue')
plt.xlabel('Canonical Variate 1 (X1)')
plt.ylabel('Canonical Variate 1 (X2)')
plt.title('First Canonical Variates')

plt.subplot(2, 3, 2)
plt.scatter(cca_result[0][:, 1], cca_result[1][:, 1], alpha=0.7, color='lightgreen')
plt.xlabel('Canonical Variate 2 (X1)')
plt.ylabel('Canonical Variate 2 (X2)')
plt.title('Second Canonical Variates')

# Canonical correlations
canonical_correlations = np.corrcoef(cca_result[0].T, cca_result[1].T)[:2, 2:]
diagonal_correlations = np.diag(canonical_correlations)

plt.subplot(2, 3, 3)
plt.bar(range(1, len(diagonal_correlations) + 1), diagonal_correlations, alpha=0.7, color='orange')
plt.xlabel('Canonical Pair')
plt.ylabel('Canonical Correlation')
plt.title('Canonical Correlations')

# Loadings for first canonical variate
plt.subplot(2, 3, 4)
loadings1 = cca_model.x_weights_[:, 0]
plt.bar(range(len(loadings1)), loadings1, alpha=0.7, color='purple')
plt.xlabel('X1 Variables')
plt.ylabel('Loading')
plt.title('X1 Loadings (First Canonical Variate)')

plt.subplot(2, 3, 5)
loadings2 = cca_model.y_weights_[:, 0]
plt.bar(range(len(loadings2)), loadings2, alpha=0.7, color='brown')
plt.xlabel('X2 Variables')
plt.ylabel('Loading')
plt.title('X2 Loadings (First Canonical Variate)')

# Correlation matrix between original variables
plt.subplot(2, 3, 6)
combined_data = np.column_stack([X1_canon, X2_canon])
corr_matrix = np.corrcoef(combined_data.T)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix (X1 + X2)')

plt.tight_layout()
plt.show()

print(f"Canonical Correlations: {diagonal_correlations}")
```

## Practical Applications

### Customer Segmentation

```python
def generate_customer_data(n_customers=500):
    """Generate customer data for segmentation"""
    
    np.random.seed(42)
    
    # Generate customer features
    age = np.random.normal(35, 10, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    spending = np.random.normal(2000, 800, n_customers)
    frequency = np.random.poisson(5, n_customers)
    satisfaction = np.random.uniform(1, 10, n_customers)
    
    # Create customer segments
    segment_1 = (age < 30) & (income > 60000)
    segment_2 = (age > 45) & (spending > 2500)
    segment_3 = (frequency > 7) & (satisfaction > 8)
    
    segments = np.zeros(n_customers)
    segments[segment_1] = 1
    segments[segment_2] = 2
    segments[segment_3] = 3
    
    # Create DataFrame
    customer_data = pd.DataFrame({
        'age': age,
        'income': income,
        'spending': spending,
        'frequency': frequency,
        'satisfaction': satisfaction,
        'segment': segments
    })
    
    return customer_data

customer_df = generate_customer_data()

print("Customer Segmentation Analysis")
print(f"Data shape: {customer_df.shape}")
print(f"Number of segments: {len(customer_df['segment'].unique())}")

# Perform customer segmentation
features = ['age', 'income', 'spending', 'frequency', 'satisfaction']
X_customers = customer_df[features].values
y_customers = customer_df['segment'].values

# Standardize features
scaler_customers = StandardScaler()
X_customers_scaled = scaler_customers.fit_transform(X_customers)

# Perform clustering
kmeans_customers = KMeans(n_clusters=4, random_state=42)
kmeans_customers.fit(X_customers_scaled)
customer_clusters = kmeans_customers.labels_

# Visualize customer segmentation
plt.figure(figsize=(15, 10))

# Age vs Income
plt.subplot(2, 3, 1)
plt.scatter(customer_df['age'], customer_df['income'], c=customer_clusters, 
           cmap='viridis', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income (Clusters)')

# Spending vs Frequency
plt.subplot(2, 3, 2)
plt.scatter(customer_df['spending'], customer_df['frequency'], c=customer_clusters, 
           cmap='viridis', alpha=0.7)
plt.xlabel('Spending')
plt.ylabel('Frequency')
plt.title('Spending vs Frequency (Clusters)')

# PCA projection
pca_customers = PCA(n_components=2)
pca_customers_result = pca_customers.fit_transform(X_customers_scaled)

plt.subplot(2, 3, 3)
plt.scatter(pca_customers_result[:, 0], pca_customers_result[:, 1], c=customer_clusters, 
           cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection (Clusters)')

# Cluster characteristics
plt.subplot(2, 3, 4)
cluster_means = []
for i in range(4):
    cluster_mask = customer_clusters == i
    cluster_means.append(customer_df[features][cluster_mask].mean())

cluster_means_df = pd.DataFrame(cluster_means, columns=features)
sns.heatmap(cluster_means_df.T, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.1f')
plt.title('Cluster Characteristics')

# Feature importance in clustering
plt.subplot(2, 3, 5)
feature_importance = np.abs(kmeans_customers.cluster_centers_).mean(axis=0)
plt.bar(features, feature_importance, alpha=0.7, color='orange')
plt.xlabel('Features')
plt.ylabel('Average Cluster Center Magnitude')
plt.title('Feature Importance in Clustering')
plt.xticks(rotation=45)

# Cluster sizes
plt.subplot(2, 3, 6)
cluster_sizes = [np.sum(customer_clusters == i) for i in range(4)]
plt.bar(range(4), cluster_sizes, alpha=0.7, color='purple')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Sizes')

plt.tight_layout()
plt.show()

# Customer segment profiles
print("\nCustomer Segment Profiles:")
for i in range(4):
    cluster_mask = customer_clusters == i
    cluster_data = customer_df[cluster_mask]
    print(f"\nSegment {i+1} (n={len(cluster_data)}):")
    print(f"  Average age: {cluster_data['age'].mean():.1f}")
    print(f"  Average income: ${cluster_data['income'].mean():,.0f}")
    print(f"  Average spending: ${cluster_data['spending'].mean():,.0f}")
    print(f"  Average frequency: {cluster_data['frequency'].mean():.1f}")
    print(f"  Average satisfaction: {cluster_data['satisfaction'].mean():.1f}")
```

## Practice Problems

1. **Dimensionality Reduction**: Implement PCA with automatic component selection and visualization.

2. **Clustering Evaluation**: Create comprehensive clustering evaluation frameworks with multiple metrics.

3. **Feature Engineering**: Build automated feature engineering pipelines using multivariate techniques.

4. **Anomaly Detection**: Develop multivariate anomaly detection methods using clustering and dimensionality reduction.

## Further Reading

- "Multivariate Data Analysis" by Hair, Black, Babin, and Anderson
- "Applied Multivariate Statistical Analysis" by Johnson and Wichern
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

## Key Takeaways

- **PCA** reduces dimensionality while preserving variance
- **Factor Analysis** identifies latent variables underlying observed features
- **Clustering** groups similar observations without predefined labels
- **Discriminant Analysis** finds optimal projections for classification
- **Canonical Correlation** analyzes relationships between two sets of variables
- **Multivariate techniques** are essential for high-dimensional data analysis
- **Feature engineering** benefits greatly from multivariate statistical methods
- **Real-world applications** include customer segmentation, feature selection, and data exploration

In the next chapter, we'll explore Bayesian statistics, including Bayesian inference, MCMC methods, and their applications in machine learning. 