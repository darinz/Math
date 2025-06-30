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

## Multivariate Normal Distribution

The **multivariate normal distribution** is the most important distribution in multivariate statistics, serving as the foundation for many statistical methods.

### Mathematical Definition

A random vector **X** = (X₁, X₂, ..., Xₚ)ᵀ follows a **p-dimensional multivariate normal distribution** if its joint probability density function is:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})\right)$$

Where:
- **μ** = (μ₁, μ₂, ..., μₚ)ᵀ is the **mean vector**
- **Σ** is the **covariance matrix** (p × p symmetric positive definite)
- |**Σ**| is the determinant of **Σ**

**Notation:** **X** ~ Nₚ(**μ**, **Σ**)

### Properties of Multivariate Normal Distribution

**1. Marginal Distributions:**
If **X** ~ Nₚ(**μ**, **Σ**), then any subset of components follows a multivariate normal distribution:
- Xᵢ ~ N(μᵢ, σᵢ²) for individual components
- **X**₍₁,₂₎ ~ N₂(**μ**₍₁,₂₎, **Σ**₍₁,₂₎) for any 2-dimensional subset

**2. Linear Transformations:**
If **X** ~ Nₚ(**μ**, **Σ**) and **Y** = **AX** + **b**, then:
**Y** ~ Nₘ(**Aμ** + **b**, **AΣA**ᵀ)

Where **A** is an m × p matrix and **b** is an m-dimensional vector.

**3. Independence:**
For multivariate normal, uncorrelated components are independent:
- If Cov(Xᵢ, Xⱼ) = 0 for all i ≠ j, then Xᵢ and Xⱼ are independent
- This is a unique property of the normal distribution

**4. Conditional Distributions:**
If **X** ~ Nₚ(**μ**, **Σ**), partitioned as:
$$\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}, \quad \mathbf{\mu} = \begin{pmatrix} \mathbf{\mu}_1 \\ \mathbf{\mu}_2 \end{pmatrix}, \quad \mathbf{\Sigma} = \begin{pmatrix} \mathbf{\Sigma}_{11} & \mathbf{\Sigma}_{12} \\ \mathbf{\Sigma}_{21} & \mathbf{\Sigma}_{22} \end{pmatrix}$$

Then the conditional distribution is:
$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim N_{p_1}(\mathbf{\mu}_{1|2}, \mathbf{\Sigma}_{1|2})$$

Where:
$$\mathbf{\mu}_{1|2} = \mathbf{\mu}_1 + \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \mathbf{\mu}_2)$$
$$\mathbf{\Sigma}_{1|2} = \mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}$$

**5. Maximum Likelihood Estimation:**
For a sample **X**₁, **X**₂, ..., **X**ₙ ~ Nₚ(**μ**, **Σ**), the MLEs are:
$$\hat{\mathbf{\mu}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{X}_i = \bar{\mathbf{X}}$$
$$\hat{\mathbf{\Sigma}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{X}_i - \bar{\mathbf{X}})(\mathbf{X}_i - \bar{\mathbf{X}})^T$$

**6. Mahalanobis Distance:**
The squared Mahalanobis distance is:
$$D^2(\mathbf{x}) = (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})$$

This follows a χ²(p) distribution under the null hypothesis.

### Characteristic Function

The characteristic function of **X** ~ Nₚ(**μ**, **Σ**) is:
$$\phi_{\mathbf{X}}(\mathbf{t}) = E[e^{i\mathbf{t}^T\mathbf{X}}] = \exp\left(i\mathbf{t}^T\mathbf{\mu} - \frac{1}{2}\mathbf{t}^T\mathbf{\Sigma}\mathbf{t}\right)$$

### Moment Generating Function

The moment generating function is:
$$M_{\mathbf{X}}(\mathbf{t}) = E[e^{\mathbf{t}^T\mathbf{X}}] = \exp\left(\mathbf{t}^T\mathbf{\mu} + \frac{1}{2}\mathbf{t}^T\mathbf{\Sigma}\mathbf{t}\right)$$

### Central Limit Theorem (Multivariate)

If **X**₁, **X**₂, ..., **X**ₙ are independent random vectors with E[**X**ᵢ] = **μ** and Cov(**X**ᵢ) = **Σ**, then:
$$\sqrt{n}(\bar{\mathbf{X}} - \mathbf{\mu}) \xrightarrow{d} N_p(\mathbf{0}, \mathbf{\Sigma})$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, chi2
from scipy.linalg import cholesky, inv
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def multivariate_normal_pdf(x, mu, sigma):
    """
    Calculate multivariate normal PDF
    
    Mathematical implementation:
    f(x) = (2π)^(-p/2) |Σ|^(-1/2) exp(-0.5(x-μ)ᵀΣ⁻¹(x-μ))
    
    Parameters:
    x: array, point to evaluate
    mu: array, mean vector
    sigma: array, covariance matrix
    
    Returns:
    float: PDF value
    """
    p = len(mu)
    
    # Check dimensions
    if len(x) != p:
        raise ValueError("Dimensions of x and mu must match")
    
    if sigma.shape != (p, p):
        raise ValueError("Sigma must be p×p matrix")
    
    # Calculate determinant
    det_sigma = np.linalg.det(sigma)
    if det_sigma <= 0:
        raise ValueError("Sigma must be positive definite")
    
    # Calculate inverse
    sigma_inv = np.linalg.inv(sigma)
    
    # Calculate quadratic form
    diff = x - mu
    quadratic_form = diff.T @ sigma_inv @ diff
    
    # Calculate PDF
    pdf = (1 / ((2 * np.pi) ** (p/2) * np.sqrt(det_sigma))) * np.exp(-0.5 * quadratic_form)
    
    return pdf

def generate_multivariate_normal(n, mu, sigma, method='cholesky'):
    """
    Generate samples from multivariate normal distribution
    
    Mathematical implementation:
    X = μ + LZ where L is Cholesky factor of Σ and Z ~ N(0,I)
    
    Parameters:
    n: int, number of samples
    mu: array, mean vector
    sigma: array, covariance matrix
    method: str, generation method
    
    Returns:
    array: n×p matrix of samples
    """
    p = len(mu)
    
    if method == 'cholesky':
        # Cholesky decomposition method
        L = cholesky(sigma, lower=True)
        Z = np.random.normal(0, 1, (n, p))
        X = mu + Z @ L.T
        return X
    
    elif method == 'eigenvalue':
        # Eigenvalue decomposition method
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        D_sqrt = np.diag(np.sqrt(eigenvals))
        Z = np.random.normal(0, 1, (n, p))
        X = mu + Z @ D_sqrt @ eigenvecs.T
        return X
    
    else:
        raise ValueError(f"Unknown method: {method}")

def mahalanobis_distance(x, mu, sigma):
    """
    Calculate Mahalanobis distance
    
    Mathematical implementation:
    D²(x) = (x-μ)ᵀΣ⁻¹(x-μ)
    
    Parameters:
    x: array, point
    mu: array, mean vector
    sigma: array, covariance matrix
    
    Returns:
    float: Mahalanobis distance
    """
    sigma_inv = np.linalg.inv(sigma)
    diff = x - mu
    distance_squared = diff.T @ sigma_inv @ diff
    return np.sqrt(distance_squared)

def conditional_multivariate_normal(x2, mu, sigma, partition_idx):
    """
    Calculate conditional distribution parameters
    
    Mathematical implementation:
    μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ - μ₂)
    Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁
    
    Parameters:
    x2: array, conditioning values
    mu: array, mean vector
    sigma: array, covariance matrix
    partition_idx: list, indices for partition
    
    Returns:
    tuple: (conditional_mean, conditional_covariance)
    """
    p = len(mu)
    
    # Create partition indices
    idx1 = [i for i in range(p) if i not in partition_idx]
    idx2 = partition_idx
    
    # Extract submatrices
    mu1 = mu[idx1]
    mu2 = mu[idx2]
    sigma11 = sigma[np.ix_(idx1, idx1)]
    sigma12 = sigma[np.ix_(idx1, idx2)]
    sigma21 = sigma[np.ix_(idx2, idx1)]
    sigma22 = sigma[np.ix_(idx2, idx2)]
    
    # Calculate conditional parameters
    sigma22_inv = np.linalg.inv(sigma22)
    conditional_mean = mu1 + sigma12 @ sigma22_inv @ (x2 - mu2)
    conditional_covariance = sigma11 - sigma12 @ sigma22_inv @ sigma21
    
    return conditional_mean, conditional_covariance

def multivariate_normal_mle(X):
    """
    Calculate maximum likelihood estimates for multivariate normal
    
    Mathematical implementation:
    μ̂ = (1/n)ΣᵢXᵢ
    Σ̂ = (1/n)Σᵢ(Xᵢ-μ̂)(Xᵢ-μ̂)ᵀ
    
    Parameters:
    X: array, n×p data matrix
    
    Returns:
    tuple: (mu_hat, sigma_hat)
    """
    n, p = X.shape
    
    # MLE for mean
    mu_hat = np.mean(X, axis=0)
    
    # MLE for covariance
    centered_X = X - mu_hat
    sigma_hat = (centered_X.T @ centered_X) / n
    
    return mu_hat, sigma_hat

def multivariate_normal_log_likelihood(X, mu, sigma):
    """
    Calculate log-likelihood for multivariate normal
    
    Mathematical implementation:
    log L = -(n/2)log|Σ| - (1/2)Σᵢ(xᵢ-μ)ᵀΣ⁻¹(xᵢ-μ) - (np/2)log(2π)
    
    Parameters:
    X: array, n×p data matrix
    mu: array, mean vector
    sigma: array, covariance matrix
    
    Returns:
    float: log-likelihood
    """
    n, p = X.shape
    
    # Calculate determinant and inverse
    det_sigma = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    
    # Calculate log-likelihood
    log_det = np.log(det_sigma)
    
    # Calculate quadratic form for all observations
    centered_X = X - mu
    quadratic_terms = np.sum(centered_X @ sigma_inv * centered_X, axis=1)
    total_quadratic = np.sum(quadratic_terms)
    
    log_likelihood = -(n/2) * log_det - (1/2) * total_quadratic - (n*p/2) * np.log(2*np.pi)
    
    return log_likelihood

def multivariate_normal_entropy(mu, sigma):
    """
    Calculate entropy of multivariate normal distribution
    
    Mathematical implementation:
    H(X) = (p/2)log(2πe) + (1/2)log|Σ|
    
    Parameters:
    mu: array, mean vector
    sigma: array, covariance matrix
    
    Returns:
    float: entropy
    """
    p = len(mu)
    det_sigma = np.linalg.det(sigma)
    
    entropy = (p/2) * np.log(2*np.pi*np.e) + (1/2) * np.log(det_sigma)
    return entropy

def multivariate_normal_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between two multivariate normal distributions
    
    Mathematical implementation:
    KL(N₁||N₂) = (1/2)[tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - p - log(|Σ₁|/|Σ₂|)]
    
    Parameters:
    mu1, sigma1: parameters of first distribution
    mu2, sigma2: parameters of second distribution
    
    Returns:
    float: KL divergence
    """
    p = len(mu1)
    
    sigma2_inv = np.linalg.inv(sigma2)
    diff = mu2 - mu1
    
    term1 = np.trace(sigma2_inv @ sigma1)
    term2 = diff.T @ sigma2_inv @ diff
    term3 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    
    kl_div = (1/2) * (term1 + term2 - p - term3)
    return kl_div

# Example: 2-dimensional multivariate normal
np.random.seed(42)

# Parameters
mu = np.array([2.0, 3.0])
sigma = np.array([[4.0, 1.5],
                  [1.5, 2.0]])

print("Multivariate Normal Distribution Analysis")
print("=" * 50)

# Generate samples
n_samples = 1000
X = generate_multivariate_normal(n_samples, mu, sigma, method='cholesky')

print(f"Generated {n_samples} samples from N₂(μ, Σ)")
print(f"μ = {mu}")
print(f"Σ = \n{sigma}")

# Calculate MLE
mu_hat, sigma_hat = multivariate_normal_mle(X)
print(f"\nMaximum Likelihood Estimates:")
print(f"μ̂ = {mu_hat}")
print(f"Σ̂ = \n{sigma_hat}")

# Calculate log-likelihood
log_likelihood = multivariate_normal_log_likelihood(X, mu_hat, sigma_hat)
print(f"Log-likelihood: {log_likelihood:.4f}")

# Calculate entropy
entropy = multivariate_normal_entropy(mu, sigma)
print(f"Entropy: {entropy:.4f}")

# Calculate Mahalanobis distances
mahal_distances = np.array([mahalanobis_distance(x, mu, sigma) for x in X])
print(f"Mean Mahalanobis distance: {np.mean(mahal_distances):.4f}")
print(f"Mahalanobis distance variance: {np.var(mahal_distances):.4f}")

# Test theoretical properties
print(f"\nTheoretical Properties Verification:")

# 1. Marginal distributions
print(f"1. Marginal distributions:")
print(f"   X₁ ~ N({mu[0]}, {sigma[0,0]})")
print(f"   X₂ ~ N({mu[1]}, {sigma[1,1]})")
print(f"   Sample means: {np.mean(X, axis=0)}")
print(f"   Sample variances: {np.var(X, axis=0)}")

# 2. Correlation
theoretical_corr = sigma[0,1] / np.sqrt(sigma[0,0] * sigma[1,1])
sample_corr = np.corrcoef(X.T)[0,1]
print(f"2. Correlation:")
print(f"   Theoretical: {theoretical_corr:.4f}")
print(f"   Sample: {sample_corr:.4f}")

# 3. Mahalanobis distance distribution
# Should follow χ²(2) distribution
chi2_quantiles = chi2.ppf(np.linspace(0.01, 0.99, 100), df=2)
mahal_squared = mahal_distances**2
mahal_quantiles = np.percentile(mahal_squared, np.linspace(1, 99, 100))

print(f"3. Mahalanobis distance squared ~ χ²(2):")
print(f"   Sample mean: {np.mean(mahal_squared):.4f} (theoretical: 2.0)")
print(f"   Sample variance: {np.var(mahal_squared):.4f} (theoretical: 4.0)")

# Conditional distribution example
print(f"\nConditional Distribution Example:")
x2_condition = 4.0  # Condition on X₂ = 4.0
cond_mu, cond_sigma = conditional_multivariate_normal(x2_condition, mu, sigma, [1])
print(f"X₁ | X₂ = {x2_condition} ~ N({cond_mu[0]:.4f}, {cond_sigma[0,0]:.4f})")

# Generate conditional samples
conditional_samples = np.random.normal(cond_mu[0], np.sqrt(cond_sigma[0,0]), 100)
print(f"Conditional sample mean: {np.mean(conditional_samples):.4f}")
print(f"Conditional sample variance: {np.var(conditional_samples):.4f}")

# Visualize multivariate normal
plt.figure(figsize=(15, 12))

# 1. Scatter plot with contours
plt.subplot(2, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)

# Create contour plot
x1_range = np.linspace(mu[0] - 3*np.sqrt(sigma[0,0]), mu[0] + 3*np.sqrt(sigma[0,0]), 100)
x2_range = np.linspace(mu[1] - 3*np.sqrt(sigma[1,1]), mu[1] + 3*np.sqrt(sigma[1,1]), 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i,j], X2[i,j]])
        Z[i,j] = multivariate_normal_pdf(x, mu, sigma)

plt.contour(X1, X2, Z, levels=10, colors='red', alpha=0.7)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('Multivariate Normal Samples with Contours')
plt.grid(True, alpha=0.3)

# 2. Marginal distributions
plt.subplot(2, 3, 2)
plt.hist(X[:, 0], bins=30, alpha=0.7, density=True, label='X₁')
plt.hist(X[:, 1], bins=30, alpha=0.7, density=True, label='X₂')

# Theoretical marginal PDFs
x1_theoretical = np.linspace(mu[0] - 4*np.sqrt(sigma[0,0]), mu[0] + 4*np.sqrt(sigma[0,0]), 100)
x2_theoretical = np.linspace(mu[1] - 4*np.sqrt(sigma[1,1]), mu[1] + 4*np.sqrt(sigma[1,1]), 100)

pdf1 = (1/np.sqrt(2*np.pi*sigma[0,0])) * np.exp(-0.5*(x1_theoretical - mu[0])**2/sigma[0,0])
pdf2 = (1/np.sqrt(2*np.pi*sigma[1,1])) * np.exp(-0.5*(x2_theoretical - mu[1])**2/sigma[1,1])

plt.plot(x1_theoretical, pdf1, 'r-', linewidth=2, label='X₁ PDF')
plt.plot(x2_theoretical, pdf2, 'g-', linewidth=2, label='X₂ PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Marginal Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Mahalanobis distance distribution
plt.subplot(2, 3, 3)
plt.hist(mahal_squared, bins=30, alpha=0.7, density=True, label='Sample')

# Theoretical χ²(2) distribution
chi2_x = np.linspace(0, np.max(mahal_squared), 100)
chi2_pdf = chi2.pdf(chi2_x, df=2)
plt.plot(chi2_x, chi2_pdf, 'r-', linewidth=2, label='χ²(2) PDF')

plt.xlabel('Mahalanobis Distance²')
plt.ylabel('Density')
plt.title('Mahalanobis Distance Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Q-Q plot for Mahalanobis distances
plt.subplot(2, 3, 4)
from scipy.stats import probplot
probplot(mahal_squared, dist=chi2, sparams=(2,), plot=plt)
plt.title('Q-Q Plot: Mahalanobis Distance² vs χ²(2)')
plt.grid(True, alpha=0.3)

# 5. Correlation structure
plt.subplot(2, 3, 5)
correlation_matrix = np.corrcoef(X.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            xticklabels=['X₁', 'X₂'], yticklabels=['X₁', 'X₂'])
plt.title('Sample Correlation Matrix')

# 6. 3D surface plot of PDF
ax = plt.subplot(2, 3, 6, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_zlabel('PDF')
ax.set_title('Multivariate Normal PDF Surface')

plt.tight_layout()
plt.show()

# Demonstrate linear transformation properties
print(f"\nLinear Transformation Properties:")

# Define transformation: Y = AX + b
A = np.array([[1.5, 0.5],
              [0.5, 1.0]])
b = np.array([1.0, 2.0])

# Theoretical transformed parameters
mu_transformed = A @ mu + b
sigma_transformed = A @ sigma @ A.T

print(f"Transformation: Y = AX + b")
print(f"A = \n{A}")
print(f"b = {b}")
print(f"Theoretical μ_Y = {mu_transformed}")
print(f"Theoretical Σ_Y = \n{sigma_transformed}")

# Apply transformation to samples
Y = X @ A.T + b

# Calculate sample parameters
mu_Y_hat, sigma_Y_hat = multivariate_normal_mle(Y)
print(f"Sample μ_Y = {mu_Y_hat}")
print(f"Sample Σ_Y = \n{sigma_Y_hat}")

# Verify transformation property
transformation_error_mu = np.linalg.norm(mu_transformed - mu_Y_hat)
transformation_error_sigma = np.linalg.norm(sigma_transformed - sigma_Y_hat)
print(f"Transformation property verification:")
print(f"  μ error: {transformation_error_mu:.6f}")
print(f"  Σ error: {transformation_error_sigma:.6f}")

# KL divergence example
print(f"\nKL Divergence Example:")
# Create two different multivariate normal distributions
mu2 = np.array([3.0, 4.0])
sigma2 = np.array([[3.0, 1.0],
                   [1.0, 2.5]])

kl_div_12 = multivariate_normal_kl_divergence(mu, sigma, mu2, sigma2)
kl_div_21 = multivariate_normal_kl_divergence(mu2, sigma2, mu, sigma)

print(f"KL(N₁||N₂) = {kl_div_12:.4f}")
print(f"KL(N₂||N₁) = {kl_div_21:.4f}")
print(f"KL divergence is asymmetric: {abs(kl_div_12 - kl_div_21) > 1e-10}")

# Entropy comparison
entropy1 = multivariate_normal_entropy(mu, sigma)
entropy2 = multivariate_normal_entropy(mu2, sigma2)

print(f"Entropy N₁: {entropy1:.4f}")
print(f"Entropy N₂: {entropy2:.4f}")
print(f"Entropy difference: {abs(entropy1 - entropy2):.4f}")

# Visualize transformation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20, label='Original X')
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.6, s=20, label='Transformed Y')
plt.xlabel('X₁ / Y₁')
plt.ylabel('X₂ / Y₂')
plt.title('Linear Transformation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Show how transformation affects the distribution
plt.hist(X[:, 0], bins=30, alpha=0.7, density=True, label='X₁')
plt.hist(Y[:, 0], bins=30, alpha=0.7, density=True, label='Y₁')

# Theoretical PDFs
x1_pdf = (1/np.sqrt(2*np.pi*sigma[0,0])) * np.exp(-0.5*(x1_theoretical - mu[0])**2/sigma[0,0])
y1_theoretical = np.linspace(mu_transformed[0] - 4*np.sqrt(sigma_transformed[0,0]), 
                            mu_transformed[0] + 4*np.sqrt(sigma_transformed[0,0]), 100)
y1_pdf = (1/np.sqrt(2*np.pi*sigma_transformed[0,0])) * np.exp(-0.5*(y1_theoretical - mu_transformed[0])**2/sigma_transformed[0,0])

plt.plot(x1_theoretical, x1_pdf, 'r-', linewidth=2, label='X₁ PDF')
plt.plot(y1_theoretical, y1_pdf, 'g-', linewidth=2, label='Y₁ PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Marginal Distribution Transformation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate independence property
print(f"\nIndependence Property:")
# Create uncorrelated multivariate normal
sigma_uncorr = np.array([[4.0, 0.0],
                         [0.0, 2.0]])

X_uncorr = generate_multivariate_normal(n_samples, mu, sigma_uncorr)
corr_uncorr = np.corrcoef(X_uncorr.T)[0,1]

print(f"Uncorrelated case:")
print(f"  Theoretical correlation: 0.0")
print(f"  Sample correlation: {corr_uncorr:.6f}")
print(f"  Components independent: {abs(corr_uncorr) < 0.1}")

# Test independence by checking if joint PDF equals product of marginals
def test_independence(X, mu, sigma):
    """Test independence by comparing joint and marginal PDFs"""
    n_test = 100
    test_points = generate_multivariate_normal(n_test, mu, sigma)
    
    joint_pdf_values = np.array([multivariate_normal_pdf(x, mu, sigma) for x in test_points])
    
    # Calculate product of marginal PDFs
    marginal_pdf_values = np.ones(n_test)
    for i, x in enumerate(test_points):
        pdf1 = (1/np.sqrt(2*np.pi*sigma[0,0])) * np.exp(-0.5*(x[0] - mu[0])**2/sigma[0,0])
        pdf2 = (1/np.sqrt(2*np.pi*sigma[1,1])) * np.exp(-0.5*(x[1] - mu[1])**2/sigma[1,1])
        marginal_pdf_values[i] = pdf1 * pdf2
    
    # Calculate correlation between joint and marginal PDFs
    independence_corr = np.corrcoef(joint_pdf_values, marginal_pdf_values)[0,1]
    return independence_corr

independence_corr_uncorr = test_independence(X_uncorr, mu, sigma_uncorr)
independence_corr_corr = test_independence(X, mu, sigma)

print(f"Independence test (correlation between joint and marginal PDFs):")
print(f"  Uncorrelated case: {independence_corr_uncorr:.6f} (should be ≈ 1.0)")
print(f"  Correlated case: {independence_corr_corr:.6f} (should be < 1.0)")
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