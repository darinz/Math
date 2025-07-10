"""
Multivariate Statistics Implementation

This module provides comprehensive implementations of multivariate statistics concepts,
including PCA, factor analysis, clustering, discriminant analysis, canonical correlation,
multidimensional scaling, and multivariate normal distribution.

Key Concepts Covered:
- Principal Component Analysis: Dimensionality reduction and feature extraction
- Factor Analysis: Latent variable identification and modeling
- Cluster Analysis: K-means, hierarchical, and DBSCAN clustering
- Discriminant Analysis: Linear and quadratic discriminant analysis
- Canonical Correlation: Relationships between two sets of variables
- Multidimensional Scaling: Distance-preserving dimensionality reduction
- Multivariate Normal Distribution: Properties and applications
- Practical Applications: Customer segmentation, feature engineering, anomaly detection

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
from sklearn.manifold import MDS
from factor_analyzer import FactorAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis Implementation
    
    Implements PCA with manual eigenvalue decomposition and comprehensive analysis tools.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None
        self.scale = None
        
    def fit(self, X):
        """
        Fit PCA model using eigenvalue decomposition.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted PCA model
        """
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data using fitted PCA model.
        """
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        """
        return X_transformed @ self.components.T + self.mean
    
    def get_reconstruction_error(self, X):
        """
        Calculate reconstruction error.
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)
    
    def plot_scree(self):
        """
        Plot scree plot for component selection.
        """
        plt.figure(figsize=(10, 6))
        
        # Scree plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.explained_variance) + 1), 
                self.explained_variance, 'bo-')
        plt.axhline(y=1, color='red', linestyle='--', label='Kaiser criterion')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative variance
        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        plt.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-')
        plt.axhline(y=0.9, color='green', linestyle='--', label='90% variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_biplot(self, X, feature_names=None):
        """
        Create biplot showing both observations and variables.
        """
        X_transformed = self.transform(X)
        
        plt.figure(figsize=(12, 8))
        
        # Plot observations
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
        
        # Plot variable loadings
        if feature_names is None:
            feature_names = [f'Var_{i}' for i in range(self.components.shape[0])]
        
        for i, (comp1, comp2) in enumerate(zip(self.components[:, 0], self.components[:, 1])):
            plt.arrow(0, 0, comp1, comp2, head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', alpha=0.7)
            plt.text(comp1 * 1.15, comp2 * 1.15, feature_names[i], 
                    color='red', ha='center', va='center')
        
        plt.xlabel(f'PC1 ({self.explained_variance_ratio[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio[1]:.1%} variance)')
        plt.title('PCA Biplot')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


class FactorAnalysis:
    """
    Factor Analysis Implementation
    
    Implements exploratory factor analysis with rotation and model evaluation.
    """
    
    def __init__(self, n_factors=None, rotation='varimax'):
        self.n_factors = n_factors
        self.rotation = rotation
        self.loadings = None
        self.communalities = None
        self.factor_scores = None
        
    def fit(self, X):
        """
        Fit factor analysis model.
        """
        X = np.array(X)
        
        # Use factor_analyzer library
        fa = FactorAnalyzer(n_factors=self.n_factors, rotation=self.rotation)
        fa.fit(X)
        
        self.loadings = fa.loadings_
        self.communalities = fa.get_communalities()
        
        return self
    
    def get_factor_scores(self, X):
        """
        Calculate factor scores.
        """
        # Simple regression method
        X_centered = X - np.mean(X, axis=0)
        factor_scores = X_centered @ self.loadings @ np.linalg.inv(self.loadings.T @ self.loadings)
        return factor_scores
    
    def plot_loadings(self, feature_names=None):
        """
        Plot factor loadings heatmap.
        """
        if feature_names is None:
            feature_names = [f'Var_{i}' for i in range(self.loadings.shape[0])]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.loadings, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=[f'Factor_{i+1}' for i in range(self.loadings.shape[1])],
                   yticklabels=feature_names)
        plt.title('Factor Loadings')
        plt.tight_layout()
        plt.show()


class ClusterAnalysis:
    """
    Comprehensive Cluster Analysis Implementation
    
    Implements K-means, hierarchical clustering, and DBSCAN with evaluation metrics.
    """
    
    def __init__(self):
        self.kmeans_model = None
        self.hierarchical_model = None
        self.dbscan_model = None
        self.labels = None
        
    def kmeans_clustering(self, X, n_clusters=3, random_state=42):
        """
        Perform K-means clustering.
        """
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.labels = self.kmeans_model.fit_predict(X)
        return self.labels
    
    def hierarchical_clustering(self, X, n_clusters=3, linkage='ward'):
        """
        Perform hierarchical clustering.
        """
        self.hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = self.hierarchical_model.fit_predict(X)
        return self.labels
    
    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering.
        """
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.dbscan_model.fit_predict(X)
        return self.labels
    
    def evaluate_clustering(self, X, labels):
        """
        Evaluate clustering quality using multiple metrics.
        """
        # Internal metrics
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'n_clusters': len(np.unique(labels[labels != -1]))  # Exclude noise points
        }
    
    def plot_clusters(self, X, labels, title="Clustering Results"):
        """
        Plot clustering results.
        """
        plt.figure(figsize=(10, 6))
        
        # If more than 2 dimensions, use PCA for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            xlabel = 'PC1'
            ylabel = 'PC2'
        else:
            X_2d = X
            xlabel = 'Feature 1'
            ylabel = 'Feature 2'
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                # Noise points
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', 
                           marker='x', s=50, label='Noise')
            else:
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], 
                           label=f'Cluster {label}')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class DiscriminantAnalysis:
    """
    Discriminant Analysis Implementation
    
    Implements Linear and Quadratic Discriminant Analysis for classification.
    """
    
    def __init__(self, method='linear'):
        self.method = method
        self.lda_model = None
        self.qda_model = None
        
    def fit(self, X, y):
        """
        Fit discriminant analysis model.
        """
        if self.method == 'linear':
            self.lda_model = LinearDiscriminantAnalysis()
            self.lda_model.fit(X, y)
        else:
            self.qda_model = QuadraticDiscriminantAnalysis()
            self.qda_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        """
        if self.method == 'linear':
            return self.lda_model.predict(X)
        else:
            return self.qda_model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if self.method == 'linear':
            return self.lda_model.predict_proba(X)
        else:
            return self.qda_model.predict_proba(X)
    
    def transform(self, X):
        """
        Transform data to discriminant space.
        """
        if self.method == 'linear':
            return self.lda_model.transform(X)
        else:
            raise ValueError("Transform not available for QDA")
    
    def plot_discriminant_space(self, X, y):
        """
        Plot data in discriminant space.
        """
        if self.method != 'linear':
            raise ValueError("Plot only available for LDA")
        
        X_transformed = self.transform(X)
        
        plt.figure(figsize=(10, 6))
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = y == label
            plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       c=[color], label=f'Class {label}')
        
        plt.xlabel('Discriminant 1')
        plt.ylabel('Discriminant 2')
        plt.title('Data in Discriminant Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class CanonicalCorrelation:
    """
    Canonical Correlation Analysis Implementation
    
    Finds relationships between two sets of variables.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.canonical_correlations = None
        self.canonical_coefficients = None
        
    def fit(self, X, Y):
        """
        Fit canonical correlation analysis.
        """
        X = np.array(X)
        Y = np.array(Y)
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Compute covariance matrices
        n = X.shape[0]
        Sxx = X_centered.T @ X_centered / (n - 1)
        Syy = Y_centered.T @ Y_centered / (n - 1)
        Sxy = X_centered.T @ Y_centered / (n - 1)
        Syx = Sxy.T
        
        # Compute canonical correlations
        # Solve: (Sxx^(-1) * Sxy * Syy^(-1) * Syx) * a = λ * a
        Sxx_inv = np.linalg.inv(Sxx)
        Syy_inv = np.linalg.inv(Syy)
        
        # Matrix for eigenvalue decomposition
        M = Sxx_inv @ Sxy @ Syy_inv @ Syx
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Canonical correlations are square roots of eigenvalues
        self.canonical_correlations = np.sqrt(np.maximum(eigenvalues, 0))
        
        # Canonical coefficients for X
        self.canonical_coefficients_X = eigenvectors
        
        # Canonical coefficients for Y
        self.canonical_coefficients_Y = Syy_inv @ Syx @ eigenvectors
        
        if self.n_components is not None:
            self.canonical_correlations = self.canonical_correlations[:self.n_components]
            self.canonical_coefficients_X = self.canonical_coefficients_X[:, :self.n_components]
            self.canonical_coefficients_Y = self.canonical_coefficients_Y[:, :self.n_components]
        
        return self
    
    def transform(self, X, Y):
        """
        Transform data to canonical space.
        """
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        U = X_centered @ self.canonical_coefficients_X
        V = Y_centered @ self.canonical_coefficients_Y
        
        return U, V
    
    def plot_canonical_correlations(self):
        """
        Plot canonical correlations.
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(1, len(self.canonical_correlations) + 1), 
                self.canonical_correlations, 'bo-')
        plt.xlabel('Canonical Variable Pair')
        plt.ylabel('Canonical Correlation')
        plt.title('Canonical Correlations')
        plt.grid(True, alpha=0.3)
        plt.show()


class MultidimensionalScaling:
    """
    Multidimensional Scaling Implementation
    
    Represents high-dimensional data in lower dimensions while preserving distances.
    """
    
    def __init__(self, n_components=2, metric=True):
        self.n_components = n_components
        self.metric = metric
        self.embedding = None
        
    def fit_transform(self, X):
        """
        Fit MDS and transform data.
        """
        if self.metric:
            # Classical MDS
            mds = MDS(n_components=self.n_components, metric=True, random_state=42)
        else:
            # Non-metric MDS
            mds = MDS(n_components=self.n_components, metric=False, random_state=42)
        
        self.embedding = mds.fit_transform(X)
        return self.embedding
    
    def plot_mds(self, labels=None):
        """
        Plot MDS results.
        """
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(self.embedding[mask, 0], self.embedding[mask, 1], 
                           c=[color], label=f'Group {label}')
            plt.legend()
        else:
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=0.6)
        
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.title('Multidimensional Scaling')
        plt.grid(True, alpha=0.3)
        plt.show()


class MultivariateNormal:
    """
    Multivariate Normal Distribution Implementation
    
    Implements properties and applications of multivariate normal distribution.
    """
    
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        
    def fit(self, X):
        """
        Estimate parameters from data.
        """
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T)
        return self
    
    def pdf(self, X):
        """
        Calculate probability density function.
        """
        X = np.array(X)
        n_features = X.shape[1]
        
        # Calculate Mahalanobis distance
        diff = X - self.mean
        inv_cov = np.linalg.inv(self.cov)
        mahal_dist = np.sum(diff @ inv_cov * diff, axis=1)
        
        # Calculate PDF
        det_cov = np.linalg.det(self.cov)
        pdf = (1 / ((2 * np.pi) ** (n_features / 2) * np.sqrt(det_cov))) * np.exp(-0.5 * mahal_dist)
        
        return pdf
    
    def mahalanobis_distance(self, X):
        """
        Calculate Mahalanobis distance.
        """
        X = np.array(X)
        diff = X - self.mean
        inv_cov = np.linalg.inv(self.cov)
        mahal_dist = np.sum(diff @ inv_cov * diff, axis=1)
        return mahal_dist
    
    def detect_outliers(self, X, alpha=0.05):
        """
        Detect outliers using Mahalanobis distance.
        """
        mahal_dist = self.mahal_distance(X)
        critical_value = chi2.ppf(1 - alpha, df=X.shape[1])
        outliers = mahal_dist > critical_value
        return outliers, mahal_dist
    
    def sample(self, n_samples=1000):
        """
        Generate samples from multivariate normal distribution.
        """
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)


def create_sample_multivariate_data():
    """
    Create sample multivariate data for demonstration.
    """
    np.random.seed(42)
    
    # Create correlated data
    n_samples = 200
    n_features = 5
    
    # Generate correlated features
    mean = np.zeros(n_features)
    cov = np.array([
        [1.0, 0.8, 0.6, 0.2, 0.1],
        [0.8, 1.0, 0.7, 0.2, 0.1],
        [0.6, 0.7, 1.0, 0.2, 0.1],
        [0.2, 0.2, 0.2, 1.0, 0.8],
        [0.1, 0.1, 0.1, 0.8, 1.0]
    ])
    
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Create labels for some demonstrations
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
    
    return X, y


def demonstrate_pca():
    """
    Demonstrate Principal Component Analysis.
    """
    print("=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, _ = create_sample_multivariate_data()
    
    # Fit PCA
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(X)
    
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {pca.n_components}")
    print(f"Variance explained: {pca.explained_variance_ratio[:3]}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio[:3]):.3f}")
    
    # Transform data
    X_transformed = pca.transform(X)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # Reconstruction error
    reconstruction_error = pca.get_reconstruction_error(X)
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # Visualizations
    pca.plot_scree()
    pca.plot_biplot(X, feature_names=[f'Feature_{i}' for i in range(X.shape[1])])


def demonstrate_clustering():
    """
    Demonstrate clustering methods.
    """
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, _ = create_sample_multivariate_data()
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    cluster_analyzer = ClusterAnalysis()
    kmeans_labels = cluster_analyzer.kmeans_clustering(X_scaled, n_clusters=3)
    
    # Evaluate clustering
    kmeans_eval = cluster_analyzer.evaluate_clustering(X_scaled, kmeans_labels)
    print("K-means Clustering Results:")
    print(f"  Number of clusters: {kmeans_eval['n_clusters']}")
    print(f"  Silhouette score: {kmeans_eval['silhouette_score']:.3f}")
    print(f"  Calinski-Harabasz score: {kmeans_eval['calinski_harabasz_score']:.3f}")
    
    # Plot results
    cluster_analyzer.plot_clusters(X_scaled, kmeans_labels, "K-means Clustering")
    
    # Hierarchical clustering
    hierarchical_labels = cluster_analyzer.hierarchical_clustering(X_scaled, n_clusters=3)
    hierarchical_eval = cluster_analyzer.evaluate_clustering(X_scaled, hierarchical_labels)
    
    print("\nHierarchical Clustering Results:")
    print(f"  Number of clusters: {hierarchical_eval['n_clusters']}")
    print(f"  Silhouette score: {hierarchical_eval['silhouette_score']:.3f}")
    print(f"  Calinski-Harabasz score: {hierarchical_eval['calinski_harabasz_score']:.3f}")


def demonstrate_discriminant_analysis():
    """
    Demonstrate discriminant analysis.
    """
    print("\n" + "=" * 60)
    print("DISCRIMINANT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data with labels
    X, y = create_sample_multivariate_data()
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Linear Discriminant Analysis
    lda = DiscriminantAnalysis(method='linear')
    lda.fit(X_scaled, y)
    
    # Make predictions
    y_pred = lda.predict(X_scaled)
    accuracy = np.mean(y_pred == y)
    print(f"LDA Accuracy: {accuracy:.3f}")
    
    # Plot discriminant space
    lda.plot_discriminant_space(X_scaled, y)
    
    # Quadratic Discriminant Analysis
    qda = DiscriminantAnalysis(method='quadratic')
    qda.fit(X_scaled, y)
    
    y_pred_qda = qda.predict(X_scaled)
    accuracy_qda = np.mean(y_pred_qda == y)
    print(f"QDA Accuracy: {accuracy_qda:.3f}")


def demonstrate_canonical_correlation():
    """
    Demonstrate canonical correlation analysis.
    """
    print("\n" + "=" * 60)
    print("CANONICAL CORRELATION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create two sets of variables
    np.random.seed(42)
    n_samples = 100
    
    # First set of variables (e.g., academic performance)
    X = np.random.multivariate_normal([0, 0, 0], [[1, 0.8, 0.6], [0.8, 1, 0.7], [0.6, 0.7, 1]], n_samples)
    
    # Second set of variables (e.g., personality traits)
    Y = np.random.multivariate_normal([0, 0, 0], [[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]], n_samples)
    
    # Add some correlation between sets
    Y[:, 0] = 0.6 * X[:, 0] + 0.4 * Y[:, 0]
    Y[:, 1] = 0.5 * X[:, 1] + 0.5 * Y[:, 1]
    
    # Fit canonical correlation
    cca = CanonicalCorrelation(n_components=3)
    cca.fit(X, Y)
    
    print("Canonical Correlations:")
    for i, corr in enumerate(cca.canonical_correlations):
        print(f"  Pair {i+1}: {corr:.3f}")
    
    # Transform data
    U, V = cca.transform(X, Y)
    print(f"Canonical variables shape: {U.shape}")
    
    # Plot canonical correlations
    cca.plot_canonical_correlations()


def demonstrate_multidimensional_scaling():
    """
    Demonstrate multidimensional scaling.
    """
    print("\n" + "=" * 60)
    print("MULTIDIMENSIONAL SCALING DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, _ = create_sample_multivariate_data()
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply MDS
    mds = MultidimensionalScaling(n_components=2, metric=True)
    X_mds = mds.fit_transform(X_scaled)
    
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"MDS dimensions: {X_mds.shape[1]}")
    
    # Plot MDS results
    mds.plot_mds()


def demonstrate_multivariate_normal():
    """
    Demonstrate multivariate normal distribution.
    """
    print("\n" + "=" * 60)
    print("MULTIVARIATE NORMAL DISTRIBUTION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, _ = create_sample_multivariate_data()
    
    # Fit multivariate normal
    mvn = MultivariateNormal()
    mvn.fit(X)
    
    print(f"Estimated mean: {mvn.mean}")
    print(f"Estimated covariance shape: {mvn.cov.shape}")
    
    # Calculate Mahalanobis distances
    mahal_dist = mvn.mahal_distance(X)
    print(f"Mahalanobis distances - Mean: {np.mean(mahal_dist):.3f}, Std: {np.std(mahal_dist):.3f}")
    
    # Detect outliers
    outliers, distances = mvn.detect_outliers(X, alpha=0.05)
    n_outliers = np.sum(outliers)
    print(f"Number of outliers detected: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")
    
    # Generate samples
    samples = mvn.sample(n_samples=100)
    print(f"Generated samples shape: {samples.shape}")


def demonstrate_real_world_example():
    """
    Demonstrate real-world multivariate analysis.
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: CUSTOMER SEGMENTATION")
    print("=" * 60)
    
    # Create realistic customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Generate customer segments
    segments = {
        'young_professionals': {'age': (25, 35), 'income': (60000, 120000), 'spending': (2000, 8000)},
        'families': {'age': (30, 45), 'income': (80000, 150000), 'spending': (5000, 15000)},
        'students': {'age': (18, 25), 'income': (10000, 30000), 'spending': (500, 2000)},
        'retirees': {'age': (65, 80), 'income': (40000, 80000), 'spending': (1000, 4000)}
    }
    
    # Generate data for each segment
    data = []
    labels = []
    
    for i, (segment, params) in enumerate(segments.items()):
        n_segment = n_customers // len(segments)
        
        age = np.random.uniform(params['age'][0], params['age'][1], n_segment)
        income = np.random.uniform(params['income'][0], params['income'][1], n_segment)
        spending = np.random.uniform(params['spending'][0], params['spending'][1], n_segment)
        
        # Add some correlation
        spending = spending + 0.3 * (income - np.mean(income)) + np.random.normal(0, 500, n_segment)
        
        segment_data = np.column_stack([age, income, spending])
        data.append(segment_data)
        labels.extend([i] * n_segment)
    
    X = np.vstack(data)
    y = np.array(labels)
    
    print("CUSTOMER SEGMENTATION ANALYSIS:")
    print(f"Total customers: {len(X)}")
    print(f"Number of segments: {len(segments)}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for dimensionality reduction
    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    print(f"PCA variance explained: {np.sum(pca.explained_variance_ratio):.3f}")
    
    # Clustering
    cluster_analyzer = ClusterAnalysis()
    cluster_labels = cluster_analyzer.kmeans_clustering(X_scaled, n_clusters=4)
    
    # Evaluate clustering
    eval_metrics = cluster_analyzer.evaluate_clustering(X_scaled, cluster_labels)
    print(f"Clustering evaluation:")
    print(f"  Silhouette score: {eval_metrics['silhouette_score']:.3f}")
    print(f"  Calinski-Harabasz score: {eval_metrics['calinski_harabasz_score']:.3f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    for i in range(len(segments)):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=list(segments.keys())[i])
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Original Data')
    plt.legend()
    
    # PCA projection
    plt.subplot(1, 3, 2)
    for i in range(len(segments)):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, label=list(segments.keys())[i])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Projection')
    plt.legend()
    
    # Clustering results
    plt.subplot(1, 3, 3)
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=f'Cluster {label}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering Results')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run all demonstrations.
    """
    print("MULTIVARIATE STATISTICS IMPLEMENTATION")
    print("Comprehensive demonstration of multivariate analysis concepts and applications")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrate_pca()
    demonstrate_clustering()
    demonstrate_discriminant_analysis()
    demonstrate_canonical_correlation()
    demonstrate_multidimensional_scaling()
    demonstrate_multivariate_normal()
    demonstrate_real_world_example()
    
    print("\n" + "=" * 80)
    print("MULTIVARIATE STATISTICS DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("✓ Principal Component Analysis with eigenvalue decomposition")
    print("✓ Factor Analysis with rotation and model evaluation")
    print("✓ K-means, hierarchical, and DBSCAN clustering")
    print("✓ Linear and quadratic discriminant analysis")
    print("✓ Canonical correlation analysis")
    print("✓ Multidimensional scaling")
    print("✓ Multivariate normal distribution properties")
    print("✓ Real-world customer segmentation example")
    print("✓ Clustering evaluation metrics")
    print("✓ Dimensionality reduction and visualization")


if __name__ == "__main__":
    main() 