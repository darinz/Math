# Descriptive Statistics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)

Descriptive statistics provide a way to summarize and describe the main features of a dataset. In AI/ML and data science, understanding your data through descriptive statistics is the first crucial step before building any models.

## Table of Contents
- [Measures of Central Tendency](#measures-of-central-tendency)
- [Measures of Dispersion](#measures-of-dispersion)
- [Data Visualization](#data-visualization)
- [Data Distribution](#data-distribution)
- [Correlation Analysis](#correlation-analysis)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample datasets for demonstration
np.random.seed(42)

# Normal distribution
normal_data = np.random.normal(100, 15, 1000)

# Skewed distribution
skewed_data = np.random.exponential(2, 1000)

# Multiple variables for correlation
x = np.random.normal(0, 1, 1000)
y = 0.7 * x + np.random.normal(0, 0.5, 1000)
z = -0.3 * x + 0.4 * y + np.random.normal(0, 0.3, 1000)

# Create DataFrame
df = pd.DataFrame({
    'normal': normal_data,
    'skewed': skewed_data,
    'x': x,
    'y': y,
    'z': z
})
```

## Measures of Central Tendency

Central tendency measures describe the center or typical value of a dataset.

### Mean (Arithmetic Average)

The mean is the sum of all values divided by the number of values.

```python
def calculate_mean(data):
    """Calculate arithmetic mean of a dataset"""
    return np.sum(data) / len(data)

# Example calculations
mean_normal = calculate_mean(normal_data)
mean_skewed = calculate_mean(skewed_data)

print(f"Mean of normal distribution: {mean_normal:.2f}")
print(f"Mean of skewed distribution: {mean_skewed:.2f}")

# Using NumPy
print(f"NumPy mean - normal: {np.mean(normal_data):.2f}")
print(f"NumPy mean - skewed: {np.mean(skewed_data):.2f}")

# Pandas DataFrame
print(f"DataFrame mean:\n{df.mean()}")
```

### Median

The median is the middle value when data is ordered. It's robust to outliers.

```python
def calculate_median(data):
    """Calculate median of a dataset"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

# Example with outliers
data_with_outliers = np.append(normal_data, [1000, 2000])
print(f"Mean with outliers: {np.mean(data_with_outliers):.2f}")
print(f"Median with outliers: {np.median(data_with_outliers):.2f}")

# Compare mean vs median for skewed data
print(f"Skewed data - Mean: {np.mean(skewed_data):.2f}, Median: {np.median(skewed_data):.2f}")
```

### Mode

The mode is the most frequently occurring value.

```python
from scipy.stats import mode

def calculate_mode(data):
    """Calculate mode of a dataset"""
    return mode(data, keepdims=True)

# Example
categorical_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D']
mode_result = calculate_mode(categorical_data)
print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# For continuous data, we can use histogram bins
hist, bins = np.histogram(normal_data, bins=20)
mode_bin = bins[np.argmax(hist)]
print(f"Mode bin center: {mode_bin:.2f}")
```

## Measures of Dispersion

Dispersion measures describe how spread out the data is.

### Variance and Standard Deviation

```python
def calculate_variance(data, sample=True):
    """Calculate variance of a dataset"""
    mean_val = np.mean(data)
    n = len(data)
    if sample:
        return np.sum((data - mean_val)**2) / (n - 1)
    else:
        return np.sum((data - mean_val)**2) / n

def calculate_std(data, sample=True):
    """Calculate standard deviation of a dataset"""
    return np.sqrt(calculate_variance(data, sample))

# Example calculations
variance_normal = calculate_variance(normal_data)
std_normal = calculate_std(normal_data)

print(f"Variance: {variance_normal:.2f}")
print(f"Standard Deviation: {std_normal:.2f}")
print(f"NumPy std: {np.std(normal_data, ddof=1):.2f}")

# Population vs Sample
print(f"Sample std: {np.std(normal_data, ddof=1):.2f}")
print(f"Population std: {np.std(normal_data, ddof=0):.2f}")
```

### Range and Interquartile Range (IQR)

```python
def calculate_range(data):
    """Calculate range of a dataset"""
    return np.max(data) - np.min(data)

def calculate_iqr(data):
    """Calculate interquartile range"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1

# Example
print(f"Range: {calculate_range(normal_data):.2f}")
print(f"IQR: {calculate_iqr(normal_data):.2f}")

# Five-number summary
def five_number_summary(data):
    """Calculate five-number summary"""
    return {
        'min': np.min(data),
        'q1': np.percentile(data, 25),
        'median': np.median(data),
        'q3': np.percentile(data, 75),
        'max': np.max(data)
    }

summary = five_number_summary(normal_data)
for key, value in summary.items():
    print(f"{key.upper()}: {value:.2f}")
```

## Data Visualization

### Histograms and Density Plots

```python
# Create subplots for comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram
axes[0, 0].hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Histogram - Normal Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Density plot
axes[0, 1].hist(normal_data, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Density Plot - Normal Distribution')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')

# Skewed distribution
axes[1, 0].hist(skewed_data, bins=30, alpha=0.7, color='salmon', edgecolor='black')
axes[1, 0].set_title('Histogram - Skewed Distribution')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

# Box plot
axes[1, 1].boxplot([normal_data, skewed_data], labels=['Normal', 'Skewed'])
axes[1, 1].set_title('Box Plot Comparison')
axes[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()
```

### Box Plots and Violin Plots

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot
axes[0].boxplot([normal_data, skewed_data], labels=['Normal', 'Skewed'])
axes[0].set_title('Box Plot')
axes[0].set_ylabel('Value')

# Violin plot
axes[1].violinplot([normal_data, skewed_data], labels=['Normal', 'Skewed'])
axes[1].set_title('Violin Plot')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()
```

## Data Distribution

### Skewness and Kurtosis

```python
def calculate_skewness(data):
    """Calculate skewness of a dataset"""
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    n = len(data)
    skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean_val) / std_val)**3)
    return skewness

def calculate_kurtosis(data):
    """Calculate kurtosis of a dataset"""
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    n = len(data)
    kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean_val) / std_val)**4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
    return kurtosis

# Calculate for our datasets
print(f"Normal distribution - Skewness: {calculate_skewness(normal_data):.3f}")
print(f"Normal distribution - Kurtosis: {calculate_kurtosis(normal_data):.3f}")
print(f"Skewed distribution - Skewness: {calculate_skewness(skewed_data):.3f}")
print(f"Skewed distribution - Kurtosis: {calculate_kurtosis(skewed_data):.3f}")

# Using SciPy
print(f"SciPy skewness - Normal: {stats.skew(normal_data):.3f}")
print(f"SciPy kurtosis - Normal: {stats.kurtosis(normal_data):.3f}")
```

### Distribution Fitting

```python
# Fit normal distribution to data
mu, sigma = stats.norm.fit(normal_data)
print(f"Fitted normal distribution - μ: {mu:.2f}, σ: {sigma:.2f}")

# Generate fitted curve
x = np.linspace(normal_data.min(), normal_data.max(), 100)
y_fitted = stats.norm.pdf(x, mu, sigma)

# Plot fitted distribution
plt.figure(figsize=(10, 6))
plt.hist(normal_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Data')
plt.plot(x, y_fitted, 'r-', linewidth=2, label='Fitted Normal Distribution')
plt.title('Normal Distribution Fitting')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

## Correlation Analysis

### Pearson Correlation

```python
def calculate_pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    
    return numerator / denominator

# Calculate correlations
corr_xy = calculate_pearson_correlation(x, y)
corr_xz = calculate_pearson_correlation(x, z)
corr_yz = calculate_pearson_correlation(y, z)

print(f"Correlation X-Y: {corr_xy:.3f}")
print(f"Correlation X-Z: {corr_xz:.3f}")
print(f"Correlation Y-Z: {corr_yz:.3f}")

# Using NumPy
print(f"NumPy correlation X-Y: {np.corrcoef(x, y)[0, 1]:.3f}")
```

### Correlation Matrix and Heatmap

```python
# Calculate correlation matrix
correlation_matrix = df[['x', 'y', 'z']].corr()

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Scatter plot matrix
sns.pairplot(df[['x', 'y', 'z']], diag_kind='kde')
plt.show()
```

## Practical Applications

### Data Quality Assessment

```python
def data_quality_report(df):
    """Generate comprehensive data quality report"""
    report = {}
    
    for column in df.columns:
        data = df[column].dropna()
        
        report[column] = {
            'count': len(data),
            'missing': len(df[column]) - len(data),
            'missing_pct': (len(df[column]) - len(data)) / len(df[column]) * 100,
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    return pd.DataFrame(report).T

# Generate report
quality_report = data_quality_report(df)
print("Data Quality Report:")
print(quality_report.round(3))
```

### Outlier Detection

```python
def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > threshold]
    return outliers

# Detect outliers
outliers_iqr, lb, ub = detect_outliers_iqr(normal_data)
outliers_zscore = detect_outliers_zscore(normal_data)

print(f"IQR outliers: {len(outliers_iqr)} ({len(outliers_iqr)/len(normal_data)*100:.1f}%)")
print(f"Z-score outliers: {len(outliers_zscore)} ({len(outliers_zscore)/len(normal_data)*100:.1f}%)")

# Visualize outliers
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.boxplot(normal_data)
plt.title('Box Plot with Outliers')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
z_scores = np.abs(stats.zscore(normal_data))
plt.scatter(range(len(normal_data)), z_scores, alpha=0.6)
plt.axhline(y=3, color='r', linestyle='--', label='Threshold (3)')
plt.title('Z-Score Outlier Detection')
plt.xlabel('Data Point')
plt.ylabel('Absolute Z-Score')
plt.legend()

plt.tight_layout()
plt.show()
```

## Summary Statistics with Pandas

```python
# Comprehensive summary
print("Comprehensive Summary Statistics:")
print(df.describe())

# Additional statistics
print("\nAdditional Statistics:")
print(f"Skewness:\n{df.skew()}")
print(f"Kurtosis:\n{df.kurtosis()}")

# Group by statistics (if categorical data available)
# Example with simulated categories
df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
print(f"\nGroup by Statistics:")
print(df.groupby('category')['normal'].describe())
```

## Practice Problems

1. **Data Exploration**: Load a real dataset (e.g., from sklearn.datasets) and generate a comprehensive descriptive statistics report.

2. **Outlier Analysis**: Create a function that compares different outlier detection methods and visualizes their results.

3. **Distribution Comparison**: Generate multiple distributions (normal, exponential, uniform) and compare their descriptive statistics.

4. **Correlation Study**: Analyze correlations in a multivariate dataset and create a correlation heatmap with significance levels.

## Further Reading

- "Statistics in Plain English" by Timothy C. Urdan
- "The Art of Statistics" by David Spiegelhalter
- Python Data Science Handbook by Jake VanderPlas
- NumPy and Pandas documentation for statistical functions

## Key Takeaways

- **Central tendency** measures help understand the typical value in your data
- **Dispersion** measures indicate how spread out your data is
- **Visualization** is crucial for understanding data distributions
- **Correlation** analysis reveals relationships between variables
- **Data quality** assessment should always precede modeling
- **Outlier detection** helps identify unusual data points that may need special treatment

In the next chapter, we'll explore probability fundamentals, which form the foundation for statistical inference and machine learning algorithms. 