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

Central tendency measures describe the center or typical value of a dataset. These measures help us understand where the "middle" of our data lies, which is crucial for understanding the distribution and making informed decisions in data analysis.

### Mean (Arithmetic Average)

The **arithmetic mean** is the most commonly used measure of central tendency. It represents the sum of all values divided by the number of values.

**Mathematical Definition:**
For a dataset with n observations: x₁, x₂, ..., xₙ

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + ... + x_n}{n}$$

**Properties of the Mean:**
1. **Linearity**: If we multiply each value by a constant c and add a constant a, the new mean becomes: $\bar{x}_{new} = c\bar{x} + a$
2. **Minimizes Sum of Squared Deviations**: The mean minimizes $\sum_{i=1}^{n} (x_i - \bar{x})^2$
3. **Sensitivity to Outliers**: The mean is heavily influenced by extreme values

**When to Use:**
- Data is approximately normally distributed
- No extreme outliers
- Need a measure that uses all data points

```python
def calculate_mean(data):
    """
    Calculate arithmetic mean of a dataset
    
    Mathematical implementation:
    mean = (sum of all values) / (number of values)
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: arithmetic mean
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate mean of empty dataset")
    
    return np.sum(data) / len(data)

# Example calculations with detailed explanation
mean_normal = calculate_mean(normal_data)
mean_skewed = calculate_mean(skewed_data)

print(f"Mean of normal distribution: {mean_normal:.2f}")
print(f"Mean of skewed distribution: {mean_skewed:.2f}")

# Mathematical verification
print(f"Sum of normal data: {np.sum(normal_data):.2f}")
print(f"Count of normal data: {len(normal_data)}")
print(f"Calculated mean: {np.sum(normal_data) / len(normal_data):.2f}")

# Using NumPy (vectorized computation)
print(f"NumPy mean - normal: {np.mean(normal_data):.2f}")
print(f"NumPy mean - skewed: {np.mean(skewed_data):.2f}")

# Pandas DataFrame (handles missing values automatically)
print(f"DataFrame mean:\n{df.mean()}")

# Demonstrate linearity property
c, a = 2, 10
transformed_data = c * normal_data + a
print(f"Original mean: {np.mean(normal_data):.2f}")
print(f"Transformed mean: {np.mean(transformed_data):.2f}")
print(f"Expected: {c * np.mean(normal_data) + a:.2f}")
```

### Median

The **median** is the middle value when data is ordered from smallest to largest. It's a robust measure that is not affected by extreme values.

**Mathematical Definition:**
For ordered data: x₁ ≤ x₂ ≤ ... ≤ xₙ

$$\text{Median} = \begin{cases} 
x_{\frac{n+1}{2}} & \text{if } n \text{ is odd} \\
\frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2} & \text{if } n \text{ is even}
\end{cases}$$

**Properties of the Median:**
1. **Robustness**: Unaffected by extreme values (outliers)
2. **Order Preservation**: If all values are multiplied by a positive constant, the median is multiplied by the same constant
3. **Minimizes Sum of Absolute Deviations**: The median minimizes $\sum_{i=1}^{n} |x_i - \text{median}|$

**When to Use:**
- Data has outliers or is skewed
- Need a robust measure of central tendency
- Ordinal data where order matters but differences don't

```python
def calculate_median(data):
    """
    Calculate median of a dataset
    
    Mathematical implementation:
    - Sort the data
    - If n is odd: median = middle value
    - If n is even: median = average of two middle values
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: median value
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    if n == 0:
        raise ValueError("Cannot calculate median of empty dataset")
    
    if n % 2 == 0:
        # Even number of elements: average of two middle values
        mid1 = sorted_data[n//2 - 1]
        mid2 = sorted_data[n//2]
        return (mid1 + mid2) / 2
    else:
        # Odd number of elements: middle value
        return sorted_data[n//2]

# Example with outliers to demonstrate robustness
data_with_outliers = np.append(normal_data, [1000, 2000])
print(f"Mean with outliers: {np.mean(data_with_outliers):.2f}")
print(f"Median with outliers: {np.median(data_with_outliers):.2f}")

# Compare mean vs median for skewed data
print(f"Skewed data - Mean: {np.mean(skewed_data):.2f}, Median: {np.median(skewed_data):.2f}")

# Demonstrate order preservation property
c = 2
transformed_data = c * normal_data
print(f"Original median: {np.median(normal_data):.2f}")
print(f"Transformed median: {np.median(transformed_data):.2f}")
print(f"Expected: {c * np.median(normal_data):.2f}")

# Mathematical verification of median calculation
sorted_normal = np.sort(normal_data)
n = len(sorted_normal)
if n % 2 == 0:
    median_calc = (sorted_normal[n//2 - 1] + sorted_normal[n//2]) / 2
else:
    median_calc = sorted_normal[n//2]
print(f"Manual median calculation: {median_calc:.2f}")
print(f"NumPy median: {np.median(normal_data):.2f}")
```

### Mode

The **mode** is the most frequently occurring value in a dataset. A dataset can have one mode (unimodal), two modes (bimodal), or more modes (multimodal).

**Mathematical Definition:**
For discrete data, the mode is the value x that maximizes the frequency function f(x):

$$\text{Mode} = \arg\max_{x} f(x)$$

For continuous data, the mode is the value that maximizes the probability density function.

**Properties of the Mode:**
1. **Not Unique**: A dataset can have multiple modes
2. **Categorical Data**: Works well with nominal and ordinal data
3. **Peak of Distribution**: Represents the most common value

**When to Use:**
- Categorical or discrete data
- Need to identify the most common category
- Data has clear peaks in distribution

```python
from scipy.stats import mode

def calculate_mode(data):
    """
    Calculate mode of a dataset
    
    Mathematical implementation:
    - Count frequency of each unique value
    - Return value(s) with highest frequency
    
    Parameters:
    data: array-like, input data
    
    Returns:
    tuple: (mode_value, count)
    """
    return mode(data, keepdims=True)

# Example with categorical data
categorical_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D']
mode_result = calculate_mode(categorical_data)
print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# Manual calculation for categorical data
from collections import Counter
counter = Counter(categorical_data)
most_common = counter.most_common(1)[0]
print(f"Manual calculation - Mode: {most_common[0]} (appears {most_common[1]} times)")

# For continuous data, we can use histogram bins to approximate mode
hist, bins = np.histogram(normal_data, bins=20)
mode_bin_index = np.argmax(hist)
mode_bin_center = (bins[mode_bin_index] + bins[mode_bin_index + 1]) / 2
print(f"Mode bin center: {mode_bin_center:.2f}")

# Demonstrate multimodal data
multimodal_data = np.concatenate([
    np.random.normal(0, 1, 200),
    np.random.normal(5, 1, 200),
    np.random.normal(10, 1, 200)
])

# Find multiple modes using histogram
hist_multi, bins_multi = np.histogram(multimodal_data, bins=30)
# Find local maxima
from scipy.signal import find_peaks
peaks, _ = find_peaks(hist_multi, height=np.max(hist_multi)*0.5)
mode_centers = [(bins_multi[i] + bins_multi[i+1])/2 for i in peaks]
print(f"Multiple modes detected at: {[f'{x:.2f}' for x in mode_centers]}")
```

### Geometric Mean

The **geometric mean** is useful for data that represents rates of change or multiplicative relationships.

**Mathematical Definition:**
$$\text{Geometric Mean} = \sqrt[n]{x_1 \times x_2 \times ... \times x_n} = \left(\prod_{i=1}^{n} x_i\right)^{\frac{1}{n}}$$

**Properties:**
1. **Logarithmic Relationship**: $\log(\text{GM}) = \frac{1}{n}\sum_{i=1}^{n} \log(x_i)$
2. **Multiplicative Data**: Appropriate for growth rates, ratios, and percentages
3. **Always ≤ Arithmetic Mean**: By the arithmetic mean-geometric mean inequality

```python
def geometric_mean(data):
    """
    Calculate geometric mean of a dataset
    
    Mathematical implementation:
    GM = (product of all values)^(1/n)
    
    Parameters:
    data: array-like, positive values
    
    Returns:
    float: geometric mean
    """
    if np.any(data <= 0):
        raise ValueError("Geometric mean requires all positive values")
    
    return np.exp(np.mean(np.log(data)))

# Example with growth rates
growth_rates = [1.05, 1.12, 0.98, 1.08, 1.15]  # 5% growth, 12% growth, etc.
gm = geometric_mean(growth_rates)
am = np.mean(growth_rates)

print(f"Growth rates: {growth_rates}")
print(f"Geometric mean: {gm:.4f}")
print(f"Arithmetic mean: {am:.4f}")
print(f"GM ≤ AM: {gm <= am}")

# Demonstrate logarithmic relationship
log_gm = np.mean(np.log(growth_rates))
print(f"Log of geometric mean: {log_gm:.4f}")
print(f"Exp of log mean: {np.exp(log_gm):.4f}")
```

### Harmonic Mean

The **harmonic mean** is useful for rates, speeds, and other situations involving reciprocals.

**Mathematical Definition:**
$$\text{Harmonic Mean} = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

**Properties:**
1. **Reciprocal Relationship**: Appropriate for rates and speeds
2. **Always ≤ Geometric Mean ≤ Arithmetic Mean**: For positive data
3. **Weighted Version**: $\text{HM} = \frac{\sum w_i}{\sum \frac{w_i}{x_i}}$

```python
def harmonic_mean(data):
    """
    Calculate harmonic mean of a dataset
    
    Mathematical implementation:
    HM = n / (sum of reciprocals)
    
    Parameters:
    data: array-like, non-zero values
    
    Returns:
    float: harmonic mean
    """
    if np.any(data == 0):
        raise ValueError("Harmonic mean requires all non-zero values")
    
    return len(data) / np.sum(1 / data)

# Example with speeds
speeds = [60, 40, 80]  # km/h for different segments
hm = harmonic_mean(speeds)
am = np.mean(speeds)

print(f"Speeds: {speeds} km/h")
print(f"Harmonic mean: {hm:.2f} km/h")
print(f"Arithmetic mean: {am:.2f} km/h")

# Demonstrate the relationship: HM ≤ GM ≤ AM
gm = geometric_mean(speeds)
print(f"Geometric mean: {gm:.2f} km/h")
print(f"HM ≤ GM ≤ AM: {hm <= gm <= am}")
```

## Measures of Dispersion

Dispersion measures describe how spread out the data is around the central tendency. Understanding dispersion is crucial for assessing the reliability of the central tendency measures and the variability in your data.

### Variance and Standard Deviation

**Variance** measures the average squared deviation from the mean, while **standard deviation** is the square root of variance.

**Mathematical Definition:**

**Population Variance:**
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2$$

**Sample Variance (Bessel's correction):**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Standard Deviation:**
$$\sigma = \sqrt{\sigma^2} \quad \text{or} \quad s = \sqrt{s^2}$$

**Why n-1 for Sample Variance?**
The n-1 correction (Bessel's correction) makes the sample variance an unbiased estimator of the population variance. This is because:
1. We estimate the population mean μ with the sample mean x̄
2. This estimation reduces the degrees of freedom by 1
3. Using n-1 compensates for this reduction

**Properties:**
1. **Non-negative**: Variance is always ≥ 0
2. **Scale Dependent**: If we multiply data by c, variance becomes c² times original
3. **Translation Invariant**: Adding a constant doesn't change variance
4. **Additive for Independent Variables**: Var(X+Y) = Var(X) + Var(Y) if X,Y independent

```python
def calculate_variance(data, sample=True):
    """
    Calculate variance of a dataset
    
    Mathematical implementation:
    - Calculate mean
    - Calculate squared deviations from mean
    - Average the squared deviations
    - Use n-1 for sample variance (Bessel's correction)
    
    Parameters:
    data: array-like, input data
    sample: bool, if True use n-1 (sample variance), if False use n (population variance)
    
    Returns:
    float: variance
    """
    mean_val = np.mean(data)
    n = len(data)
    
    if n == 0:
        raise ValueError("Cannot calculate variance of empty dataset")
    
    squared_deviations = (data - mean_val)**2
    
    if sample and n > 1:
        # Sample variance: use n-1 (Bessel's correction)
        return np.sum(squared_deviations) / (n - 1)
    else:
        # Population variance: use n
        return np.sum(squared_deviations) / n

def calculate_std(data, sample=True):
    """
    Calculate standard deviation of a dataset
    
    Mathematical implementation:
    std = sqrt(variance)
    
    Parameters:
    data: array-like, input data
    sample: bool, if True use n-1 (sample std), if False use n (population std)
    
    Returns:
    float: standard deviation
    """
    return np.sqrt(calculate_variance(data, sample))

# Example calculations with detailed explanation
variance_normal = calculate_variance(normal_data)
std_normal = calculate_std(normal_data)

print(f"Variance: {variance_normal:.2f}")
print(f"Standard Deviation: {std_normal:.2f}")
print(f"NumPy std: {np.std(normal_data, ddof=1):.2f}")

# Demonstrate Bessel's correction
print(f"Sample variance (n-1): {np.var(normal_data, ddof=1):.2f}")
print(f"Population variance (n): {np.var(normal_data, ddof=0):.2f}")
print(f"Difference: {np.var(normal_data, ddof=1) - np.var(normal_data, ddof=0):.2f}")

# Mathematical verification
mean_val = np.mean(normal_data)
squared_deviations = (normal_data - mean_val)**2
manual_variance = np.sum(squared_deviations) / (len(normal_data) - 1)
print(f"Manual calculation: {manual_variance:.2f}")

# Demonstrate scale property
c = 2
scaled_data = c * normal_data
print(f"Original std: {np.std(normal_data):.2f}")
print(f"Scaled std: {np.std(scaled_data):.2f}")
print(f"Expected: {c * np.std(normal_data):.2f}")

# Demonstrate translation invariance
a = 10
translated_data = normal_data + a
print(f"Original std: {np.std(normal_data):.2f}")
print(f"Translated std: {np.std(translated_data):.2f}")
```

### Range and Interquartile Range (IQR)

**Range** is the difference between the maximum and minimum values, while **IQR** is the difference between the 75th and 25th percentiles.

**Mathematical Definition:**

**Range:**
$$\text{Range} = x_{max} - x_{min}$$

**IQR:**
$$\text{IQR} = Q_3 - Q_1$$

Where Q₁ (25th percentile) and Q₃ (75th percentile) are defined as:
- Q₁: Value below which 25% of data falls
- Q₃: Value below which 75% of data falls

**Properties:**
1. **Range**: Simple but sensitive to outliers
2. **IQR**: Robust measure of spread, not affected by outliers
3. **Percentiles**: Q₁, Q₂ (median), Q₃ provide five-number summary

```python
def calculate_range(data):
    """
    Calculate range of a dataset
    
    Mathematical implementation:
    range = max - min
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: range
    """
    return np.max(data) - np.min(data)

def calculate_iqr(data):
    """
    Calculate interquartile range
    
    Mathematical implementation:
    IQR = Q3 - Q1
    where Q1 = 25th percentile, Q3 = 75th percentile
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: interquartile range
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return q3 - q1

# Example
print(f"Range: {calculate_range(normal_data):.2f}")
print(f"IQR: {calculate_iqr(normal_data):.2f}")

# Five-number summary
def five_number_summary(data):
    """
    Calculate five-number summary
    
    Mathematical implementation:
    - Minimum: smallest value
    - Q1: 25th percentile
    - Median: 50th percentile
    - Q3: 75th percentile
    - Maximum: largest value
    
    Parameters:
    data: array-like, input data
    
    Returns:
    dict: five-number summary
    """
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

# Demonstrate robustness of IQR vs Range with outliers
data_with_outliers = np.append(normal_data, [1000, 2000])
print(f"Original range: {calculate_range(normal_data):.2f}")
print(f"Range with outliers: {calculate_range(data_with_outliers):.2f}")
print(f"Original IQR: {calculate_iqr(normal_data):.2f}")
print(f"IQR with outliers: {calculate_iqr(data_with_outliers):.2f}")

# Mathematical verification of percentiles
sorted_data = np.sort(normal_data)
n = len(sorted_data)

# Q1 (25th percentile)
q1_index = 0.25 * (n - 1)
q1_lower = int(q1_index)
q1_upper = q1_lower + 1
q1_weight = q1_index - q1_lower
q1_manual = (1 - q1_weight) * sorted_data[q1_lower] + q1_weight * sorted_data[q1_upper]
print(f"Manual Q1: {q1_manual:.2f}")
print(f"NumPy Q1: {np.percentile(normal_data, 25):.2f}")
```

### Coefficient of Variation (CV)

The **coefficient of variation** is a standardized measure of dispersion that expresses standard deviation as a percentage of the mean.

**Mathematical Definition:**
$$\text{CV} = \frac{s}{\bar{x}} \times 100\%$$

**Properties:**
1. **Dimensionless**: Allows comparison across different scales
2. **Relative Measure**: Shows dispersion relative to the mean
3. **Useful for**: Comparing variability across different datasets

```python
def coefficient_of_variation(data):
    """
    Calculate coefficient of variation
    
    Mathematical implementation:
    CV = (std / mean) * 100%
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: coefficient of variation as percentage
    """
    mean_val = np.mean(data)
    if mean_val == 0:
        raise ValueError("Cannot calculate CV when mean is zero")
    
    std_val = np.std(data, ddof=1)
    return (std_val / mean_val) * 100

# Example
cv_normal = coefficient_of_variation(normal_data)
cv_skewed = coefficient_of_variation(skewed_data)

print(f"CV of normal data: {cv_normal:.2f}%")
print(f"CV of skewed data: {cv_skewed:.2f}%")

# Compare variability across different scales
small_data = np.random.normal(10, 2, 1000)  # mean=10, std=2
large_data = np.random.normal(1000, 200, 1000)  # mean=1000, std=200

cv_small = coefficient_of_variation(small_data)
cv_large = coefficient_of_variation(large_data)

print(f"Small scale data CV: {cv_small:.2f}%")
print(f"Large scale data CV: {cv_large:.2f}%")
print(f"Same relative variability: {abs(cv_small - cv_large) < 0.1}")
```

### Mean Absolute Deviation (MAD)

**Mean absolute deviation** measures the average absolute deviation from the mean.

**Mathematical Definition:**
$$\text{MAD} = \frac{1}{n}\sum_{i=1}^{n} |x_i - \bar{x}|$$

**Properties:**
1. **Robust**: Less sensitive to outliers than variance
2. **Interpretable**: Same units as original data
3. **Computationally Simple**: No squaring required

```python
def mean_absolute_deviation(data):
    """
    Calculate mean absolute deviation
    
    Mathematical implementation:
    MAD = mean of absolute deviations from mean
    
    Parameters:
    data: array-like, input data
    
    Returns:
    float: mean absolute deviation
    """
    mean_val = np.mean(data)
    return np.mean(np.abs(data - mean_val))

# Example
mad_normal = mean_absolute_deviation(normal_data)
std_normal = np.std(normal_data, ddof=1)

print(f"MAD: {mad_normal:.2f}")
print(f"Standard deviation: {std_normal:.2f}")

# Relationship between MAD and standard deviation for normal distribution
# For normal distribution: MAD ≈ 0.7979 × σ
expected_mad = 0.7979 * std_normal
print(f"Expected MAD for normal distribution: {expected_mad:.2f}")
print(f"Actual MAD: {mad_normal:.2f}")
print(f"Ratio MAD/σ: {mad_normal/std_normal:.4f}")
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