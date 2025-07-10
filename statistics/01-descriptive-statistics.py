#!/usr/bin/env python3
"""
Descriptive Statistics Implementation
====================================

This file implements comprehensive descriptive statistics concepts with detailed
examples, visualizations, and educational annotations. It covers:

1. Measures of Central Tendency (Mean, Median, Mode, Geometric Mean, Harmonic Mean)
2. Measures of Dispersion (Variance, Standard Deviation, Range, IQR, CV, MAD)
3. Data Visualization (Histograms, Box Plots, Scatter Plots)
4. Distribution Analysis (Normal, Skewed, Multimodal)
5. Correlation Analysis (Pearson, Spearman)
6. Practical Applications and Real-world Examples

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# SECTION 1: MEASURES OF CENTRAL TENDENCY
# =============================================================================

def demonstrate_central_tendency():
    """
    Demonstrate different measures of central tendency with examples.
    
    This function shows how mean, median, and mode behave differently
    under various data distributions and conditions.
    """
    print("=" * 60)
    print("SECTION 1: MEASURES OF CENTRAL TENDENCY")
    print("=" * 60)
    
    # Example 1: Normal Distribution
    print("\n1.1 Normal Distribution Example")
    print("-" * 40)
    
    # Generate normal data
    normal_data = np.random.normal(100, 15, 1000)
    
    # Calculate measures
    mean_normal = np.mean(normal_data)
    median_normal = np.median(normal_data)
    mode_normal = stats.mode(normal_data)[0][0]
    
    print(f"Mean: {mean_normal:.2f}")
    print(f"Median: {median_normal:.2f}")
    print(f"Mode: {mode_normal:.2f}")
    print("Note: In normal distributions, mean ≈ median ≈ mode")
    
    # Example 2: Skewed Distribution (Income-like data)
    print("\n1.2 Skewed Distribution Example")
    print("-" * 40)
    
    # Generate skewed data (similar to income distribution)
    skewed_data = np.concatenate([
        np.random.normal(50, 10, 800),  # Most people
        np.random.normal(100, 20, 150), # Middle class
        np.random.normal(200, 50, 50)   # High earners
    ])
    
    mean_skewed = np.mean(skewed_data)
    median_skewed = np.median(skewed_data)
    mode_skewed = stats.mode(skewed_data)[0][0]
    
    print(f"Mean: {mean_skewed:.2f}")
    print(f"Median: {median_skewed:.2f}")
    print(f"Mode: {mode_skewed:.2f}")
    print("Note: In skewed distributions, mean ≠ median ≠ mode")
    print(f"Skewness: {skew(skewed_data):.3f}")
    
    # Example 3: Outlier Effect
    print("\n1.3 Outlier Effect Example")
    print("-" * 40)
    
    # Create dataset with outlier
    data_with_outlier = [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    
    mean_outlier = np.mean(data_with_outlier)
    median_outlier = np.median(data_with_outlier)
    
    print(f"Data: {data_with_outlier}")
    print(f"Mean: {mean_outlier:.2f}")
    print(f"Median: {median_outlier:.2f}")
    print("Note: Mean is heavily influenced by outlier, median is robust")
    
    # Example 4: Categorical Data (Mode)
    print("\n1.4 Categorical Data Example")
    print("-" * 40)
    
    colors = ['Red', 'Blue', 'Green', 'Blue', 'Yellow', 'Blue', 'Red']
    unique_colors, counts = np.unique(colors, return_counts=True)
    mode_color = unique_colors[np.argmax(counts)]
    
    print(f"Color data: {colors}")
    print(f"Mode: {mode_color}")
    print("Note: Mean and median don't make sense for categorical data")
    
    return {
        'normal': {'data': normal_data, 'mean': mean_normal, 'median': median_normal, 'mode': mode_normal},
        'skewed': {'data': skewed_data, 'mean': mean_skewed, 'median': median_skewed, 'mode': mode_skewed},
        'outlier': {'data': data_with_outlier, 'mean': mean_outlier, 'median': median_outlier}
    }

def demonstrate_specialized_means():
    """
    Demonstrate geometric mean and harmonic mean with practical examples.
    """
    print("\n1.5 Specialized Means")
    print("-" * 40)
    
    # Geometric Mean Example: Investment Returns
    print("Geometric Mean - Investment Returns:")
    returns = [1.10, 1.20, 0.95, 1.15, 1.08]  # 10%, 20%, -5%, 15%, 8%
    
    arithmetic_mean = np.mean(returns)
    geometric_mean = stats.gmean(returns)
    
    print(f"Annual returns: {returns}")
    print(f"Arithmetic mean: {arithmetic_mean:.4f}")
    print(f"Geometric mean: {geometric_mean:.4f}")
    print(f"Compound growth: {geometric_mean**5:.4f}")
    print("Note: Geometric mean is appropriate for multiplicative data")
    
    # Harmonic Mean Example: Average Speed
    print("\nHarmonic Mean - Average Speed:")
    speeds = [60, 40]  # km/h for equal distances
    distances = [100, 100]  # km
    
    arithmetic_speed = np.mean(speeds)
    harmonic_speed = stats.hmean(speeds)
    
    print(f"Speeds: {speeds} km/h")
    print(f"Arithmetic mean speed: {arithmetic_speed:.1f} km/h")
    print(f"Harmonic mean speed: {harmonic_speed:.1f} km/h")
    
    # Verify with actual calculation
    total_distance = sum(distances)
    total_time = sum(d/s for d, s in zip(distances, speeds))
    actual_avg_speed = total_distance / total_time
    print(f"Actual average speed: {actual_avg_speed:.1f} km/h")
    print("Note: Harmonic mean correctly represents average speed")

# =============================================================================
# SECTION 2: MEASURES OF DISPERSION
# =============================================================================

def demonstrate_dispersion():
    """
    Demonstrate different measures of dispersion and their properties.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: MEASURES OF DISPERSION")
    print("=" * 60)
    
    # Generate sample data
    data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    
    print("\n2.1 Basic Dispersion Measures")
    print("-" * 40)
    print(f"Data: {data}")
    
    # Range
    data_range = np.max(data) - np.min(data)
    print(f"Range: {data_range}")
    
    # Variance and Standard Deviation
    variance = np.var(data, ddof=1)  # Sample variance
    std_dev = np.std(data, ddof=1)   # Sample standard deviation
    print(f"Variance: {variance:.3f}")
    print(f"Standard Deviation: {std_dev:.3f}")
    
    # IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"IQR: {iqr}")
    
    # Coefficient of Variation
    mean_val = np.mean(data)
    cv = (std_dev / mean_val) * 100
    print(f"Coefficient of Variation: {cv:.1f}%")
    
    # Mean Absolute Deviation
    mad = np.mean(np.abs(data - mean_val))
    print(f"Mean Absolute Deviation: {mad:.3f}")
    
    # Five-Number Summary
    five_num = np.percentile(data, [0, 25, 50, 75, 100])
    print(f"Five-Number Summary: {five_num}")
    
    return {
        'data': data,
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'iqr': iqr,
        'cv': cv,
        'mad': mad,
        'five_num': five_num
    }

def demonstrate_outlier_detection():
    """
    Demonstrate outlier detection using IQR method.
    """
    print("\n2.2 Outlier Detection")
    print("-" * 40)
    
    # Create dataset with outliers
    data_with_outliers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 20])
    
    q1 = np.percentile(data_with_outliers, 25)
    q3 = np.percentile(data_with_outliers, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data_with_outliers[(data_with_outliers < lower_bound) | 
                                 (data_with_outliers > upper_bound)]
    
    print(f"Data: {data_with_outliers}")
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print(f"Outliers: {outliers}")
    
    return {
        'data': data_with_outliers,
        'outliers': outliers,
        'bounds': (lower_bound, upper_bound)
    }

# =============================================================================
# SECTION 3: DATA VISUALIZATION
# =============================================================================

def create_visualizations():
    """
    Create comprehensive visualizations for descriptive statistics.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: DATA VISUALIZATION")
    print("=" * 60)
    
    # Generate different types of data
    np.random.seed(42)
    
    # Normal distribution
    normal_data = np.random.normal(100, 15, 1000)
    
    # Skewed distribution
    skewed_data = np.concatenate([
        np.random.normal(50, 10, 800),
        np.random.normal(100, 20, 150),
        np.random.normal(200, 50, 50)
    ])
    
    # Bimodal distribution
    bimodal_data = np.concatenate([
        np.random.normal(80, 10, 500),
        np.random.normal(120, 10, 500)
    ])
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Descriptive Statistics Visualizations', fontsize=16)
    
    # 1. Histograms
    axes[0, 0].hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(normal_data), color='red', linestyle='--', label=f'Mean: {np.mean(normal_data):.1f}')
    axes[0, 0].axvline(np.median(normal_data), color='green', linestyle='--', label=f'Median: {np.median(normal_data):.1f}')
    axes[0, 0].set_title('Normal Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(skewed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(np.mean(skewed_data), color='red', linestyle='--', label=f'Mean: {np.mean(skewed_data):.1f}')
    axes[0, 1].axvline(np.median(skewed_data), color='green', linestyle='--', label=f'Median: {np.median(skewed_data):.1f}')
    axes[0, 1].set_title('Skewed Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].hist(bimodal_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].axvline(np.mean(bimodal_data), color='red', linestyle='--', label=f'Mean: {np.mean(bimodal_data):.1f}')
    axes[0, 2].axvline(np.median(bimodal_data), color='green', linestyle='--', label=f'Median: {np.median(bimodal_data):.1f}')
    axes[0, 2].set_title('Bimodal Distribution')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # 2. Box Plots
    data_for_box = [normal_data, skewed_data, bimodal_data]
    labels = ['Normal', 'Skewed', 'Bimodal']
    
    bp = axes[1, 0].boxplot(data_for_box, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_title('Box Plots Comparison')
    axes[1, 0].set_ylabel('Value')
    
    # 3. Scatter Plot (Correlation Example)
    x = np.random.normal(0, 1, 100)
    y = 0.7 * x + np.random.normal(0, 0.5, 100)
    
    axes[1, 1].scatter(x, y, alpha=0.6, color='purple')
    axes[1, 1].set_title('Positive Correlation Example')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1, 1].plot(x, p(x), "r--", alpha=0.8)
    
    # 4. Correlation Heatmap
    # Create correlated data
    np.random.seed(42)
    n = 100
    data_corr = pd.DataFrame({
        'A': np.random.normal(0, 1, n),
        'B': np.random.normal(0, 1, n),
        'C': np.random.normal(0, 1, n)
    })
    
    # Create correlations
    data_corr['B'] = 0.7 * data_corr['A'] + 0.3 * data_corr['B']
    data_corr['C'] = -0.5 * data_corr['A'] + 0.8 * data_corr['C']
    
    corr_matrix = data_corr.corr()
    
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Heatmap')
    axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 2].set_xticklabels(corr_matrix.columns)
    axes[1, 2].set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'normal_data': normal_data,
        'skewed_data': skewed_data,
        'bimodal_data': bimodal_data,
        'correlation_data': data_corr
    }

# =============================================================================
# SECTION 4: DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distributions():
    """
    Analyze different types of distributions and their characteristics.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Generate different distributions
    np.random.seed(42)
    
    # Normal distribution
    normal_data = np.random.normal(100, 15, 1000)
    
    # Right-skewed distribution (like income)
    right_skewed = np.random.exponential(50, 1000)
    
    # Left-skewed distribution
    left_skewed = 200 - np.random.exponential(30, 1000)
    left_skewed = left_skewed[left_skewed > 0]  # Remove negative values
    
    # Uniform distribution
    uniform_data = np.random.uniform(0, 100, 1000)
    
    # Analyze each distribution
    distributions = {
        'Normal': normal_data,
        'Right-Skewed': right_skewed,
        'Left-Skewed': left_skewed,
        'Uniform': uniform_data
    }
    
    print("\n4.1 Distribution Characteristics")
    print("-" * 40)
    
    for name, data in distributions.items():
        print(f"\n{name} Distribution:")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Median: {np.median(data):.2f}")
        print(f"  Standard Deviation: {np.std(data):.2f}")
        print(f"  Skewness: {skew(data):.3f}")
        print(f"  Kurtosis: {kurtosis(data):.3f}")
        
        # Check for normality
        _, p_value = stats.normaltest(data)
        print(f"  Normal test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  → Not normally distributed")
        else:
            print("  → Approximately normal")
    
    # Create distribution comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution Analysis', fontsize=16)
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, (name, data) in enumerate(distributions.items()):
        row = i // 2
        col = i % 2
        
        axes[row, col].hist(data, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        axes[row, col].axvline(np.mean(data), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(data):.1f}')
        axes[row, col].axvline(np.median(data), color='green', linestyle='--', 
                              label=f'Median: {np.median(data):.1f}')
        axes[row, col].set_title(f'{name} Distribution')
        axes[row, col].legend()
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return distributions

# =============================================================================
# SECTION 5: CORRELATION ANALYSIS
# =============================================================================

def demonstrate_correlation():
    """
    Demonstrate different types of correlation and their interpretation.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: CORRELATION ANALYSIS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate different correlation patterns
    n = 100
    
    # Strong positive correlation
    x1 = np.random.normal(0, 1, n)
    y1 = 0.9 * x1 + np.random.normal(0, 0.3, n)
    
    # Weak positive correlation
    x2 = np.random.normal(0, 1, n)
    y2 = 0.3 * x2 + np.random.normal(0, 0.9, n)
    
    # Strong negative correlation
    x3 = np.random.normal(0, 1, n)
    y3 = -0.8 * x3 + np.random.normal(0, 0.5, n)
    
    # No correlation
    x4 = np.random.normal(0, 1, n)
    y4 = np.random.normal(0, 1, n)
    
    # Non-linear relationship (quadratic)
    x5 = np.random.uniform(-2, 2, n)
    y5 = x5**2 + np.random.normal(0, 0.5, n)
    
    datasets = {
        'Strong Positive': (x1, y1),
        'Weak Positive': (x2, y2),
        'Strong Negative': (x3, y3),
        'No Correlation': (x4, y4),
        'Non-linear': (x5, y5)
    }
    
    print("\n5.1 Correlation Analysis")
    print("-" * 40)
    
    for name, (x, y) in datasets.items():
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        print(f"\n{name}:")
        print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.4f})")
        print(f"  Spearman ρ: {spearman_r:.3f} (p={spearman_p:.4f})")
        
        # Interpretation
        if abs(pearson_r) >= 0.7:
            strength = "Strong"
        elif abs(pearson_r) >= 0.3:
            strength = "Moderate"
        elif abs(pearson_r) >= 0.1:
            strength = "Weak"
        else:
            strength = "Negligible"
        
        direction = "Positive" if pearson_r > 0 else "Negative" if pearson_r < 0 else "No"
        print(f"  Interpretation: {strength} {direction} correlation")
    
    # Create correlation visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Correlation Examples', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'gray', 'purple']
    
    for i, (name, (x, y)) in enumerate(datasets.items()):
        row = i // 3
        col = i % 3
        
        axes[row, col].scatter(x, y, alpha=0.6, color=colors[i])
        axes[row, col].set_title(f'{name}\nr={stats.pearsonr(x, y)[0]:.3f}')
        axes[row, col].set_xlabel('X')
        axes[row, col].set_ylabel('Y')
        
        # Add trend line for linear relationships
        if name != 'Non-linear':
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[row, col].plot(x, p(x), "r--", alpha=0.8)
    
    # Remove the last subplot if not needed
    if len(datasets) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.show()
    
    return datasets

def demonstrate_correlation_vs_causation():
    """
    Demonstrate the difference between correlation and causation.
    """
    print("\n5.2 Correlation vs Causation")
    print("-" * 40)
    
    # Example: Ice cream sales vs drowning deaths
    np.random.seed(42)
    
    # Generate temperature data (confounding variable)
    temperature = np.random.uniform(20, 35, 100)
    
    # Both ice cream sales and drowning deaths depend on temperature
    ice_cream_sales = 50 + 2 * temperature + np.random.normal(0, 5, 100)
    drowning_deaths = 10 + 0.5 * temperature + np.random.normal(0, 2, 100)
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(ice_cream_sales, drowning_deaths)
    
    print(f"Ice Cream Sales vs Drowning Deaths:")
    print(f"Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.4f}")
    print("This is a spurious correlation!")
    print("Both variables are caused by temperature (confounding variable)")
    
    # Visualize the relationship
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(ice_cream_sales, drowning_deaths, alpha=0.6, color='red')
    ax1.set_xlabel('Ice Cream Sales')
    ax1.set_ylabel('Drowning Deaths')
    ax1.set_title(f'Correlation: {correlation:.3f}')
    
    # Add trend line
    z = np.polyfit(ice_cream_sales, drowning_deaths, 1)
    p = np.poly1d(z)
    ax1.plot(ice_cream_sales, p(ice_cream_sales), "b--", alpha=0.8)
    
    # Temperature vs both variables
    ax2.scatter(temperature, ice_cream_sales, alpha=0.6, color='blue', label='Ice Cream Sales')
    ax2.scatter(temperature, drowning_deaths, alpha=0.6, color='red', label='Drowning Deaths')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Value')
    ax2.set_title('Both variables depend on temperature')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'temperature': temperature,
        'ice_cream_sales': ice_cream_sales,
        'drowning_deaths': drowning_deaths,
        'correlation': correlation
    }

# =============================================================================
# SECTION 6: PRACTICAL APPLICATIONS
# =============================================================================

def practical_applications():
    """
    Demonstrate practical applications of descriptive statistics.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: PRACTICAL APPLICATIONS")
    print("=" * 60)
    
    # Example 1: Student Performance Analysis
    print("\n6.1 Student Performance Analysis")
    print("-" * 40)
    
    # Generate student data
    np.random.seed(42)
    n_students = 50
    
    # Create realistic student data
    student_data = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'math_score': np.random.normal(75, 10, n_students),
        'science_score': np.random.normal(78, 12, n_students),
        'english_score': np.random.normal(80, 8, n_students),
        'study_hours': np.random.exponential(10, n_students),
        'attendance': np.random.beta(5, 2, n_students) * 100
    })
    
    # Ensure scores are within reasonable bounds
    for col in ['math_score', 'science_score', 'english_score']:
        student_data[col] = np.clip(student_data[col], 0, 100)
    
    print("Student Performance Summary:")
    print(student_data.describe())
    
    # Analyze correlations
    score_cols = ['math_score', 'science_score', 'english_score']
    correlation_matrix = student_data[score_cols].corr()
    
    print("\nCorrelation Matrix (Subject Scores):")
    print(correlation_matrix)
    
    # Study hours vs performance
    study_corr = student_data[['study_hours'] + score_cols].corr()
    print("\nStudy Hours vs Performance:")
    print(study_corr['study_hours'][score_cols])
    
    # Example 2: Quality Control
    print("\n6.2 Quality Control Example")
    print("-" * 40)
    
    # Generate manufacturing data
    np.random.seed(42)
    n_products = 200
    
    # Normal production with some outliers
    product_weights = np.random.normal(100, 5, n_products)
    
    # Add some outliers (defective products)
    outliers = np.random.normal(120, 8, 10)
    product_weights = np.concatenate([product_weights, outliers])
    
    # Calculate control limits
    mean_weight = np.mean(product_weights)
    std_weight = np.std(product_weights)
    
    # 3-sigma control limits
    ucl = mean_weight + 3 * std_weight
    lcl = mean_weight - 3 * std_weight
    
    # Identify outliers
    outliers_detected = product_weights[(product_weights > ucl) | (product_weights < lcl)]
    
    print(f"Total products: {len(product_weights)}")
    print(f"Mean weight: {mean_weight:.2f} g")
    print(f"Standard deviation: {std_weight:.2f} g")
    print(f"Upper control limit: {ucl:.2f} g")
    print(f"Lower control limit: {lcl:.2f} g")
    print(f"Outliers detected: {len(outliers_detected)}")
    print(f"Outlier percentage: {len(outliers_detected)/len(product_weights)*100:.1f}%")
    
    # Create quality control chart
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(product_weights)), product_weights, 'o', alpha=0.6, label='Product Weights')
    plt.axhline(mean_weight, color='green', linestyle='-', label=f'Mean: {mean_weight:.2f}')
    plt.axhline(ucl, color='red', linestyle='--', label=f'UCL: {ucl:.2f}')
    plt.axhline(lcl, color='red', linestyle='--', label=f'LCL: {lcl:.2f}')
    plt.xlabel('Product Number')
    plt.ylabel('Weight (g)')
    plt.title('Quality Control Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'student_data': student_data,
        'product_weights': product_weights,
        'outliers_detected': outliers_detected,
        'control_limits': (lcl, ucl)
    }

# =============================================================================
# SECTION 7: EXERCISES AND PRACTICE PROBLEMS
# =============================================================================

def practice_exercises():
    """
    Provide practice exercises for descriptive statistics.
    """
    print("\n" + "=" * 60)
    print("SECTION 7: EXERCISES AND PRACTICE PROBLEMS")
    print("=" * 60)
    
    print("\nExercise 1: Central Tendency Comparison")
    print("-" * 40)
    
    # Generate different datasets
    np.random.seed(42)
    
    datasets = {
        'Normal': np.random.normal(100, 15, 100),
        'Skewed': np.random.exponential(50, 100),
        'Bimodal': np.concatenate([
            np.random.normal(80, 10, 50),
            np.random.normal(120, 10, 50)
        ]),
        'With Outliers': np.concatenate([
            np.random.normal(100, 10, 95),
            [200, 250, 300]  # Outliers
        ])
    }
    
    for name, data in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Median: {np.median(data):.2f}")
        print(f"  Mode: {stats.mode(data)[0][0]:.2f}")
        print(f"  Standard Deviation: {np.std(data):.2f}")
    
    print("\nExercise 2: Correlation Analysis")
    print("-" * 40)
    
    # Generate correlated data
    x = np.random.normal(0, 1, 100)
    y_strong = 0.8 * x + np.random.normal(0, 0.3, 100)
    y_weak = 0.2 * x + np.random.normal(0, 0.9, 100)
    y_none = np.random.normal(0, 1, 100)
    
    correlations = {
        'Strong Positive': stats.pearsonr(x, y_strong),
        'Weak Positive': stats.pearsonr(x, y_weak),
        'No Correlation': stats.pearsonr(x, y_none)
    }
    
    for name, (r, p) in correlations.items():
        print(f"{name}: r = {r:.3f}, p = {p:.4f}")
    
    return {
        'datasets': datasets,
        'correlations': correlations
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run all descriptive statistics demonstrations.
    """
    print("DESCRIPTIVE STATISTICS IMPLEMENTATION")
    print("=" * 60)
    print("This script demonstrates comprehensive descriptive statistics concepts")
    print("including measures of central tendency, dispersion, visualization,")
    print("distribution analysis, correlation, and practical applications.")
    print("=" * 60)
    
    # Run all demonstrations
    central_tendency_results = demonstrate_central_tendency()
    demonstrate_specialized_means()
    
    dispersion_results = demonstrate_dispersion()
    outlier_results = demonstrate_outlier_detection()
    
    viz_results = create_visualizations()
    
    distribution_results = analyze_distributions()
    
    correlation_results = demonstrate_correlation()
    causation_results = demonstrate_correlation_vs_causation()
    
    practical_results = practical_applications()
    
    exercise_results = practice_exercises()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This implementation covered:")
    print("✓ Measures of Central Tendency (Mean, Median, Mode, Geometric, Harmonic)")
    print("✓ Measures of Dispersion (Variance, SD, Range, IQR, CV, MAD)")
    print("✓ Data Visualization (Histograms, Box Plots, Scatter Plots)")
    print("✓ Distribution Analysis (Normal, Skewed, Multimodal)")
    print("✓ Correlation Analysis (Pearson, Spearman, Causation vs Correlation)")
    print("✓ Practical Applications (Student Analysis, Quality Control)")
    print("✓ Practice Exercises and Examples")
    print("\nAll concepts are now ready for real-world application!")

if __name__ == "__main__":
    main() 