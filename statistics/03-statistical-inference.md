# Statistical Inference

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)

Statistical inference allows us to draw conclusions about populations based on sample data. This chapter covers hypothesis testing, confidence intervals, and p-values - essential tools for making data-driven decisions in AI/ML.

## Table of Contents
- [Hypothesis Testing Fundamentals](#hypothesis-testing-fundamentals)
- [One-Sample Tests](#one-sample-tests)
- [Two-Sample Tests](#two-sample-tests)
- [Confidence Intervals](#confidence-intervals)
- [P-Values and Significance](#p-values-and-significance)
- [Multiple Testing](#multiple-testing)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Hypothesis Testing Fundamentals

### Null and Alternative Hypotheses

```python
def hypothesis_testing_example():
    """Example: Testing if a coin is fair"""
    # Null hypothesis: p = 0.5 (fair coin)
    # Alternative hypothesis: p ≠ 0.5 (biased coin)
    
    # Simulate coin flips
    n_flips = 100
    p_true = 0.6  # True probability (biased coin)
    flips = np.random.binomial(1, p_true, n_flips)
    
    # Test statistic: number of heads
    observed_heads = np.sum(flips)
    observed_proportion = observed_heads / n_flips
    
    # Expected under null hypothesis
    expected_heads = n_flips * 0.5
    
    # Z-test statistic
    z_stat = (observed_heads - expected_heads) / np.sqrt(n_flips * 0.5 * 0.5)
    
    # P-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'observed_heads': observed_heads,
        'observed_proportion': observed_proportion,
        'expected_heads': expected_heads,
        'z_statistic': z_stat,
        'p_value': p_value
    }

results = hypothesis_testing_example()
print("Coin Fairness Test")
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# Visualize the test
plt.figure(figsize=(12, 4))

# Observed vs expected
plt.subplot(1, 3, 1)
categories = ['Heads', 'Tails']
observed = [results['observed_heads'], 100 - results['observed_heads']]
expected = [50, 50]
x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, observed, width, label='Observed', alpha=0.7)
plt.bar(x + width/2, expected, width, label='Expected', alpha=0.7)
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Observed vs Expected')
plt.xticks(x, categories)
plt.legend()

# Z-distribution
plt.subplot(1, 3, 2)
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, y, where=(x > abs(results['z_statistic'])) | (x < -abs(results['z_statistic'])), 
                 alpha=0.3, color='red', label='Rejection region')
plt.axvline(results['z_statistic'], color='red', linestyle='--', label=f'z = {results["z_statistic"]:.2f}')
plt.axvline(-results['z_statistic'], color='red', linestyle='--')
plt.title('Z-Distribution')
plt.xlabel('Z-score')
plt.ylabel('Density')
plt.legend()

# P-value interpretation
plt.subplot(1, 3, 3)
alpha = 0.05
decision = "Reject H₀" if results['p_value'] < alpha else "Fail to reject H₀"
plt.text(0.5, 0.5, f"P-value: {results['p_value']:.4f}\nα = {alpha}\nDecision: {decision}", 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.title('Test Decision')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### Type I and Type II Errors

```python
def error_types_demonstration():
    """Demonstrate Type I and Type II errors"""
    # Simulate multiple hypothesis tests
    n_tests = 1000
    alpha = 0.05  # Significance level
    
    # True null hypotheses (no effect)
    true_null = 800
    # True alternative hypotheses (effect exists)
    true_alternative = 200
    
    # Type I errors (false positives)
    type_i_errors = np.random.binomial(true_null, alpha)
    
    # Type II errors (false negatives) - assume 80% power
    power = 0.8
    type_ii_errors = np.random.binomial(true_alternative, 1 - power)
    
    # Calculate rates
    type_i_rate = type_i_errors / true_null
    type_ii_rate = type_ii_errors / true_alternative
    
    return {
        'type_i_errors': type_i_errors,
        'type_ii_errors': type_ii_errors,
        'type_i_rate': type_i_rate,
        'type_ii_rate': type_ii_rate,
        'power': power
    }

error_results = error_types_demonstration()
print("Error Types Demonstration")
for key, value in error_results.items():
    print(f"{key}: {value:.4f}")

# Visualize error matrix
plt.figure(figsize=(10, 4))

# Error matrix
plt.subplot(1, 2, 1)
error_matrix = np.array([
    [800 - error_results['type_i_errors'], error_results['type_i_errors']],
    [error_results['type_ii_errors'], 200 - error_results['type_ii_errors']]
])

sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail to Reject', 'Reject'],
            yticklabels=['H₀ True', 'H₀ False'])
plt.title('Error Matrix')
plt.ylabel('Truth')
plt.xlabel('Decision')

# Error rates over different alpha levels
plt.subplot(1, 2, 2)
alpha_levels = np.linspace(0.01, 0.2, 20)
type_i_rates = alpha_levels
type_ii_rates = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha_levels) - 0.5)  # Simplified

plt.plot(alpha_levels, type_i_rates, 'r-', label='Type I Error Rate', linewidth=2)
plt.plot(alpha_levels, type_ii_rates, 'b-', label='Type II Error Rate', linewidth=2)
plt.axvline(0.05, color='g', linestyle='--', label='α = 0.05')
plt.xlabel('Significance Level (α)')
plt.ylabel('Error Rate')
plt.title('Error Rates vs Significance Level')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## One-Sample Tests

### One-Sample t-Test

```python
def one_sample_t_test_example():
    """Test if sample mean differs from hypothesized value"""
    # Generate sample data
    true_mean = 100
    true_std = 15
    n = 30
    sample = np.random.normal(true_mean, true_std, n)
    
    # Hypothesized mean under null hypothesis
    hypothesized_mean = 95
    
    # Perform t-test
    t_stat, p_value = ttest_1samp(sample, hypothesized_mean)
    
    # Calculate confidence interval
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin_of_error = t_critical * sample_std / np.sqrt(n)
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return {
        'sample_mean': sample_mean,
        'hypothesized_mean': hypothesized_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'sample': sample
    }

t_test_results = one_sample_t_test_example()
print("One-Sample t-Test")
for key, value in t_test_results.items():
    if key != 'sample':
        print(f"{key}: {value}")

# Visualize the test
plt.figure(figsize=(15, 5))

# Sample distribution
plt.subplot(1, 3, 1)
plt.hist(t_test_results['sample'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(t_test_results['sample_mean'], color='red', linestyle='--', linewidth=2, label='Sample Mean')
plt.axvline(t_test_results['hypothesized_mean'], color='green', linestyle='--', linewidth=2, label='Hypothesized Mean')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Sample Distribution')
plt.legend()

# t-distribution
plt.subplot(1, 3, 2)
df = len(t_test_results['sample']) - 1
x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, y, where=(x > abs(t_test_results['t_statistic'])) | (x < -abs(t_test_results['t_statistic'])), 
                 alpha=0.3, color='red', label='Rejection region')
plt.axvline(t_test_results['t_statistic'], color='red', linestyle='--', label=f't = {t_test_results["t_statistic"]:.2f}')
plt.axvline(-t_test_results['t_statistic'], color='red', linestyle='--')
plt.title('t-Distribution')
plt.xlabel('t-score')
plt.ylabel('Density')
plt.legend()

# Confidence interval
plt.subplot(1, 3, 3)
ci_lower, ci_upper = t_test_results['confidence_interval']
plt.errorbar([1], [t_test_results['sample_mean']], 
             yerr=[[t_test_results['sample_mean'] - ci_lower], [ci_upper - t_test_results['sample_mean']]], 
             fmt='o', capsize=5, capthick=2, linewidth=2, label='95% CI')
plt.axhline(t_test_results['hypothesized_mean'], color='green', linestyle='--', label='Hypothesized Mean')
plt.xlim(0.5, 1.5)
plt.ylabel('Mean')
plt.title('Confidence Interval')
plt.legend()
plt.xticks([])

plt.tight_layout()
plt.show()
```

### Chi-Square Goodness-of-Fit Test

```python
def chi_square_goodness_of_fit():
    """Test if observed frequencies match expected frequencies"""
    # Simulate dice rolls
    n_rolls = 600
    true_probabilities = [1/6] * 6  # Fair die
    observed_counts = np.random.multinomial(n_rolls, true_probabilities)
    
    # Expected counts
    expected_counts = [n_rolls * p for p in true_probabilities]
    
    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
    
    return {
        'observed_counts': observed_counts,
        'expected_counts': expected_counts,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': 5
    }

chi2_results = chi_square_goodness_of_fit()
print("Chi-Square Goodness-of-Fit Test")
for key, value in chi2_results.items():
    if key != 'observed_counts' and key != 'expected_counts':
        print(f"{key}: {value:.4f}")

# Visualize
plt.figure(figsize=(12, 4))

# Observed vs expected
plt.subplot(1, 2, 1)
x = np.arange(1, 7)
width = 0.35
plt.bar(x - width/2, chi2_results['observed_counts'], width, label='Observed', alpha=0.7)
plt.bar(x + width/2, chi2_results['expected_counts'], width, label='Expected', alpha=0.7)
plt.xlabel('Die Face')
plt.ylabel('Count')
plt.title('Observed vs Expected Frequencies')
plt.legend()

# Chi-square distribution
plt.subplot(1, 2, 2)
df = chi2_results['degrees_of_freedom']
x = np.linspace(0, 20, 1000)
y = stats.chi2.pdf(x, df)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, y, where=x > chi2_results['chi2_statistic'], 
                 alpha=0.3, color='red', label='Rejection region')
plt.axvline(chi2_results['chi2_statistic'], color='red', linestyle='--', 
           label=f'χ² = {chi2_results["chi2_statistic"]:.2f}')
plt.title('Chi-Square Distribution')
plt.xlabel('Chi-Square Statistic')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
```

## Two-Sample Tests

### Independent t-Test

```python
def independent_t_test_example():
    """Compare means of two independent groups"""
    # Generate two groups
    n1, n2 = 30, 25
    mean1, mean2 = 100, 110
    std1, std2 = 15, 18
    
    group1 = np.random.normal(mean1, std1, n1)
    group2 = np.random.normal(mean2, std2, n2)
    
    # Perform independent t-test
    t_stat, p_value = ttest_ind(group1, group2)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return {
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_std': np.std(group1, ddof=1),
        'group2_std': np.std(group2, ddof=1),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'group1': group1,
        'group2': group2
    }

ind_t_results = independent_t_test_example()
print("Independent t-Test")
for key, value in ind_t_results.items():
    if key not in ['group1', 'group2']:
        print(f"{key}: {value:.4f}")

# Visualize
plt.figure(figsize=(15, 5))

# Box plots
plt.subplot(1, 3, 1)
plt.boxplot([ind_t_results['group1'], ind_t_results['group2']], labels=['Group 1', 'Group 2'])
plt.ylabel('Value')
plt.title('Group Comparison')

# Histograms
plt.subplot(1, 3, 2)
plt.hist(ind_t_results['group1'], alpha=0.7, label='Group 1', bins=15)
plt.hist(ind_t_results['group2'], alpha=0.7, label='Group 2', bins=15)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Group Distributions')
plt.legend()

# Effect size interpretation
plt.subplot(1, 3, 3)
effect_sizes = [0.2, 0.5, 0.8]
interpretations = ['Small', 'Medium', 'Large']
colors = ['lightblue', 'orange', 'red']

for i, (size, interpretation, color) in enumerate(zip(effect_sizes, interpretations, colors)):
    plt.axvline(size, color=color, linestyle='--', alpha=0.7, label=f'{interpretation} effect')
    plt.axvline(-size, color=color, linestyle='--', alpha=0.7)

plt.axvline(ind_t_results['cohens_d'], color='black', linewidth=2, label=f"Cohen's d = {ind_t_results['cohens_d']:.2f}")
plt.xlabel("Cohen's d")
plt.ylabel('Density')
plt.title("Effect Size (Cohen's d)")
plt.legend()
plt.xlim(-2, 2)

plt.tight_layout()
plt.show()
```

### Paired t-Test

```python
def paired_t_test_example():
    """Test for difference in paired observations"""
    # Simulate before/after measurements
    n_pairs = 25
    true_improvement = 5
    measurement_error = 3
    
    # Generate paired data
    before = np.random.normal(100, 15, n_pairs)
    after = before + np.random.normal(true_improvement, measurement_error, n_pairs)
    
    # Calculate differences
    differences = after - before
    
    # Perform paired t-test
    t_stat, p_value = ttest_rel(before, after)
    
    # Alternative: test if differences are different from zero
    t_stat_diff, p_value_diff = ttest_1samp(differences, 0)
    
    return {
        'before_mean': np.mean(before),
        'after_mean': np.mean(after),
        'difference_mean': np.mean(differences),
        't_statistic': t_stat,
        'p_value': p_value,
        't_statistic_diff': t_stat_diff,
        'p_value_diff': p_value_diff,
        'before': before,
        'after': after,
        'differences': differences
    }

paired_results = paired_t_test_example()
print("Paired t-Test")
for key, value in paired_results.items():
    if key not in ['before', 'after', 'differences']:
        print(f"{key}: {value:.4f}")

# Visualize
plt.figure(figsize=(15, 5))

# Before vs After
plt.subplot(1, 3, 1)
plt.scatter(paired_results['before'], paired_results['after'], alpha=0.7)
plt.plot([80, 120], [80, 120], 'r--', alpha=0.7, label='No change')
plt.xlabel('Before')
plt.ylabel('After')
plt.title('Before vs After')
plt.legend()

# Differences distribution
plt.subplot(1, 3, 2)
plt.hist(paired_results['differences'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
plt.axvline(np.mean(paired_results['differences']), color='blue', linestyle='--', linewidth=2, label='Mean difference')
plt.xlabel('Difference (After - Before)')
plt.ylabel('Frequency')
plt.title('Distribution of Differences')
plt.legend()

# Individual changes
plt.subplot(1, 3, 3)
x = np.arange(len(paired_results['before']))
plt.plot([x, x], [paired_results['before'], paired_results['after']], 'b-', alpha=0.5)
plt.scatter(x, paired_results['before'], color='red', label='Before', alpha=0.7)
plt.scatter(x, paired_results['after'], color='green', label='After', alpha=0.7)
plt.xlabel('Subject')
plt.ylabel('Value')
plt.title('Individual Changes')
plt.legend()

plt.tight_layout()
plt.show()
```

## Confidence Intervals

### Confidence Interval Construction

```python
def confidence_interval_demonstration():
    """Demonstrate confidence interval construction"""
    # Generate population
    population = np.random.normal(100, 15, 10000)
    true_mean = np.mean(population)
    
    # Take multiple samples
    n_samples = 50
    sample_size = 30
    confidence_level = 0.95
    
    sample_means = []
    confidence_intervals = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        # Calculate confidence interval
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
        margin_of_error = t_value * sample_std / np.sqrt(sample_size)
        
        sample_means.append(sample_mean)
        confidence_intervals.append((sample_mean - margin_of_error, sample_mean + margin_of_error))
    
    # Count intervals containing true mean
    intervals_containing_true = sum(1 for ci in confidence_intervals 
                                  if ci[0] <= true_mean <= ci[1])
    
    return sample_means, confidence_intervals, true_mean, intervals_containing_true

means, intervals, true_mean, count = confidence_interval_demonstration()

print(f"Confidence Interval Demonstration")
print(f"True population mean: {true_mean:.2f}")
print(f"Intervals containing true mean: {count}/{len(intervals)} ({count/len(intervals)*100:.1f}%)")

# Visualize confidence intervals
plt.figure(figsize=(12, 8))
x_positions = np.arange(len(intervals))

# Plot confidence intervals
for i, (lower, upper) in enumerate(intervals):
    if lower <= true_mean <= upper:
        plt.plot([i, i], [lower, upper], 'b-', alpha=0.7)
    else:
        plt.plot([i, i], [lower, upper], 'r-', alpha=0.7)

plt.axhline(y=true_mean, color='g', linestyle='--', linewidth=2, label='True Mean')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.title('Confidence Intervals (95%)')
plt.legend()
plt.show()
```

## P-Values and Significance

### P-Value Interpretation

```python
def p_value_interpretation():
    """Demonstrate p-value interpretation"""
    # Simulate multiple hypothesis tests
    n_tests = 1000
    alpha = 0.05
    
    # Generate p-values under null hypothesis (uniform distribution)
    p_values_null = np.random.uniform(0, 1, n_tests)
    
    # Generate p-values under alternative hypothesis
    # (some will be small, some large)
    p_values_alt = np.concatenate([
        np.random.beta(0.5, 5, n_tests//2),  # Small p-values
        np.random.uniform(0, 1, n_tests//2)   # Large p-values
    ])
    
    # Mix null and alternative
    p_values = np.concatenate([p_values_null[:800], p_values_alt[:200]])
    
    return p_values, alpha

p_vals, alpha = p_value_interpretation()

# Visualize p-value distribution
plt.figure(figsize=(15, 5))

# P-value histogram
plt.subplot(1, 3, 1)
plt.hist(p_vals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
plt.xlabel('P-value')
plt.ylabel('Frequency')
plt.title('P-Value Distribution')
plt.legend()

# P-value vs significance
plt.subplot(1, 3, 2)
significant = p_vals < alpha
plt.scatter(range(len(p_vals)), p_vals, c=significant, cmap='RdYlBu', alpha=0.7)
plt.axhline(alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
plt.xlabel('Test Number')
plt.ylabel('P-value')
plt.title('P-Values vs Significance')
plt.legend()

# Significance rate
plt.subplot(1, 3, 3)
significance_rate = np.mean(significant)
plt.bar(['Significant', 'Not Significant'], 
        [significance_rate, 1-significance_rate], 
        color=['red', 'blue'], alpha=0.7)
plt.ylabel('Proportion')
plt.title(f'Significance Rate: {significance_rate:.3f}')

plt.tight_layout()
plt.show()

print(f"P-Value Analysis:")
print(f"Total tests: {len(p_vals)}")
print(f"Significant tests: {np.sum(significant)}")
print(f"Significance rate: {np.mean(significant):.3f}")
print(f"Expected under null: {alpha:.3f}")
```

## Multiple Testing

### Multiple Testing Correction

```python
def multiple_testing_correction():
    """Demonstrate multiple testing corrections"""
    # Simulate multiple hypothesis tests
    n_tests = 1000
    alpha = 0.05
    
    # Generate p-values (mostly null, some alternative)
    p_values = np.concatenate([
        np.random.uniform(0, 1, 900),  # Null hypotheses
        np.random.beta(0.5, 5, 100)    # Alternative hypotheses
    ])
    
    # No correction
    significant_uncorrected = p_values < alpha
    
    # Bonferroni correction
    alpha_bonferroni = alpha / n_tests
    significant_bonferroni = p_values < alpha_bonferroni
    
    # Benjamini-Hochberg (FDR) correction
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Calculate critical values
    critical_values = alpha * np.arange(1, n_tests + 1) / n_tests
    
    # Find largest k where p_k <= critical_k
    bh_significant = np.zeros(n_tests, dtype=bool)
    for i in range(n_tests):
        if sorted_p_values[i] <= critical_values[i]:
            bh_significant[sorted_indices[i]] = True
    
    return {
        'p_values': p_values,
        'significant_uncorrected': significant_uncorrected,
        'significant_bonferroni': significant_bonferroni,
        'significant_bh': bh_significant,
        'alpha': alpha,
        'alpha_bonferroni': alpha_bonferroni
    }

mt_results = multiple_testing_correction()

print("Multiple Testing Correction")
print(f"Uncorrected significant: {np.sum(mt_results['significant_uncorrected'])}")
print(f"Bonferroni significant: {np.sum(mt_results['significant_bonferroni'])}")
print(f"Benjamini-Hochberg significant: {np.sum(mt_results['significant_bh'])}")

# Visualize
plt.figure(figsize=(15, 5))

# P-value distribution
plt.subplot(1, 3, 1)
plt.hist(mt_results['p_values'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mt_results['alpha'], color='red', linestyle='--', label=f'α = {mt_results["alpha"]}')
plt.axvline(mt_results['alpha_bonferroni'], color='orange', linestyle='--', label=f'Bonferroni α = {mt_results["alpha_bonferroni"]:.6f}')
plt.xlabel('P-value')
plt.ylabel('Frequency')
plt.title('P-Value Distribution')
plt.legend()

# Comparison of methods
plt.subplot(1, 3, 2)
methods = ['Uncorrected', 'Bonferroni', 'BH']
counts = [np.sum(mt_results['significant_uncorrected']),
          np.sum(mt_results['significant_bonferroni']),
          np.sum(mt_results['significant_bh'])]
colors = ['red', 'orange', 'green']

plt.bar(methods, counts, color=colors, alpha=0.7)
plt.ylabel('Number of Significant Tests')
plt.title('Significant Tests by Method')

# P-value vs rank plot (for BH)
plt.subplot(1, 3, 3)
sorted_p = np.sort(mt_results['p_values'])
ranks = np.arange(1, len(sorted_p) + 1)
critical_line = mt_results['alpha'] * ranks / len(sorted_p)

plt.plot(ranks, sorted_p, 'b.', alpha=0.7, label='P-values')
plt.plot(ranks, critical_line, 'r-', linewidth=2, label='Critical line')
plt.xlabel('Rank')
plt.ylabel('P-value')
plt.title('Benjamini-Hochberg Procedure')
plt.legend()

plt.tight_layout()
plt.show()
```

## Practical Applications

### A/B Testing Example

```python
def ab_testing_example():
    """Simulate A/B testing scenario"""
    # Simulate conversion rates
    n_a, n_b = 1000, 1000
    true_rate_a = 0.10  # 10% conversion rate
    true_rate_b = 0.12  # 12% conversion rate (improvement)
    
    # Generate data
    conversions_a = np.random.binomial(n_a, true_rate_a)
    conversions_b = np.random.binomial(n_b, true_rate_b)
    
    # Calculate conversion rates
    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b
    
    # Perform z-test for proportions
    pooled_rate = (conversions_a + conversions_b) / (n_a + n_b)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_a + 1/n_b))
    z_stat = (rate_a - rate_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Calculate confidence interval for difference
    se_diff = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    z_critical = stats.norm.ppf(0.975)
    margin_of_error = z_critical * se_diff
    ci_lower = (rate_a - rate_b) - margin_of_error
    ci_upper = (rate_a - rate_b) + margin_of_error
    
    return {
        'rate_a': rate_a,
        'rate_b': rate_b,
        'difference': rate_b - rate_a,
        'z_statistic': z_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'conversions_a': conversions_a,
        'conversions_b': conversions_b
    }

ab_results = ab_testing_example()
print("A/B Testing Results")
for key, value in ab_results.items():
    if key not in ['conversions_a', 'conversions_b']:
        print(f"{key}: {value:.4f}")

# Visualize A/B test
plt.figure(figsize=(15, 5))

# Conversion rates
plt.subplot(1, 3, 1)
rates = [ab_results['rate_a'], ab_results['rate_b']]
plt.bar(['Version A', 'Version B'], rates, color=['red', 'blue'], alpha=0.7)
plt.ylabel('Conversion Rate')
plt.title('Conversion Rates')
for i, rate in enumerate(rates):
    plt.text(i, rate + 0.001, f'{rate:.3f}', ha='center', va='bottom')

# Confidence interval for difference
plt.subplot(1, 3, 2)
ci_lower, ci_upper = ab_results['confidence_interval']
plt.errorbar([1], [ab_results['difference']], 
             yerr=[[ab_results['difference'] - ci_lower], [ci_upper - ab_results['difference']]], 
             fmt='o', capsize=5, capthick=2, linewidth=2)
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.ylabel('Difference (B - A)')
plt.title('Confidence Interval for Difference')
plt.xticks([])

# P-value interpretation
plt.subplot(1, 3, 3)
alpha = 0.05
decision = "Reject H₀" if ab_results['p_value'] < alpha else "Fail to reject H₀"
plt.text(0.5, 0.5, f"P-value: {ab_results['p_value']:.4f}\nα = {alpha}\nDecision: {decision}", 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.title('Test Decision')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## Practice Problems

1. **Hypothesis Testing**: Create a function that performs various hypothesis tests and reports results in a standardized format.

2. **Power Analysis**: Implement power analysis to determine required sample sizes for different effect sizes.

3. **Multiple Testing**: Build a function that applies different multiple testing corrections and compares their results.

4. **Effect Size**: Calculate and interpret different effect size measures (Cohen's d, eta-squared, etc.).

## Further Reading

- "Statistical Inference" by George Casella and Roger L. Berger
- "The Practice of Statistics" by David S. Moore
- "Statistics in Plain English" by Timothy C. Urdan
- "Multiple Testing Procedures" by Jason Hsu

## Key Takeaways

- **Hypothesis testing** provides a framework for making decisions about population parameters
- **P-values** measure evidence against the null hypothesis, not probability of hypothesis being true
- **Confidence intervals** provide a range of plausible values for population parameters
- **Multiple testing** requires correction to control false positive rates
- **Effect sizes** complement p-values by measuring practical significance
- **Type I and Type II errors** are fundamental concepts in statistical decision making

In the next chapter, we'll explore regression analysis, including linear and multiple regression techniques. 