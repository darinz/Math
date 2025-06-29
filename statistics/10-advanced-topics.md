# Advanced Topics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Lifelines](https://img.shields.io/badge/Lifelines-0.27+-orange.svg)](https://lifelines.readthedocs.io/)

# Chapter 10: Advanced Topics

## Overview

This chapter explores advanced statistical methods that are essential for specialized data science and AI/ML applications. Topics include non-parametric methods, survival analysis, mixed models, causal inference, robust statistics, and their practical uses.

## Learning Objectives
- Apply non-parametric statistical tests
- Analyze time-to-event data with survival analysis
- Understand and use mixed-effects models
- Grasp causal inference techniques
- Use robust statistics for outlier-prone data

## Prerequisites
- Familiarity with hypothesis testing, regression, and Python libraries (scipy, statsmodels, lifelines)

## Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from lifelines import KaplanMeierFitter, CoxPHFitter
sns.set(style="whitegrid")
```

---

## 1. Non-Parametric Methods

Non-parametric tests do not assume a specific data distribution.

### 1.1 Wilcoxon Rank-Sum Test (Mann-Whitney U)
```python
# Compare two independent samples
np.random.seed(42)
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)
stat, p = stats.mannwhitneyu(group1, group2)
print(f"Mann-Whitney U: stat={stat:.2f}, p-value={p:.4f}")
```

### 1.2 Kruskal-Wallis Test
```python
# Compare more than two groups
sample1 = np.random.normal(0, 1, 30)
sample2 = np.random.normal(0.5, 1, 30)
sample3 = np.random.normal(1, 1, 30)
stat, p = stats.kruskal(sample1, sample2, sample3)
print(f"Kruskal-Wallis: stat={stat:.2f}, p-value={p:.4f}")
```

### 1.3 Permutation Test
```python
def permutation_test(x, y, n_permutations=10000):
    observed = np.abs(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_x = combined[:len(x)]
        new_y = combined[len(x):]
        if np.abs(np.mean(new_x) - np.mean(new_y)) >= observed:
            count += 1
    return count / n_permutations

p_perm = permutation_test(group1, group2)
print(f"Permutation test p-value: {p_perm:.4f}")
```

---

## 2. Survival Analysis

Survival analysis models time-to-event data (e.g., time until failure or death).

### 2.1 Kaplan-Meier Estimator
```python
from lifelines import KaplanMeierFitter
np.random.seed(42)
times = np.random.exponential(10, 100)
event_observed = np.random.binomial(1, 0.8, 100)
kmf = KaplanMeierFitter()
kmf.fit(times, event_observed)
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()
```

### 2.2 Cox Proportional Hazards Model
```python
from lifelines import CoxPHFitter
df = pd.DataFrame({'time': times, 'event': event_observed, 'age': np.random.randint(30, 70, 100)})
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
cph.print_summary()
```

---

## 3. Mixed Models (Hierarchical/Random Effects)

Mixed models account for both fixed and random effects (e.g., repeated measures).

```python
# Simulate data
np.random.seed(42)
groups = np.repeat(np.arange(10), 10)
values = 2 * groups + np.random.normal(0, 2, 100)
df = pd.DataFrame({'group': groups, 'value': values})
# Fit mixed model
model = mixedlm('value ~ 1', df, groups=df['group'])
result = model.fit()
print(result.summary())
```

---

## 4. Causal Inference

Causal inference aims to estimate the effect of an intervention or treatment.

### 4.1 Propensity Score Matching (Concept)
- Estimate the probability of treatment given covariates (propensity score)
- Match treated and control units with similar scores
- Compare outcomes between matched groups

### 4.2 Instrumental Variables (Concept)
- Use an external variable (instrument) that affects treatment but not the outcome directly
- Two-stage least squares (2SLS) is a common method

### 4.3 Example: Causal Impact (with DoWhy)
```python
# Install: pip install dowhy
import dowhy
# See DoWhy documentation for detailed examples:
# https://microsoft.github.io/dowhy/
```

---

## 5. Robust Statistics

Robust statistics are less sensitive to outliers and violations of assumptions.

### 5.1 Median and MAD
```python
# Median and Median Absolute Deviation
x = np.random.normal(0, 1, 100)
x[::10] += 10  # Add outliers
median = np.median(x)
mad = stats.median_abs_deviation(x)
print(f"Median: {median:.2f}, MAD: {mad:.2f}")
```

### 5.2 Robust Regression
```python
import statsmodels.api as sm
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + np.random.normal(0, 1, 100)
y[::10] += 10  # Add outliers
X = sm.add_constant(X)
robust_model = sm.RLM(y, X)
robust_results = robust_model.fit()
print(robust_results.summary())
```

---

## 6. Practice Problems
- Use the Mann-Whitney U test to compare two groups with non-normal data
- Fit a Kaplan-Meier curve and interpret the survival probability at a given time
- Simulate a mixed-effects model and interpret the random effects
- Discuss how you would estimate a causal effect using propensity scores
- Compare OLS and robust regression on data with outliers

---

## 7. Further Reading
- "Survival Analysis: A Self-Learning Text" by Kleinbaum & Klein
- "Causal Inference in Statistics" by Pearl, Glymour, & Jewell
- "Robust Statistics" by Huber & Ronchetti
- DoWhy and lifelines documentation

---

## 8. Key Takeaways
- Non-parametric methods are powerful for non-normal or ordinal data
- Survival analysis is essential for time-to-event data
- Mixed models handle hierarchical and repeated measures data
- Causal inference requires careful design and specialized methods
- Robust statistics protect against outliers and model violations

---

**Congratulations!** You have completed the comprehensive statistics guide for AI/ML and data science. 