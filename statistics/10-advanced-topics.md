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

Non-parametric tests do not assume a specific data distribution. They are especially useful when data are ordinal, not normally distributed, or when sample sizes are small. These methods rely on the ranks or order of the data rather than the actual values.

### 1.1 Wilcoxon Rank-Sum Test (Mann-Whitney U)

**Mathematical Concept:**
The Mann-Whitney U test compares two independent samples to assess whether their population distributions differ. It tests the null hypothesis that the probability of an observation from one group exceeding an observation from the other group is 0.5.

- **Null Hypothesis (H₀):** The distributions of both groups are equal.
- **Alternative Hypothesis (H₁):** The distributions are not equal.

**Test Statistic:**
1. Combine all observations and rank them.
2. Calculate the sum of ranks for each group.
3. Compute the U statistic for each group:
   $$ U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 $$
   $$ U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2 $$
   Where $R_1$ and $R_2$ are the sums of ranks for groups 1 and 2, and $n_1$, $n_2$ are their sizes.
4. The smaller U is used for significance testing.

**Python Implementation:**
```python
# Compare two independent samples
np.random.seed(42)
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)
stat, p = stats.mannwhitneyu(group1, group2)
print(f"Mann-Whitney U: stat={stat:.2f}, p-value={p:.4f}")

# Manual calculation of ranks and U statistic
from scipy.stats import rankdata
data = np.concatenate([group1, group2])
ranks = rankdata(data)
R1 = np.sum(ranks[:30])
R2 = np.sum(ranks[30:])
U1 = 30*30 + 30*31/2 - R1
U2 = 30*30 + 30*31/2 - R2
print(f"Manual U1: {U1:.2f}, U2: {U2:.2f}")
```

### 1.2 Kruskal-Wallis Test

**Mathematical Concept:**
The Kruskal-Wallis test generalizes the Mann-Whitney U test to more than two groups. It tests whether samples originate from the same distribution.

- **Null Hypothesis (H₀):** All groups have the same distribution.
- **Alternative Hypothesis (H₁):** At least one group differs.

**Test Statistic:**
1. Combine all data and rank them.
2. Calculate the sum of ranks $R_j$ for each group $j$.
3. Compute:
   $$ H = \frac{12}{N(N+1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N+1) $$
   Where $N$ is the total number of observations, $n_j$ is the size of group $j$.
4. Under H₀, $H$ follows a chi-square distribution with $k-1$ degrees of freedom.

**Python Implementation:**
```python
# Compare more than two groups
sample1 = np.random.normal(0, 1, 30)
sample2 = np.random.normal(0.5, 1, 30)
sample3 = np.random.normal(1, 1, 30)
stat, p = stats.kruskal(sample1, sample2, sample3)
print(f"Kruskal-Wallis: stat={stat:.2f}, p-value={p:.4f}")
```

### 1.3 Permutation Test

**Mathematical Concept:**
A permutation test assesses the significance of an observed statistic by comparing it to the distribution of statistics computed from data with randomly permuted group labels.

- **Null Hypothesis (H₀):** The group labels are exchangeable (no effect).
- **Alternative Hypothesis (H₁):** The group labels are not exchangeable (effect exists).

**Python Implementation:**
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

**Interpretation:**
- Non-parametric tests are robust to outliers and non-normality, but may have less power than parametric tests when assumptions are met.
- Always visualize your data and check assumptions before choosing a test.

---

## 2. Survival Analysis

Survival analysis models time-to-event data, which is common in medical studies, reliability engineering, and customer churn analysis. The key feature is handling censored data—cases where the event of interest has not occurred for some subjects during the observation period.

### 2.1 Kaplan-Meier Estimator

**Mathematical Concept:**
The Kaplan-Meier estimator is a non-parametric statistic used to estimate the survival function $S(t)$ from lifetime data.

- **Survival Function:**
  $$ S(t) = P(T > t) $$
  where $T$ is the time until the event (e.g., death, failure).

- **Estimator:**
  $$ \hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right) $$
  where $t_i$ are event times, $d_i$ is the number of events at $t_i$, and $n_i$ is the number at risk just before $t_i$.

- **Censoring:**
  Censored data are those for which the event has not occurred by the end of the study or loss to follow-up. The Kaplan-Meier estimator properly accounts for right-censored data.

**Python Implementation:**
```python
from lifelines import KaplanMeierFitter
np.random.seed(42)
times = np.random.exponential(10, 100)
event_observed = np.random.binomial(1, 0.8, 100)  # 80% events, 20% censored
kmf = KaplanMeierFitter()
kmf.fit(times, event_observed)

# Plot the survival function
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()

# Print survival probabilities at specific times
time_points = [5, 10, 15]
for t in time_points:
    print(f"Estimated survival at t={t}: {kmf.survival_function_at_times(t).values[0]:.3f}")
```

### 2.2 Cox Proportional Hazards Model

**Mathematical Concept:**
The Cox model is a semi-parametric regression model for the hazard function:

- **Hazard Function:**
  $$ h(t|X) = h_0(t) \exp(\beta^T X) $$
  where $h_0(t)$ is the baseline hazard, $X$ are covariates, and $\beta$ are coefficients.

- **Interpretation:**
  The exponentiated coefficients $\exp(\beta_j)$ represent the hazard ratio for a one-unit increase in $X_j$.

- **Partial Likelihood:**
  The Cox model estimates $\beta$ by maximizing the partial likelihood, which does not require specifying $h_0(t)$.

**Python Implementation:**
```python
from lifelines import CoxPHFitter
df = pd.DataFrame({'time': times, 'event': event_observed, 'age': np.random.randint(30, 70, 100)})
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
cph.print_summary()  # Shows coefficients, hazard ratios, and p-values

# Predict survival for a new subject
new_subject = pd.DataFrame({'age': [50]})
pred_surv = cph.predict_survival_function(new_subject, times=np.arange(0, 30, 1))
pred_surv.plot()
plt.title('Predicted Survival Curve for Age 50')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()
```

**Interpretation:**
- The Kaplan-Meier estimator provides a non-parametric estimate of survival over time, useful for visualizing and comparing groups.
- The Cox model allows for covariate adjustment and quantifies the effect of predictors on the hazard rate.
- Always check the proportional hazards assumption when using the Cox model (see lifelines documentation for diagnostics).

---

## 3. Mixed Models (Hierarchical/Random Effects)

Mixed models, also known as hierarchical or multilevel models, are used when data are grouped or clustered (e.g., repeated measures, students within schools). They account for both fixed effects (parameters associated with the entire population) and random effects (parameters that vary by group or cluster).

### Mathematical Concept
- **Model Structure:**
  $$ y_{ij} = \beta_0 + u_j + \epsilon_{ij} $$
  where:
  - $y_{ij}$: observation $i$ in group $j$
  - $\beta_0$: overall intercept (fixed effect)
  - $u_j \sim N(0, \sigma_u^2)$: random effect for group $j$
  - $\epsilon_{ij} \sim N(0, \sigma^2)$: residual error

- **Interpretation:**
  - Fixed effects estimate the average relationship across all groups.
  - Random effects capture group-specific deviations from the average.

- **Variance Components:**
  - $\sigma_u^2$: variance between groups
  - $\sigma^2$: variance within groups

- **Intraclass Correlation Coefficient (ICC):**
  $$ ICC = \frac{\sigma_u^2}{\sigma_u^2 + \sigma^2} $$
  Measures the proportion of total variance attributable to group-level differences.

### Python Implementation
```python
# Simulate data for 10 groups, 10 observations each
np.random.seed(42)
groups = np.repeat(np.arange(10), 10)
# Random effect for each group
group_effects = np.random.normal(0, 2, 10)
values = 2 * groups + group_effects[groups] + np.random.normal(0, 2, 100)
df = pd.DataFrame({'group': groups, 'value': values})

# Fit mixed model: value ~ 1 + (1 | group)
from statsmodels.formula.api import mixedlm
model = mixedlm('value ~ 1', df, groups=df['group'])
result = model.fit()
print(result.summary())

# Extract variance components
var_group = result.cov_re.iloc[0, 0]
var_resid = result.scale
icc = var_group / (var_group + var_resid)
print(f"Intraclass Correlation Coefficient (ICC): {icc:.3f}")
```

**Interpretation:**
- The fixed effect (Intercept) estimates the overall mean.
- The random effect variance shows how much groups differ from each other.
- ICC quantifies the degree of similarity within groups.
- Mixed models are essential for repeated measures, longitudinal data, and nested data structures in ML/AI experiments.

---

## 4. Causal Inference

Causal inference aims to estimate the effect of an intervention or treatment, going beyond correlation to answer "what if" questions. This is crucial in AI/ML for understanding the impact of features, policies, or actions.

### 4.1 Propensity Score Matching (PSM)

**Mathematical Concept:**
- The propensity score $e(X)$ is the probability of receiving treatment given covariates $X$:
  $$ e(X) = P(T=1|X) $$
- Matching treated and control units with similar propensity scores simulates a randomized experiment, reducing confounding bias.

**Steps:**
1. Estimate propensity scores (e.g., logistic regression).
2. Match treated and control units with similar scores.
3. Compare outcomes between matched groups to estimate the average treatment effect (ATE).

**Python Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Simulate data
np.random.seed(42)
N = 200
X = np.random.normal(0, 1, (N, 2))
T = np.random.binomial(1, 1/(1 + np.exp(-X[:,0] + 0.5*X[:,1])))
Y = 2*T + X[:,0] + np.random.normal(0, 1, N)

# Estimate propensity scores
model = LogisticRegression()
model.fit(X, T)
propensity = model.predict_proba(X)[:,1]

# Match each treated unit to nearest control by propensity score
treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]
matcher = NearestNeighbors(n_neighbors=1).fit(propensity[control_idx].reshape(-1,1))
_, indices = matcher.kneighbors(propensity[treated_idx].reshape(-1,1))
matched_controls = control_idx[indices.flatten()]

# Estimate ATE
ate = np.mean(Y[treated_idx] - Y[matched_controls])
print(f"Estimated ATE by PSM: {ate:.3f}")
```

### 4.2 Instrumental Variables (IV)

**Mathematical Concept:**
- IV methods address unmeasured confounding by using an instrument $Z$ that affects treatment $T$ but not the outcome $Y$ directly (except through $T$).
- The classic two-stage least squares (2SLS) procedure:
  1. Regress $T$ on $Z$ to get predicted treatment $\hat{T}$.
  2. Regress $Y$ on $\hat{T}$ to estimate the causal effect.

**Assumptions:**
- Relevance: $Z$ is correlated with $T$.
- Exclusion: $Z$ affects $Y$ only through $T$.
- Independence: $Z$ is independent of unmeasured confounders.

**Python Implementation:**
```python
import statsmodels.api as sm
# Simulate data
np.random.seed(42)
N = 200
Z = np.random.binomial(1, 0.5, N)
T = 0.8*Z + np.random.normal(0, 1, N)
Y = 2*T + np.random.normal(0, 1, N)

# First stage: T ~ Z
T_hat = sm.OLS(T, sm.add_constant(Z)).fit().predict(sm.add_constant(Z))
# Second stage: Y ~ T_hat
iv_result = sm.OLS(Y, sm.add_constant(T_hat)).fit()
print(iv_result.summary())
```

### 4.3 Example: Causal Impact (with DoWhy)

**DoWhy** is a Python library for causal inference that provides a unified interface for modeling, identification, estimation, and refutation of causal effects.

**Python Implementation:**
```python
# Install: pip install dowhy
import dowhy
# See DoWhy documentation for detailed examples:
# https://microsoft.github.io/dowhy/
```

**Interpretation:**
- Causal inference methods help estimate the effect of interventions, policies, or features in observational data.
- Always check assumptions (e.g., no unmeasured confounding for PSM, valid instrument for IV).
- Use graphical models (causal diagrams) to clarify assumptions and guide analysis.

---

## 5. Robust Statistics

Robust statistics are designed to be less sensitive to outliers and violations of model assumptions (such as normality). They provide more reliable estimates when data contain anomalies or are not well-behaved.

### 5.1 Median and Median Absolute Deviation (MAD)

**Mathematical Concept:**
- **Median:** The value separating the higher half from the lower half of a data sample. It is a robust measure of central tendency.
- **Median Absolute Deviation (MAD):**
  $$ \text{MAD} = \text{median}(|x_i - \text{median}(x)|) $$
  MAD is a robust measure of statistical dispersion.
- **Scaling:** For normal data, $\text{MAD} \times 1.4826$ estimates the standard deviation.

**Python Implementation:**
```python
# Median and Median Absolute Deviation
x = np.random.normal(0, 1, 100)
x[::10] += 10  # Add outliers
median = np.median(x)
mad = stats.median_abs_deviation(x)
print(f"Median: {median:.2f}, MAD: {mad:.2f}")

# Compare to mean and standard deviation
mean = np.mean(x)
std = np.std(x)
print(f"Mean: {mean:.2f}, Std: {std:.2f}")
```

### 5.2 Robust Regression

**Mathematical Concept:**
- **Ordinary Least Squares (OLS):** Minimizes the sum of squared residuals, sensitive to outliers.
- **Robust Regression (e.g., M-estimators):** Minimizes a function less sensitive to large residuals (e.g., Huber loss).
- **Huber Loss:**
  $$ L(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \leq \epsilon \\ \epsilon(|\delta| - \frac{1}{2}\epsilon) & \text{if } |\delta| > \epsilon \end{cases} $$

**Python Implementation:**
```python
import statsmodels.api as sm
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + np.random.normal(0, 1, 100)
y[::10] += 10  # Add outliers
X = sm.add_constant(X)

# OLS regression
ols_model = sm.OLS(y, X).fit()
print("OLS coefficients:", ols_model.params)

# Robust regression (Huber loss)
robust_model = sm.RLM(y, X)
robust_results = robust_model.fit()
print("Robust coefficients:", robust_results.params)

# Compare fits visually
import matplotlib.pyplot as plt
plt.scatter(X[:,1], y, alpha=0.6, label='Data with outliers')
plt.plot(X[:,1], ols_model.predict(X), color='red', label='OLS fit')
plt.plot(X[:,1], robust_results.predict(X), color='green', label='Robust fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('OLS vs Robust Regression')
plt.show()
```

**Interpretation:**
- The median and MAD provide robust alternatives to the mean and standard deviation.
- Robust regression methods (like RLM) yield more reliable parameter estimates in the presence of outliers.
- Always compare robust and classical methods to assess the influence of outliers in your data.

---

## 6. Practice Problems

1. **Non-Parametric Testing:**
   - Simulate two groups of data, one normal and one skewed. Use the Mann-Whitney U test to compare their medians. Interpret the result and visualize the distributions.
   - For three or more groups with different variances, apply the Kruskal-Wallis test. Discuss when you would prefer this over ANOVA.

2. **Survival Analysis:**
   - Generate synthetic time-to-event data with censoring. Fit a Kaplan-Meier curve and interpret the survival probability at a given time. Compare survival between two groups (e.g., treatment vs. control) using the log-rank test.
   - Fit a Cox proportional hazards model with at least one covariate. Interpret the hazard ratio and check the proportional hazards assumption (e.g., using Schoenfeld residuals in lifelines).

3. **Mixed Models:**
   - Simulate repeated measures data (e.g., students tested multiple times). Fit a mixed-effects model and interpret both fixed and random effects. Calculate and interpret the intraclass correlation coefficient (ICC).
   - Compare the results of a mixed model to a standard linear regression. Discuss the consequences of ignoring group structure.

4. **Causal Inference:**
   - Simulate observational data with a confounder. Estimate the average treatment effect (ATE) using propensity score matching. Visualize the distribution of propensity scores before and after matching.
   - Use an instrumental variable approach to estimate a causal effect in the presence of unmeasured confounding. Clearly state and justify the instrument's validity.

5. **Robust Statistics:**
   - Create a dataset with outliers. Compare the mean and standard deviation to the median and MAD. Discuss which is more appropriate and why.
   - Fit both OLS and robust regression models to data with outliers. Visualize and interpret the differences in fitted lines and coefficients.

**Guidance:**
- For each problem, write out the mathematical formulation, perform the analysis in Python, and interpret the results in the context of real-world data science or AI/ML applications.
- Consider edge cases (e.g., small sample sizes, high censoring, strong confounding, extreme outliers) and discuss how robust or advanced methods address them.

## 7. Further Reading

- **"Survival Analysis: A Self-Learning Text" by Kleinbaum & Klein**  
  A comprehensive and accessible introduction to survival analysis, including practical examples and exercises.

- **"Causal Inference in Statistics: A Primer" by Pearl, Glymour, & Jewell**  
  An essential guide to the logic and mathematics of causal inference, with clear explanations and real-world examples.

- **"Robust Statistics" by Huber & Ronchetti**  
  The definitive reference on robust statistical methods, covering theory, algorithms, and applications.

- **DoWhy Documentation** ([https://microsoft.github.io/dowhy/](https://microsoft.github.io/dowhy/))  
  Official documentation for the DoWhy Python library, with tutorials and case studies on causal inference in Python.

- **lifelines Documentation** ([https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/))  
  User guide and API reference for the lifelines library, including advanced survival analysis techniques and diagnostics.

- **"Applied Longitudinal Analysis" by Fitzmaurice, Laird, & Ware**  
  A practical resource for mixed models and longitudinal data analysis, with a focus on applications in health and social sciences.

- **"The Book of Why" by Judea Pearl**  
  An accessible and thought-provoking exploration of causality, counterfactuals, and the future of AI.

- **"Modern Applied Statistics with S" by Venables & Ripley**  
  A classic text covering a wide range of advanced statistical methods, with practical code examples (much of which translates to Python).

**Tip:**
- When exploring advanced topics, always consult both theoretical and applied resources. Try to implement methods from scratch in Python to deepen your understanding.

---

## 8. Key Takeaways
- Non-parametric methods are powerful for non-normal or ordinal data
- Survival analysis is essential for time-to-event data
- Mixed models handle hierarchical and repeated measures data
- Causal inference requires careful design and specialized methods
- Robust statistics protect against outliers and model violations

---

**Congratulations!** You have completed the comprehensive statistics guide for AI/ML and data science. 