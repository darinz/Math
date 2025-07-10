# Advanced Topics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Lifelines](https://img.shields.io/badge/Lifelines-0.27+-orange.svg)](https://lifelines.readthedocs.io/)

## Introduction

This chapter explores advanced statistical methods that are essential for specialized data science and AI/ML applications. Topics include non-parametric methods, survival analysis, mixed models, causal inference, robust statistics, and their practical uses.

### Why Advanced Topics Matter

Advanced statistical methods are crucial for real-world data science because:

1. **Real Data is Messy**: Data often violates classical assumptions
2. **Complex Relationships**: Many problems require sophisticated modeling
3. **Causal Questions**: Correlation isn't always enough
4. **Time-to-Event Data**: Many outcomes are time-based
5. **Hierarchical Structures**: Data often has nested or grouped structure

### Understanding Advanced Methods

Think of advanced statistical methods as specialized tools for complex problems:
- **Non-parametric methods**: When you don't know the data distribution
- **Survival analysis**: When you're interested in "when" something happens
- **Mixed models**: When data has natural groupings or hierarchies
- **Causal inference**: When you need to understand "why" not just "what"
- **Robust statistics**: When your data has outliers or violations

### The Challenge of Real-World Data

Real-world data often presents challenges that basic statistical methods can't handle:

#### Example: Medical Study
- **Problem**: Compare treatment effectiveness
- **Challenge**: Patients drop out, some don't respond to treatment
- **Solution**: Survival analysis with censoring
- **Result**: Reliable estimates despite missing data

#### Example: Educational Research
- **Problem**: Study student performance across schools
- **Challenge**: Students within schools are more similar than across schools
- **Solution**: Mixed models with random effects
- **Result**: Proper accounting for hierarchical structure

## Learning Objectives

By the end of this chapter, you will be able to:
- Apply non-parametric statistical tests
- Analyze time-to-event data with survival analysis
- Understand and use mixed-effects models
- Grasp causal inference techniques
- Use robust statistics for outlier-prone data

## Prerequisites

- Familiarity with hypothesis testing, regression, and Python libraries (scipy, statsmodels, lifelines)
- Understanding of basic probability and statistical concepts
- Experience with data visualization and interpretation

## Required Libraries

The examples in this chapter use specialized Python libraries for advanced statistical analysis, including scipy for non-parametric tests, lifelines for survival analysis, and statsmodels for mixed models and robust statistics.

---

## 1. Non-Parametric Methods

Non-parametric tests do not assume a specific data distribution. They are especially useful when data are ordinal, not normally distributed, or when sample sizes are small. These methods rely on the ranks or order of the data rather than the actual values.

### Understanding Non-Parametric Methods

Non-parametric methods are like using a Swiss Army knife when you don't know what tools you'll need. They make fewer assumptions about your data, making them more flexible but sometimes less powerful than parametric methods.

#### Intuitive Example: Restaurant Ratings

Consider comparing two restaurants:
- **Parametric approach**: Assume ratings follow normal distribution, use t-test
- **Non-parametric approach**: Use ranks, compare median ratings
- **Advantage**: Works regardless of rating distribution
- **Trade-off**: Less powerful if data is actually normal

### Why Use Non-Parametric Methods?

1. **Distribution-free**: No assumptions about data distribution
2. **Robust to outliers**: Based on ranks, not actual values
3. **Ordinal data**: Works with rankings, not just continuous data
4. **Small samples**: Often more reliable with limited data
5. **Non-normal data**: Handles skewed, heavy-tailed distributions

### 1.1 Wilcoxon Rank-Sum Test (Mann-Whitney U)

The Mann-Whitney U test compares two independent samples to assess whether their population distributions differ. It tests the null hypothesis that the probability of an observation from one group exceeding an observation from the other group is 0.5.

#### Mathematical Foundation

**Null Hypothesis (H₀)**: The distributions of both groups are equal.
**Alternative Hypothesis (H₁)**: The distributions are not equal.

**Test Statistic Calculation**:
1. **Combine and rank**: Pool all observations and assign ranks
2. **Sum ranks**: Calculate sum of ranks for each group
3. **Compute U statistic**:
```math
U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
```
```math
U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2
```
   Where $`R_1`$ and $`R_2`$ are the sums of ranks for groups 1 and 2, and $`n_1`$, $`n_2`$ are their sizes.

4. **Test statistic**: Use the smaller U value for significance testing

#### Example: Treatment Comparison

**Scenario**: Compare pain relief scores between two treatments
**Group A**: 10 patients, scores: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
**Group B**: 8 patients, scores: [1, 2, 3, 4, 5, 6, 7, 8]

**Process**:
1. **Combined ranks**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
2. **Group A ranks**: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] → Sum = 65
3. **Group B ranks**: [1, 2, 3, 4, 5, 6, 7, 8] → Sum = 36
4. **U calculation**: $`U_1 = 10 \times 8 + \frac{10 \times 11}{2} - 65 = 80 + 55 - 65 = 70`$
5. **Result**: U = 70, p < 0.05 → Significant difference

#### Properties and Interpretation

- **Distribution-free**: No assumption about underlying distribution
- **Robust**: Insensitive to outliers
- **Power**: Less powerful than t-test when data is normal
- **Effect size**: Can calculate rank correlation or other measures

### 1.2 Kruskal-Wallis Test

The Kruskal-Wallis test generalizes the Mann-Whitney U test to more than two groups. It tests whether samples originate from the same distribution.

#### Mathematical Foundation

**Null Hypothesis (H₀)**: All groups have the same distribution.
**Alternative Hypothesis (H₁)**: At least one group differs.

**Test Statistic**:
1. **Combine and rank**: Pool all data and assign ranks
2. **Sum ranks**: Calculate sum of ranks $`R_j`$ for each group $`j`$
3. **Compute H statistic**:
   ```math
   H = \frac{12}{N(N+1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N+1)
   ```
   Where $`N`$ is the total number of observations, $`n_j`$ is the size of group $`j`$.

4. **Distribution**: Under H₀, $`H`$ follows a chi-square distribution with $`k-1`$ degrees of freedom

#### Example: Multiple Treatment Comparison

**Scenario**: Compare effectiveness of three pain medications
**Group A**: 8 patients, scores: [2, 3, 4, 5, 6, 7, 8, 9]
**Group B**: 7 patients, scores: [1, 2, 3, 4, 5, 6, 7]
**Group C**: 6 patients, scores: [3, 4, 5, 6, 7, 8]

**Process**:
1. **Combined ranks**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
2. **Group sums**: R₁ = 44, R₂ = 28, R₃ = 39
3. **H calculation**: $`H = \frac{12}{21 \times 22}(\frac{44^2}{8} + \frac{28^2}{7} + \frac{39^2}{6}) - 3 \times 22 = 2.34`$
4. **Result**: H = 2.34, p > 0.05 → No significant difference

#### When to Use Kruskal-Wallis

- **Multiple groups**: More than two independent groups
- **Non-normal data**: When ANOVA assumptions are violated
- **Ordinal data**: When data represents rankings or categories
- **Small samples**: When sample sizes are too small for ANOVA

### 1.3 Permutation Test

A permutation test assesses the significance of an observed statistic by comparing it to the distribution of statistics computed from data with randomly permuted group labels.

#### Mathematical Foundation

**Null Hypothesis (H₀)**: The group labels are exchangeable (no effect).
**Alternative Hypothesis (H₁)**: The group labels are not exchangeable (effect exists).

**Algorithm**:
1. **Observed statistic**: Calculate test statistic on original data
2. **Permutations**: Randomly shuffle group labels many times
3. **Null distribution**: Calculate test statistic for each permutation
4. **P-value**: Proportion of permuted statistics as extreme as observed

#### Example: Permutation Test

**Scenario**: Compare mean scores between two groups
**Original data**: Group A = [8, 9, 10, 11], Group B = [4, 5, 6, 7]
**Observed difference**: 4.0

**Permutations**:
- Perm 1: [8, 9, 4, 5] vs [10, 11, 6, 7] → diff = 2.0
- Perm 2: [8, 4, 10, 6] vs [9, 5, 11, 7] → diff = 1.5
- ... (many more permutations)

**Result**: 2/1000 permutations have difference ≥ 4.0 → p = 0.002

#### Advantages of Permutation Tests

- **Exact**: No distributional assumptions
- **Flexible**: Can use any test statistic
- **Intuitive**: Direct interpretation of p-value
- **Robust**: Works with any sample size

#### Python Implementation

```python
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import permutation_test

# Wilcoxon Rank-Sum Test
statistic, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

# Kruskal-Wallis Test
statistic, p_value = kruskal(group_a, group_b, group_c)

# Permutation Test
def statistic(x, y):
    return np.mean(x) - np.mean(y)

result = permutation_test((group_a, group_b), statistic, n_resamples=10000)
```

#### Interpretation Guidelines

- **Non-parametric tests** are robust to outliers and non-normality, but may have less power than parametric tests when assumptions are met
- **Always visualize your data** and check assumptions before choosing a test
- **Consider effect size** in addition to p-values
- **Use multiple tests** when in doubt about assumptions

---

## 2. Survival Analysis

Survival analysis models time-to-event data, which is common in medical studies, reliability engineering, and customer churn analysis. The key feature is handling censored data—cases where the event of interest has not occurred for some subjects during the observation period.

### Understanding Survival Analysis

Survival analysis is like studying "how long until something happens." It's used when you're interested in time-based outcomes, not just whether something happened.

#### Intuitive Example: Customer Retention

Consider a subscription service:
- **Event**: Customer cancels subscription
- **Time**: Days from signup to cancellation
- **Censoring**: Some customers still active at study end
- **Question**: How long do customers typically stay?

### Key Concepts in Survival Analysis

#### 1. Survival Function
The survival function $`S(t)`$ gives the probability of surviving beyond time $`t`$:
```math
S(t) = P(T > t)
```
Where $`T`$ is the time until the event.

#### 2. Hazard Function
The hazard function $`h(t)`$ gives the instantaneous rate of events:
```math
h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}
```

#### 3. Censoring
- **Right-censoring**: Event hasn't occurred by end of study
- **Left-censoring**: Event occurred before study began
- **Interval-censoring**: Event occurred in a time interval

### 2.1 Kaplan-Meier Estimator

The Kaplan-Meier estimator is a non-parametric statistic used to estimate the survival function $`S(t)`$ from lifetime data.

#### Mathematical Foundation

**Survival Function**:
```math
S(t) = P(T > t)
```
Where $`T`$ is the time until the event (e.g., death, failure).

**Kaplan-Meier Estimator**:
```math
\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
```
Where:
- $`t_i`$ are event times
- $`d_i`$ is the number of events at $`t_i`$
- $`n_i`$ is the number at risk just before $`t_i`$

#### Example: Medical Study

**Scenario**: Study of cancer treatment effectiveness
**Data**:
- Patient 1: Event at 6 months
- Patient 2: Censored at 8 months (still alive)
- Patient 3: Event at 12 months
- Patient 4: Event at 15 months
- Patient 5: Censored at 18 months

**Calculation**:
- At 6 months: $`\hat{S}(6) = 1 - \frac{1}{5} = 0.8`$
- At 12 months: $`\hat{S}(12) = 0.8 \times (1 - \frac{1}{3}) = 0.53`$
- At 15 months: $`\hat{S}(15) = 0.53 \times (1 - \frac{1}{2}) = 0.27`$

#### Properties of Kaplan-Meier

- **Non-parametric**: No distributional assumptions
- **Handles censoring**: Properly accounts for incomplete observations
- **Step function**: Constant between event times
- **Confidence intervals**: Can be calculated for survival estimates

### 2.2 Cox Proportional Hazards Model

The Cox model is a semi-parametric regression model for the hazard function that allows for covariate adjustment.

#### Mathematical Foundation

**Cox Model**:
```math
h(t|X) = h_0(t) \exp(\beta^T X)
```
Where:
- $`h_0(t)`$ is the baseline hazard (unspecified)
- $`X`$ are covariates
- $`\beta`$ are coefficients to be estimated

**Interpretation**:
The exponentiated coefficients $`\exp(\beta_j)`$ represent the hazard ratio for a one-unit increase in $`X_j`$.

**Partial Likelihood**:
The Cox model estimates $`\beta`$ by maximizing the partial likelihood, which does not require specifying $`h_0(t)`$.

#### Example: Treatment Effect Analysis

**Scenario**: Compare two cancer treatments
**Covariates**: Treatment (0/1), Age, Gender
**Model**: $`h(t) = h_0(t) \exp(\beta_1 \text{Treatment} + \beta_2 \text{Age} + \beta_3 \text{Gender})`$

**Results**:
- Treatment coefficient: $`\beta_1 = -0.5`$
- Hazard ratio: $`\exp(-0.5) = 0.61`$
- Interpretation: Treatment reduces hazard by 39%

#### Assumptions of Cox Model

1. **Proportional hazards**: Hazard ratios are constant over time
2. **Linear effects**: Log-hazard is linear in covariates
3. **No interactions**: Effects are additive on log-hazard scale

#### Checking Assumptions

- **Schoenfeld residuals**: Test proportional hazards assumption
- **Martingale residuals**: Check linearity and outliers
- **Deviance residuals**: Assess overall model fit

#### Python Implementation

```python
from lifelines import KaplanMeierFitter, CoxPHFitter

# Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)
kmf.plot()

# Cox Model
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')
cph.print_summary()
cph.plot_partial_effects('treatment')
```

#### Interpretation Guidelines

- **Kaplan-Meier** provides non-parametric survival estimates, useful for visualizing and comparing groups
- **Cox model** allows for covariate adjustment and quantifies the effect of predictors on the hazard rate
- **Always check** the proportional hazards assumption when using the Cox model
- **Consider competing risks** when multiple events can occur

---

## 3. Mixed Models (Hierarchical/Random Effects)

Mixed models, also known as hierarchical or multilevel models, are used when data are grouped or clustered (e.g., repeated measures, students within schools). They account for both fixed effects (parameters associated with the entire population) and random effects (parameters that vary by group or cluster).

### Understanding Mixed Models

Mixed models are like recognizing that individuals within groups are more similar to each other than to individuals in other groups. They properly account for this hierarchical structure.

#### Intuitive Example: Classroom Study

Consider studying student performance:
- **Students**: Within each classroom
- **Classrooms**: Within each school
- **Schools**: Within each district
- **Question**: How do we account for this nesting?

### Mathematical Foundation

#### Basic Mixed Model

**Model Structure**:
```math
y_{ij} = \beta_0 + u_j + \epsilon_{ij}
```
Where:
- $`y_{ij}`$: observation $`i`$ in group $`j`$
- $`\beta_0`$: overall intercept (fixed effect)
- $`u_j \sim N(0, \sigma_u^2)`$: random effect for group $`j`$
- $`\epsilon_{ij} \sim N(0, \sigma^2)`$: residual error

#### Interpretation

- **Fixed effects**: Estimate the average relationship across all groups
- **Random effects**: Capture group-specific deviations from the average
- **Variance components**: Quantify the variation at different levels

#### Variance Components

- $`\sigma_u^2`$: variance between groups
- $`\sigma^2`$: variance within groups

#### Intraclass Correlation Coefficient (ICC)

```math
ICC = \frac{\sigma_u^2}{\sigma_u^2 + \sigma^2}
```
Measures the proportion of total variance attributable to group-level differences.

#### Example: Student Performance Study

**Scenario**: Study math scores across 20 classrooms
**Model**: $`\text{Score}_{ij} = \beta_0 + u_j + \epsilon_{ij}`$

**Results**:
- Fixed effect (intercept): $`\beta_0 = 75`$
- Random effect variance: $`\sigma_u^2 = 25`$
- Residual variance: $`\sigma^2 = 100`$
- ICC: $`\frac{25}{25 + 100} = 0.20`$

**Interpretation**: 20% of variance is between classrooms, 80% within classrooms

### Advanced Mixed Models

#### Random Slopes Model

```math
y_{ij} = \beta_0 + \beta_1 x_{ij} + u_{0j} + u_{1j} x_{ij} + \epsilon_{ij}
```
Where:
- $`u_{0j}`$: random intercept for group $`j`$
- $`u_{1j}`$: random slope for group $`j`$

#### Crossed Random Effects

```math
y_{ijk} = \beta_0 + u_j + v_k + \epsilon_{ijk}
```
Where:
- $`u_j`$: random effect for factor 1
- $`v_k`$: random effect for factor 2

#### Example: Longitudinal Study

**Scenario**: Track student growth over time
**Model**: $`\text{Score}_{ij} = \beta_0 + \beta_1 \text{Time}_{ij} + u_{0j} + u_{1j} \text{Time}_{ij} + \epsilon_{ij}`$

**Results**:
- Average growth: $`\beta_1 = 2.5`$ points per month
- Student variation in growth: $`\sigma_{u1}^2 = 0.5`$
- Student variation in starting level: $`\sigma_{u0}^2 = 15`$

### Model Comparison

#### Mixed Model vs. Standard Regression

**Standard Regression**:
- Assumes independence of observations
- Ignores group structure
- Can lead to incorrect standard errors

**Mixed Model**:
- Accounts for group structure
- Provides proper standard errors
- Estimates group-level effects

#### Example: Treatment Effect Study

**Scenario**: Compare treatments across 10 clinics
**Standard regression**: $`\text{Outcome} = \beta_0 + \beta_1 \text{Treatment}`$
**Mixed model**: $`\text{Outcome}_{ij} = \beta_0 + \beta_1 \text{Treatment}_{ij} + u_j + \epsilon_{ij}`$

**Results**:
- Standard regression: Treatment effect = 5.2 (SE = 1.1)
- Mixed model: Treatment effect = 5.2 (SE = 1.8)

**Interpretation**: Mixed model provides more realistic uncertainty estimates

#### Python Implementation

```python
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Simple random intercept model
model = MixedLM(y, X, groups=group_ids)
result = model.fit()
print(result.summary())

# Random slopes model
model = MixedLM(y, X, groups=group_ids, re_formula="~time")
result = model.fit()
```

#### Interpretation Guidelines

- **Fixed effects** estimate average relationships across all groups
- **Random effects** capture group-specific deviations
- **ICC** quantifies the degree of similarity within groups
- **Mixed models** are essential for repeated measures, longitudinal data, and nested data structures
- **Always compare** mixed models to standard regression to understand the impact of ignoring group structure

---

## 4. Causal Inference

Causal inference aims to estimate the effect of an intervention or treatment, going beyond correlation to answer "what if" questions. This is crucial in AI/ML for understanding the impact of features, policies, or actions.

### Understanding Causal Inference

Causal inference is like trying to understand "what would have happened if things were different." It's about moving from correlation to causation.

#### Intuitive Example: Marketing Campaign

Consider a marketing campaign:
- **Correlation**: Customers who received emails spent more
- **Causation**: Did the emails cause increased spending?
- **Challenge**: Maybe high-value customers were more likely to receive emails
- **Solution**: Causal inference methods

### The Fundamental Problem

The fundamental problem of causal inference is that we can never observe the same unit under both treatment and control conditions simultaneously.

#### Potential Outcomes Framework

For each unit $`i`$:
- $`Y_i(1)`$: Outcome if treated
- $`Y_i(0)`$: Outcome if not treated
- **Observed outcome**: $`Y_i = T_i Y_i(1) + (1-T_i) Y_i(0)`$
- **Causal effect**: $`\tau_i = Y_i(1) - Y_i(0)`$

#### Average Treatment Effect (ATE)

```math
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

### 4.1 Propensity Score Matching (PSM)

Propensity score matching simulates a randomized experiment by matching treated and control units with similar characteristics.

#### Mathematical Foundation

**Propensity Score**:
The propensity score $`e(X)`$ is the probability of receiving treatment given covariates $`X`$:
```math
e(X) = P(T=1|X)
```

**Balancing Property**:
If $`X`$ is balanced (no confounding), then:
```math
T \perp X | e(X)
```

**Matching Process**:
1. Estimate propensity scores (e.g., logistic regression)
2. Match treated and control units with similar scores
3. Compare outcomes between matched groups

#### Example: Job Training Program

**Scenario**: Evaluate job training program effectiveness
**Covariates**: Age, education, previous income, gender
**Treatment**: Job training program participation

**Process**:
1. **Estimate propensity scores**: $`P(\text{Treatment} = 1 | \text{Age}, \text{Education}, \text{Income}, \text{Gender})`$
2. **Match participants**: Find control individuals with similar propensity scores
3. **Compare outcomes**: Average income difference between matched groups

**Results**:
- **Before matching**: Treated group earned $2,000 more (confounded)
- **After matching**: Treated group earned $500 more (causal estimate)

#### Matching Methods

1. **Nearest neighbor**: Match to closest propensity score
2. **Caliper matching**: Match within specified distance
3. **Stratification**: Group by propensity score quintiles
4. **Inverse probability weighting**: Weight by inverse of propensity score

### 4.2 Instrumental Variables (IV)

Instrumental variables address unmeasured confounding by using an instrument $`Z`$ that affects treatment $`T`$ but not the outcome $`Y`$ directly.

#### Mathematical Foundation

**IV Assumptions**:
1. **Relevance**: $`Z`$ is correlated with $`T`$
2. **Exclusion**: $`Z`$ affects $`Y`$ only through $`T`$
3. **Independence**: $`Z`$ is independent of unmeasured confounders

**Two-Stage Least Squares (2SLS)**:
1. **First stage**: Regress $`T`$ on $`Z`$ to get predicted treatment $`\hat{T}`$
2. **Second stage**: Regress $`Y`$ on $`\hat{T}`$ to estimate causal effect

#### Example: Education and Income

**Scenario**: Study effect of education on income
**Challenge**: Ability affects both education and income (confounding)
**Instrument**: Distance to college (affects education but not income directly)

**Process**:
1. **First stage**: $`\text{Education} = \alpha_0 + \alpha_1 \text{Distance} + \text{Controls} + \epsilon`$
2. **Second stage**: $`\text{Income} = \beta_0 + \beta_1 \hat{\text{Education}} + \text{Controls} + \epsilon`$

**Results**:
- **OLS estimate**: 0.08 (confounded by ability)
- **IV estimate**: 0.12 (causal effect of education)

### 4.3 Example: Causal Impact (with DoWhy)

DoWhy is a Python library for causal inference that provides a unified interface for modeling, identification, estimation, and refutation of causal effects.

#### Causal Graph

```python
# Define causal graph
causal_graph = """
digraph {
    U[label="Unmeasured Confounder"];
    T[label="Treatment"];
    Y[label="Outcome"];
    U -> T;
    U -> Y;
    T -> Y;
}
"""
```

#### Identification and Estimation

```python
from dowhy import CausalModel

# Create causal model
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    graph=causal_graph
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate causal effect
estimate = model.estimate_effect(identified_estimand)

# Refute results
refutation_results = model.refute_estimate(identified_estimand, estimate)
```

#### Example: Marketing Campaign Analysis

**Scenario**: Evaluate email campaign effectiveness
**Variables**: Email sent (T), Purchase (Y), Customer value (U)
**Challenge**: High-value customers more likely to receive emails

**Results**:
- **Naive estimate**: 15% increase in purchases
- **Causal estimate**: 8% increase in purchases
- **Confidence interval**: [5%, 11%]

#### Interpretation Guidelines

- **Causal inference methods** help estimate the effect of interventions, policies, or features in observational data
- **Always check assumptions** (e.g., no unmeasured confounding for PSM, valid instrument for IV)
- **Use graphical models** (causal diagrams) to clarify assumptions and guide analysis
- **Consider multiple methods** and triangulate results
- **Be transparent** about assumptions and limitations

---

## 5. Robust Statistics

Robust statistics are designed to be less sensitive to outliers and violations of model assumptions (such as normality). They provide more reliable estimates when data contain anomalies or are not well-behaved.

### Understanding Robust Statistics

Robust statistics are like having a backup plan when your data doesn't behave as expected. They provide reliable estimates even when some assumptions are violated.

#### Intuitive Example: Income Data

Consider analyzing household income:
- **Mean**: Sensitive to billionaires and extreme values
- **Median**: Resistant to outliers, represents typical household
- **Robust methods**: Provide reliable estimates despite outliers

### Why Use Robust Statistics?

1. **Outlier resistance**: Less sensitive to extreme values
2. **Distribution-free**: Work with non-normal data
3. **Breakdown point**: Measure of robustness
4. **Efficiency**: Good performance under ideal conditions
5. **Real-world data**: Often contains outliers and violations

### 5.1 Median and Median Absolute Deviation (MAD)

#### Mathematical Foundation

**Median**:
The value separating the higher half from the lower half of a data sample. It is a robust measure of central tendency.

**Median Absolute Deviation (MAD)**:
```math
\text{MAD} = \text{median}(|x_i - \text{median}(x)|)
```
MAD is a robust measure of statistical dispersion.

**Scaling**:
For normal data, $`\text{MAD} \times 1.4826`$ estimates the standard deviation.

#### Example: Income Distribution

**Dataset**: [30, 35, 40, 45, 50, 55, 60, 65, 70, 1000]
**Mean**: 140 (inflated by outlier)
**Median**: 52.5 (robust to outlier)
**Standard deviation**: 304 (inflated by outlier)
**MAD**: 15 (robust to outlier)

#### Properties of MAD

- **Breakdown point**: 50% (can handle up to 50% outliers)
- **Efficiency**: 37% relative to standard deviation for normal data
- **Computation**: Simple and fast
- **Interpretation**: Similar to standard deviation

### 5.2 Robust Regression

#### Mathematical Foundation

**Ordinary Least Squares (OLS)**:
Minimizes the sum of squared residuals, sensitive to outliers.

**Robust Regression (M-estimators)**:
Minimizes a function less sensitive to large residuals.

**Huber Loss**:
```math
L(\delta) = \begin{cases} 
\frac{1}{2}\delta^2 & \text{if } |\delta| \leq \epsilon \\ 
\epsilon(|\delta| - \frac{1}{2}\epsilon) & \text{if } |\delta| > \epsilon 
\end{cases}
```

Where $`\delta`$ is the residual and $`\epsilon`$ is a tuning parameter.

#### Example: Housing Price Prediction

**Dataset**: House prices with some outliers
**Models**: OLS vs. Robust regression

**Results**:
- **OLS**: Influenced by expensive outliers
- **Robust**: More stable estimates
- **Coefficients**: Similar for most variables, different for outliers

#### Other Robust Methods

1. **Least Absolute Deviations (LAD)**:
   ```math
   \min_{\beta} \sum_{i=1}^{n} |y_i - x_i^T\beta|
   ```

2. **RANSAC (Random Sample Consensus)**:
   - Fit model to random subset
   - Identify inliers and outliers
   - Repeat and choose best model

3. **Theil-Sen Estimator**:
   - Calculate slopes between all pairs
   - Use median slope as estimate

#### Python Implementation

```python
from scipy.stats import median_abs_deviation
from statsmodels.robust.robust_linear_model import RLM

# MAD
mad = median_abs_deviation(data)

# Robust regression
model = RLM(y, X, M=HuberT())
result = model.fit()
print(result.summary())
```

#### Comparison of Methods

| Method | Breakdown Point | Efficiency | Computation |
|--------|----------------|------------|-------------|
| Mean | 0% | 100% | Fast |
| Median | 50% | 64% | Fast |
| OLS | 0% | 100% | Fast |
| Huber | 0% | 95% | Fast |
| LAD | 50% | 64% | Slow |

#### Interpretation Guidelines

- **Median and MAD** provide robust alternatives to mean and standard deviation
- **Robust regression methods** yield more reliable parameter estimates in the presence of outliers
- **Always compare** robust and classical methods to assess the influence of outliers
- **Consider the trade-off** between robustness and efficiency
- **Use multiple methods** when data quality is uncertain

---

## 6. Practice Problems

### Problem 1: Non-Parametric Testing

**Objective**: Compare parametric and non-parametric methods.

**Tasks**:
1. **Simulate data**: Generate two groups, one normal and one skewed
2. **Apply tests**: Use t-test and Mann-Whitney U test
3. **Compare results**: Analyze differences in p-values and power
4. **Visualize**: Create distribution plots and test statistics
5. **Interpret**: Discuss when each method is appropriate

**Example Implementation**:
```python
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Generate data
group1 = np.random.normal(0, 1, 50)
group2 = np.random.exponential(1, 50)

# Compare tests
t_stat, t_p = ttest_ind(group1, group2)
u_stat, u_p = mannwhitneyu(group1, group2)

print(f"T-test: p = {t_p:.4f}")
print(f"Mann-Whitney: p = {u_p:.4f}")
```

### Problem 2: Survival Analysis

**Objective**: Analyze time-to-event data with censoring.

**Tasks**:
1. **Generate data**: Create synthetic survival data with censoring
2. **Fit models**: Implement Kaplan-Meier and Cox models
3. **Compare groups**: Test for differences in survival curves
4. **Check assumptions**: Validate proportional hazards
5. **Interpret results**: Explain survival probabilities and hazard ratios

### Problem 3: Mixed Models

**Objective**: Analyze hierarchical data structure.

**Tasks**:
1. **Simulate data**: Create nested data (e.g., students within schools)
2. **Fit models**: Compare standard regression and mixed models
3. **Analyze effects**: Interpret fixed and random effects
4. **Calculate ICC**: Determine proportion of variance between groups
5. **Compare results**: Discuss implications of ignoring group structure

### Problem 4: Causal Inference

**Objective**: Estimate causal effects in observational data.

**Tasks**:
1. **Simulate data**: Create dataset with confounding
2. **Apply methods**: Use propensity score matching and IV
3. **Compare estimates**: Analyze differences between naive and causal estimates
4. **Check assumptions**: Validate method requirements
5. **Interpret results**: Explain causal vs. correlational findings

### Problem 5: Robust Statistics

**Objective**: Handle outliers and violations of assumptions.

**Tasks**:
1. **Create dataset**: Generate data with outliers
2. **Compare methods**: Use classical and robust statistics
3. **Analyze sensitivity**: Study impact of outliers on estimates
4. **Visualize results**: Create comparison plots
5. **Recommend approach**: Suggest appropriate methods for different scenarios

**Guidance**:
- For each problem, write out the mathematical formulation, perform the analysis in Python, and interpret the results in the context of real-world data science or AI/ML applications
- Consider edge cases (e.g., small sample sizes, high censoring, strong confounding, extreme outliers) and discuss how robust or advanced methods address them
- Document assumptions and limitations of each method
- Provide practical recommendations for when to use each approach

---

## 7. Further Reading

### Books
- **"Survival Analysis: A Self-Learning Text"** by Kleinbaum & Klein  
  A comprehensive and accessible introduction to survival analysis, including practical examples and exercises.

- **"Causal Inference in Statistics: A Primer"** by Pearl, Glymour, & Jewell  
  An essential guide to the logic and mathematics of causal inference, with clear explanations and real-world examples.

- **"Robust Statistics"** by Huber & Ronchetti  
  The definitive reference on robust statistical methods, covering theory, algorithms, and applications.

- **"Applied Longitudinal Analysis"** by Fitzmaurice, Laird, & Ware  
  A practical resource for mixed models and longitudinal data analysis, with a focus on applications in health and social sciences.

- **"The Book of Why"** by Judea Pearl  
  An accessible and thought-provoking exploration of causality, counterfactuals, and the future of AI.

- **"Modern Applied Statistics with S"** by Venables & Ripley  
  A classic text covering a wide range of advanced statistical methods, with practical code examples (much of which translates to Python).

### Papers
- **"A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"** by Kohavi
- **"Random Forests"** by Breiman
- **"Greedy Function Approximation: A Gradient Boosting Machine"** by Friedman
- **"Regularization Paths for Generalized Linear Models via Coordinate Descent"** by Friedman et al.

### Online Resources
- **DoWhy Documentation** ([https://microsoft.github.io/dowhy/](https://microsoft.github.io/dowhy/))  
  Official documentation for the DoWhy Python library, with tutorials and case studies on causal inference in Python.

- **lifelines Documentation** ([https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/))  
  User guide and API reference for the lifelines library, including advanced survival analysis techniques and diagnostics.

- **Scikit-learn documentation** on model selection and evaluation
- **Cross-validation tutorials** and best practices
- **Ensemble methods guides** and implementations
- **Hyperparameter tuning** strategies and tools

### Advanced Topics
- **Bayesian Model Selection**: Using Bayesian methods for model comparison
- **Deep Learning**: Neural networks and deep architectures
- **Time Series**: Specialized methods for temporal data
- **Causal Inference**: Methods for establishing causality
- **Interpretable ML**: Making complex models understandable

### Research Areas
- **High-dimensional statistics**: Methods for p >> n problems
- **Functional data analysis**: Analyzing curves and surfaces
- **Spatial statistics**: Methods for geographic data
- **Network analysis**: Statistical methods for network data
- **Bayesian nonparametrics**: Flexible Bayesian modeling

**Tip**:
- When exploring advanced topics, always consult both theoretical and applied resources
- Try to implement methods from scratch in Python to deepen your understanding
- Consider the computational and interpretability trade-offs of different methods
- Stay current with recent developments in statistical methodology

---

## 8. Key Takeaways

### Fundamental Concepts
1. **Non-parametric methods** are powerful for non-normal or ordinal data, providing robust alternatives to classical tests.

2. **Survival analysis** is essential for time-to-event data, properly handling censoring and providing insights into duration outcomes.

3. **Mixed models** handle hierarchical and repeated measures data, accounting for group structure and providing proper uncertainty estimates.

4. **Causal inference** requires careful design and specialized methods to move beyond correlation to causation.

5. **Robust statistics** protect against outliers and model violations, providing reliable estimates when data doesn't meet classical assumptions.

### Mathematical Tools
- **Rank-based methods** for distribution-free inference
- **Survival functions** and hazard models for time-to-event analysis
- **Random effects** and variance components for hierarchical modeling
- **Propensity scores** and instrumental variables for causal inference
- **M-estimators** and robust loss functions for outlier-resistant estimation

### Applications
- **Clinical trials** and medical research using survival analysis
- **Educational research** and social sciences using mixed models
- **Marketing analytics** and policy evaluation using causal inference
- **Quality control** and outlier detection using robust statistics
- **Machine learning** and AI applications requiring advanced statistical methods

### Best Practices
- **Always check assumptions** before applying statistical methods
- **Use multiple approaches** when possible to triangulate results
- **Consider the context** and interpretability of your methods
- **Document limitations** and potential sources of bias
- **Stay current** with methodological developments in your field

### Next Steps
In the following chapters, we'll build on advanced statistical foundations to explore:
- **Deep Learning**: Neural networks and modern architectures
- **Bayesian Methods**: Probabilistic modeling and inference
- **High-dimensional Statistics**: Methods for big data problems
- **Computational Statistics**: Efficient algorithms for complex models
- **Reproducible Research**: Best practices for statistical analysis

Remember that advanced statistical methods are not just mathematical tools—they are essential for addressing real-world problems where simple approaches fail. The methods covered in this chapter provide the foundation for sophisticated data analysis, causal inference, and evidence-based decision making in complex, messy, real-world scenarios.

**Congratulations!** You have completed the comprehensive statistics guide for AI/ML and data science, covering from basic concepts through advanced methods. This knowledge provides a solid foundation for tackling complex data science challenges and contributing to the field of artificial intelligence and machine learning. 