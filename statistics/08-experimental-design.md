# Experimental Design

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

Experimental design is crucial for establishing causal relationships and making valid inferences. This chapter covers randomized controlled trials, factorial designs, blocking, and their applications in AI/ML.

## Table of Contents
- [Randomized Controlled Trials](#randomized-controlled-trials)
- [Factorial Designs](#factorial-designs)
- [Blocking and Randomization](#blocking-and-randomization)
- [Sample Size Determination](#sample-size-determination)
- [A/B Testing](#ab-testing)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import power
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Randomized Controlled Trials

### Mathematical Foundation

**Randomized Controlled Trials (RCTs)** are the gold standard for establishing causal relationships. The key principle is random assignment, which ensures that treatment and control groups are comparable on average across all observed and unobserved characteristics.

**Causal Inference Framework:**
- **Potential Outcomes**: For each subject i, we define:
  - $Y_i(1)$: outcome if subject receives treatment
  - $Y_i(0)$: outcome if subject receives control
- **Individual Treatment Effect**: $\tau_i = Y_i(1) - Y_i(0)$
- **Average Treatment Effect (ATE)**: $\tau = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$

**Randomization Properties:**
1. **Unconfoundedness**: $(Y_i(1), Y_i(0)) \perp T_i$ where $T_i$ is treatment assignment
2. **Overlap**: $0 < P(T_i = 1) < 1$ for all subjects
3. **SUTVA**: Stable Unit Treatment Value Assumption (no interference between units)

**Estimation of ATE:**
$$\hat{\tau} = \bar{Y}_1 - \bar{Y}_0$$
where $\bar{Y}_1$ and $\bar{Y}_0$ are sample means of treated and control groups.

**Standard Error:**
$$SE(\hat{\tau}) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}$$
where $s_1^2, s_0^2$ are sample variances and $n_1, n_0$ are sample sizes.

### Basic RCT Design

```python
def simulate_rct(n_treatment=50, n_control=50, treatment_effect=5, noise=2):
    """
    Simulate a randomized controlled trial
    
    Mathematical implementation:
    1. Generate potential outcomes: Y(0) ~ N(μ₀, σ²), Y(1) = Y(0) + τ
    2. Randomly assign treatment: T ~ Bernoulli(p) where p = n₁/(n₀ + n₁)
    3. Observe outcomes: Y = T*Y(1) + (1-T)*Y(0)
    4. Estimate ATE: τ̂ = Ȳ₁ - Ȳ₀
    
    Parameters:
    n_treatment: int, size of treatment group
    n_control: int, size of control group
    treatment_effect: float, true treatment effect
    noise: float, standard deviation of outcomes
    
    Returns:
    tuple: (DataFrame, true_effect)
    """
    n_total = n_treatment + n_control
    
    # Step 1: Generate potential outcomes
    # Control potential outcomes: Y(0) ~ N(100, noise²)
    y0 = np.random.normal(100, noise, n_total)
    
    # Treatment potential outcomes: Y(1) = Y(0) + treatment_effect
    y1 = y0 + treatment_effect
    
    # Step 2: Random treatment assignment
    # Probability of treatment assignment
    p_treatment = n_treatment / n_total
    
    # Random assignment
    treatment_assignment = np.random.binomial(1, p_treatment, n_total)
    
    # Step 3: Observe outcomes (SUTVA assumption)
    observed_outcomes = treatment_assignment * y1 + (1 - treatment_assignment) * y0
    
    # Create DataFrame
    df_rct = pd.DataFrame({
        'subject_id': range(n_total),
        'treatment_assignment': treatment_assignment,
        'y0_potential': y0,
        'y1_potential': y1,
        'observed_outcome': observed_outcomes,
        'group': ['treatment' if t == 1 else 'control' for t in treatment_assignment]
    })
    
    return df_rct, treatment_effect

df_rct, true_effect = simulate_rct()

print("Randomized Controlled Trial Simulation")
print(f"Control group size: {len(df_rct[df_rct['group'] == 'control'])}")
print(f"Treatment group size: {len(df_rct[df_rct['group'] == 'treatment'])}")
print(f"True treatment effect: {true_effect}")

# Verify randomization balance
print(f"\nRandomization Balance Check:")
print(f"Treatment assignment probability: {df_rct['treatment_assignment'].mean():.3f}")
print(f"Expected probability: {len(df_rct[df_rct['group'] == 'treatment']) / len(df_rct):.3f}")

# Analyze RCT results
def analyze_rct(data):
    """
    Analyze RCT results with detailed statistical calculations
    
    Mathematical steps:
    1. Calculate group means: Ȳ₁, Ȳ₀
    2. Estimate ATE: τ̂ = Ȳ₁ - Ȳ₀
    3. Calculate standard error: SE(τ̂) = √(s₁²/n₁ + s₀²/n₀)
    4. Perform t-test: t = τ̂ / SE(τ̂)
    5. Calculate confidence interval: τ̂ ± t_{α/2,df} × SE(τ̂)
    
    Parameters:
    data: DataFrame, RCT data
    
    Returns:
    dict: analysis results
    """
    control_data = data[data['group'] == 'control']['observed_outcome']
    treatment_data = data[data['group'] == 'treatment']['observed_outcome']
    
    n0, n1 = len(control_data), len(treatment_data)
    
    # Step 1: Calculate group means
    control_mean = control_data.mean()
    treatment_mean = treatment_data.mean()
    
    # Step 2: Estimate ATE
    estimated_effect = treatment_mean - control_mean
    
    # Step 3: Calculate standard error
    # Pooled variance estimator (assuming equal variances)
    pooled_var = ((n0 - 1) * control_data.var() + (n1 - 1) * treatment_data.var()) / (n0 + n1 - 2)
    se_diff = np.sqrt(pooled_var * (1/n0 + 1/n1))
    
    # Step 4: Perform t-test
    t_stat = estimated_effect / se_diff
    df = n0 + n1 - 2  # degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Step 5: Calculate confidence interval
    t_critical = stats.t.ppf(0.975, df)  # 95% CI
    ci_lower = estimated_effect - t_critical * se_diff
    ci_upper = estimated_effect + t_critical * se_diff
    
    # Step 6: Calculate effect size (Cohen's d)
    cohens_d = estimated_effect / np.sqrt(pooled_var)
    
    # Step 7: Calculate power (post-hoc)
    # For a two-sample t-test with equal sample sizes
    effect_size_for_power = abs(estimated_effect) / np.sqrt(pooled_var)
    power_achieved = stats.t.cdf(stats.t.ppf(0.975, df) - effect_size_for_power * np.sqrt(n0/2), df)
    
    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'estimated_effect': estimated_effect,
        'standard_error': se_diff,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'cohens_d': cohens_d,
        'power_achieved': power_achieved,
        'pooled_variance': pooled_var
    }

rct_results = analyze_rct(df_rct)

print(f"\nRCT Analysis Results:")
print(f"Control mean: {rct_results['control_mean']:.2f}")
print(f"Treatment mean: {rct_results['treatment_mean']:.2f}")
print(f"Estimated ATE: {rct_results['estimated_effect']:.2f}")
print(f"Standard error: {rct_results['standard_error']:.3f}")
print(f"t-statistic: {rct_results['t_statistic']:.3f}")
print(f"Degrees of freedom: {rct_results['degrees_of_freedom']}")
print(f"p-value: {rct_results['p_value']:.4f}")
print(f"95% CI: [{rct_results['confidence_interval'][0]:.2f}, {rct_results['confidence_interval'][1]:.2f}]")
print(f"Cohen's d: {rct_results['cohens_d']:.3f}")
print(f"Power achieved: {rct_results['power_achieved']:.3f}")

# Verify potential outcomes framework
print(f"\nPotential Outcomes Verification:")
print(f"True ATE: {true_effect}")
print(f"Estimated ATE: {rct_results['estimated_effect']:.2f}")
print(f"Bias: {rct_results['estimated_effect'] - true_effect:.3f}")

# Visualize RCT results
plt.figure(figsize=(15, 5))

# Box plot
plt.subplot(1, 3, 1)
sns.boxplot(data=df_rct, x='group', y='observed_outcome')
plt.title('RCT Results - Box Plot')
plt.ylabel('Observed Outcome')

# Histogram
plt.subplot(1, 3, 2)
control_data = df_rct[df_rct['group'] == 'control']['observed_outcome']
treatment_data = df_rct[df_rct['group'] == 'treatment']['observed_outcome']

plt.hist(control_data, alpha=0.7, label='Control', bins=15, density=True)
plt.hist(treatment_data, alpha=0.7, label='Treatment', bins=15, density=True)
plt.xlabel('Observed Outcome')
plt.ylabel('Density')
plt.title('RCT Results - Histogram')
plt.legend()

# Effect size distribution
plt.subplot(1, 3, 3)
effect_sizes = []
for _ in range(1000):
    # Bootstrap resampling
    control_boot = np.random.choice(control_data, size=len(control_data), replace=True)
    treatment_boot = np.random.choice(treatment_data, size=len(treatment_data), replace=True)
    effect_sizes.append(treatment_boot.mean() - control_boot.mean())

plt.hist(effect_sizes, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
plt.axvline(true_effect, color='red', linestyle='--', linewidth=2, label=f'True effect: {true_effect}')
plt.axvline(rct_results['estimated_effect'], color='blue', linestyle='--', linewidth=2, 
           label=f'Estimated effect: {rct_results["estimated_effect"]:.2f}')
plt.xlabel('Treatment Effect')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Treatment Effect')
plt.legend()

plt.tight_layout()
plt.show()
```

### Stratified Randomization

**Mathematical Concept:**
Stratified randomization ensures balance across important covariates by performing separate randomizations within each stratum.

**Stratified ATE Estimation:**
$$\hat{\tau}_{stratified} = \sum_{s=1}^{S} w_s \hat{\tau}_s$$
where $w_s$ is the weight for stratum s (usually proportional to stratum size) and $\hat{\tau}_s$ is the estimated treatment effect in stratum s.

**Variance of Stratified Estimator:**
$$Var(\hat{\tau}_{stratified}) = \sum_{s=1}^{S} w_s^2 Var(\hat{\tau}_s)$$

**Benefits:**
1. **Reduced Variance**: More precise estimates when strata are homogeneous
2. **Guaranteed Balance**: Ensures treatment groups are balanced on stratifying variables
3. **Subgroup Analysis**: Enables analysis of treatment effects within strata

```python
def stratified_rct(n_per_stratum=25, n_strata=4):
    """
    Simulate stratified randomized controlled trial
    
    Mathematical implementation:
    1. Define strata (e.g., age groups, severity levels)
    2. Within each stratum s:
       - Generate potential outcomes: Yₛ(0) ~ N(μₛ, σ²), Yₛ(1) = Yₛ(0) + τₛ
       - Randomly assign treatment: T ~ Bernoulli(0.5) within stratum
    3. Estimate stratified ATE: τ̂ = Σ wₛ τ̂ₛ
    
    Parameters:
    n_per_stratum: int, sample size per stratum per group
    n_strata: int, number of strata
    
    Returns:
    DataFrame: stratified RCT data
    """
    strata = []
    outcomes = []
    groups = []
    treatment_assignments = []
    stratum_effects = []
    
    for stratum in range(n_strata):
        # Different baseline outcomes for each stratum
        baseline = 90 + stratum * 5
        
        # Different treatment effects for each stratum
        treatment_effect = 3 + stratum * 2
        
        # Generate potential outcomes for this stratum
        y0_stratum = np.random.normal(baseline, 3, n_per_stratum * 2)
        y1_stratum = y0_stratum + treatment_effect
        
        # Random treatment assignment within stratum
        treatment_assignment = np.random.binomial(1, 0.5, n_per_stratum * 2)
        
        # Observe outcomes
        observed_outcomes = treatment_assignment * y1_stratum + (1 - treatment_assignment) * y0_stratum
        
        # Store data
        strata.extend([stratum] * n_per_stratum * 2)
        outcomes.extend(observed_outcomes)
        groups.extend(['treatment' if t == 1 else 'control' for t in treatment_assignment])
        treatment_assignments.extend(treatment_assignment)
        stratum_effects.extend([treatment_effect] * n_per_stratum * 2)
    
    df_stratified = pd.DataFrame({
        'stratum': strata,
        'group': groups,
        'outcome': outcomes,
        'treatment_assignment': treatment_assignments,
        'stratum_effect': stratum_effects
    })
    
    return df_stratified

df_stratified = stratified_rct()

print("Stratified RCT Simulation")
print(f"Total sample size: {len(df_stratified)}")
print(f"Number of strata: {df_stratified['stratum'].nunique()}")
print(f"Sample size per stratum: {len(df_stratified) // (df_stratified['stratum'].nunique() * 2)}")

# Analyze stratified RCT
def analyze_stratified_rct(data):
    """
    Analyze stratified RCT results
    
    Mathematical steps:
    1. Calculate stratum-specific treatment effects: τ̂ₛ = Ȳ₁ₛ - Ȳ₀ₛ
    2. Calculate stratum weights: wₛ = nₛ / N
    3. Estimate overall ATE: τ̂ = Σ wₛ τ̂ₛ
    4. Calculate variance: Var(τ̂) = Σ wₛ² Var(τ̂ₛ)
    
    Parameters:
    data: DataFrame, stratified RCT data
    
    Returns:
    dict: analysis results
    """
    strata = data['stratum'].unique()
    stratum_effects = []
    stratum_variances = []
    stratum_weights = []
    
    for stratum in strata:
        stratum_data = data[data['stratum'] == stratum]
        control_data = stratum_data[stratum_data['group'] == 'control']['outcome']
        treatment_data = stratum_data[stratum_data['group'] == 'treatment']['outcome']
        
        # Stratum-specific treatment effect
        stratum_effect = treatment_data.mean() - control_data.mean()
        stratum_effects.append(stratum_effect)
        
        # Stratum-specific variance
        n0_s, n1_s = len(control_data), len(treatment_data)
        pooled_var_s = ((n0_s - 1) * control_data.var() + (n1_s - 1) * treatment_data.var()) / (n0_s + n1_s - 2)
        stratum_var = pooled_var_s * (1/n0_s + 1/n1_s)
        stratum_variances.append(stratum_var)
        
        # Stratum weight
        stratum_weight = len(stratum_data) / len(data)
        stratum_weights.append(stratum_weight)
    
    # Overall stratified estimate
    stratified_ate = np.sum(np.array(stratum_weights) * np.array(stratum_effects))
    
    # Overall variance
    stratified_var = np.sum(np.array(stratum_weights)**2 * np.array(stratum_variances))
    stratified_se = np.sqrt(stratified_var)
    
    # Statistical test
    t_stat = stratified_ate / stratified_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))
    
    # Confidence interval
    t_critical = stats.t.ppf(0.975, len(data) - 2)
    ci_lower = stratified_ate - t_critical * stratified_se
    ci_upper = stratified_ate + t_critical * stratified_se
    
    return {
        'stratum_effects': stratum_effects,
        'stratum_weights': stratum_weights,
        'stratified_ate': stratified_ate,
        'stratified_se': stratified_se,
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper)
    }

stratified_results = analyze_stratified_rct(df_stratified)

print(f"\nStratified RCT Analysis Results:")
for i, (effect, weight) in enumerate(zip(stratified_results['stratum_effects'], stratified_results['stratum_weights'])):
    print(f"Stratum {i}: Effect = {effect:.2f}, Weight = {weight:.3f}")
print(f"Overall stratified ATE: {stratified_results['stratified_ate']:.2f}")
print(f"Standard error: {stratified_results['stratified_se']:.3f}")
print(f"t-statistic: {stratified_results['t_statistic']:.3f}")
print(f"p-value: {stratified_results['p_value']:.4f}")
print(f"95% CI: [{stratified_results['confidence_interval'][0]:.2f}, {stratified_results['confidence_interval'][1]:.2f}]")

# Compare with unstratified analysis
unstratified_results = analyze_rct(df_stratified)
print(f"\nComparison:")
print(f"Unstratified ATE: {unstratified_results['estimated_effect']:.2f}")
print(f"Stratified ATE: {stratified_results['stratified_ate']:.2f}")
print(f"Unstratified SE: {unstratified_results['standard_error']:.3f}")
print(f"Stratified SE: {stratified_results['stratified_se']:.3f}")
print(f"Efficiency gain: {(unstratified_results['standard_error'] / stratified_results['stratified_se'])**2:.2f}x")
```

## Factorial Designs

### 2x2 Factorial Design

```python
def factorial_2x2_design(n_per_cell=30):
    """Simulate 2x2 factorial design"""
    
    # Factors: A (drug) and B (dose)
    factor_a = ['low', 'high']  # Drug levels
    factor_b = ['low', 'high']  # Dose levels
    
    # True effects
    baseline = 100
    effect_a = 5    # Main effect of drug
    effect_b = 3    # Main effect of dose
    effect_ab = 2   # Interaction effect
    
    data = []
    
    for i, a in enumerate(factor_a):
        for j, b in enumerate(factor_b):
            # Calculate cell mean
            cell_mean = baseline
            if a == 'high':
                cell_mean += effect_a
            if b == 'high':
                cell_mean += effect_b
            if a == 'high' and b == 'high':
                cell_mean += effect_ab
            
            # Generate outcomes
            outcomes = np.random.normal(cell_mean, 2, n_per_cell)
            
            for outcome in outcomes:
                data.append({
                    'factor_a': a,
                    'factor_b': b,
                    'outcome': outcome
                })
    
    return pd.DataFrame(data)

df_factorial = factorial_2x2_design()

print("2x2 Factorial Design Simulation")
print(f"Total sample size: {len(df_factorial)}")
print(f"Sample size per cell: {len(df_factorial) // 4}")

# Analyze factorial design
def analyze_factorial_design(data):
    """Analyze factorial design using ANOVA"""
    
    from scipy.stats import f_oneway
    from itertools import combinations
    
    # Cell means
    cell_means = data.groupby(['factor_a', 'factor_b'])['outcome'].mean()
    
    # Main effects
    main_effect_a = (cell_means.loc[('high', 'low')] + cell_means.loc[('high', 'high')]) / 2 - \
                   (cell_means.loc[('low', 'low')] + cell_means.loc[('low', 'high')]) / 2
    
    main_effect_b = (cell_means.loc[('low', 'high')] + cell_means.loc[('high', 'high')]) / 2 - \
                   (cell_means.loc[('low', 'low')] + cell_means.loc[('high', 'low')]) / 2
    
    # Interaction effect
    interaction = (cell_means.loc[('high', 'high')] - cell_means.loc[('high', 'low')]) - \
                 (cell_means.loc[('low', 'high')] - cell_means.loc[('low', 'low')])
    
    # ANOVA
    groups = [group['outcome'].values for name, group in data.groupby(['factor_a', 'factor_b'])]
    f_stat, p_value = f_oneway(*groups)
    
    return {
        'cell_means': cell_means,
        'main_effect_a': main_effect_a,
        'main_effect_b': main_effect_b,
        'interaction': interaction,
        'f_statistic': f_stat,
        'p_value': p_value
    }

factorial_results = analyze_factorial_design(df_factorial)

print(f"\nFactorial Design Analysis:")
print(f"Main effect A (drug): {factorial_results['main_effect_a']:.2f}")
print(f"Main effect B (dose): {factorial_results['main_effect_b']:.2f}")
print(f"Interaction effect: {factorial_results['interaction']:.2f}")
print(f"F-statistic: {factorial_results['f_statistic']:.3f}")
print(f"p-value: {factorial_results['p_value']:.4f}")

# Visualize factorial design
plt.figure(figsize=(15, 5))

# Interaction plot
plt.subplot(1, 3, 1)
for factor_b in ['low', 'high']:
    subset = df_factorial[df_factorial['factor_b'] == factor_b]
    means = subset.groupby('factor_a')['outcome'].mean()
    plt.plot(['low', 'high'], means, 'o-', label=f'Dose {factor_b}', linewidth=2, markersize=8)

plt.xlabel('Drug Level')
plt.ylabel('Outcome')
plt.title('Interaction Plot')
plt.legend()

# Cell means heatmap
plt.subplot(1, 3, 2)
pivot_table = df_factorial.pivot_table(values='outcome', index='factor_a', columns='factor_b', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=pivot_table.values.mean(), 
            square=True, linewidths=0.5, fmt='.1f')
plt.title('Cell Means')

# Main effects
plt.subplot(1, 3, 3)
effects = ['Main Effect A', 'Main Effect B', 'Interaction']
effect_values = [factorial_results['main_effect_a'], 
                factorial_results['main_effect_b'], 
                factorial_results['interaction']]

colors = ['red' if abs(effect) > 1 else 'blue' for effect in effect_values]
plt.bar(effects, effect_values, color=colors, alpha=0.7)
plt.axhline(0, color='black', linestyle='-', alpha=0.7)
plt.ylabel('Effect Size')
plt.title('Main Effects and Interaction')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Blocking and Randomization

### Randomized Block Design

```python
def randomized_block_design(n_blocks=6, n_treatments=3):
    """Simulate randomized block design"""
    
    # Generate blocks (e.g., different centers, time periods)
    blocks = []
    treatments = []
    outcomes = []
    
    for block in range(n_blocks):
        # Block-specific baseline
        baseline = 100 + block * 2
        
        # Randomize treatments within each block
        block_treatments = np.random.permutation(n_treatments)
        
        for treatment in range(n_treatments):
            # Treatment effects
            treatment_effect = treatment * 3
            
            # Generate outcome
            outcome = baseline + treatment_effect + np.random.normal(0, 2)
            
            blocks.append(block)
            treatments.append(treatment)
            outcomes.append(outcome)
    
    df_block = pd.DataFrame({
        'block': blocks,
        'treatment': treatments,
        'outcome': outcomes
    })
    
    return df_block

df_block = randomized_block_design()

print("Randomized Block Design Simulation")
print(f"Number of blocks: {df_block['block'].nunique()}")
print(f"Number of treatments: {df_block['treatment'].nunique()}")
print(f"Total sample size: {len(df_block)}")

# Analyze block design
def analyze_block_design(data):
    """Analyze randomized block design"""
    
    # Calculate means
    overall_mean = data['outcome'].mean()
    block_means = data.groupby('block')['outcome'].mean()
    treatment_means = data.groupby('treatment')['outcome'].mean()
    
    # Calculate effects
    block_effects = block_means - overall_mean
    treatment_effects = treatment_means - overall_mean
    
    # Two-way ANOVA (treatments and blocks)
    from scipy.stats import f_oneway
    
    # Treatment groups
    treatment_groups = [group['outcome'].values for name, group in data.groupby('treatment')]
    f_treatment, p_treatment = f_oneway(*treatment_groups)
    
    # Block groups
    block_groups = [group['outcome'].values for name, group in data.groupby('block')]
    f_block, p_block = f_oneway(*block_groups)
    
    return {
        'overall_mean': overall_mean,
        'block_effects': block_effects,
        'treatment_effects': treatment_effects,
        'f_treatment': f_treatment,
        'p_treatment': p_treatment,
        'f_block': f_block,
        'p_block': p_block
    }

block_results = analyze_block_design(df_block)

print(f"\nBlock Design Analysis:")
print(f"Overall mean: {block_results['overall_mean']:.2f}")
print(f"Treatment F-statistic: {block_results['f_treatment']:.3f}")
print(f"Treatment p-value: {block_results['p_treatment']:.4f}")
print(f"Block F-statistic: {block_results['f_block']:.3f}")
print(f"Block p-value: {block_results['p_block']:.4f}")

# Visualize block design
plt.figure(figsize=(15, 5))

# Treatment means
plt.subplot(1, 3, 1)
treatment_means = df_block.groupby('treatment')['outcome'].mean()
plt.bar(treatment_means.index, treatment_means.values, alpha=0.7, color='skyblue')
plt.xlabel('Treatment')
plt.ylabel('Mean Outcome')
plt.title('Treatment Means')

# Block means
plt.subplot(1, 3, 2)
block_means = df_block.groupby('block')['outcome'].mean()
plt.bar(block_means.index, block_means.values, alpha=0.7, color='lightgreen')
plt.xlabel('Block')
plt.ylabel('Mean Outcome')
plt.title('Block Means')

# Interaction plot
plt.subplot(1, 3, 3)
for treatment in df_block['treatment'].unique():
    subset = df_block[df_block['treatment'] == treatment]
    means = subset.groupby('block')['outcome'].mean()
    plt.plot(means.index, means.values, 'o-', label=f'Treatment {treatment}', linewidth=2, markersize=6)

plt.xlabel('Block')
plt.ylabel('Outcome')
plt.title('Treatment Effects Across Blocks')
plt.legend()

plt.tight_layout()
plt.show()
```

## Sample Size Determination

### Power Analysis

```python
def power_analysis_example():
    """Demonstrate power analysis for different scenarios"""
    
    # Parameters
    alpha = 0.05  # Significance level
    power_levels = [0.8, 0.9, 0.95]  # Desired power levels
    effect_sizes = [0.2, 0.5, 0.8]   # Cohen's d effect sizes
    
    results = []
    
    for power in power_levels:
        for effect_size in effect_sizes:
            # Calculate required sample size
            n_per_group = power.tt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                ratio=1.0  # Equal group sizes
            )
            
            results.append({
                'power': power,
                'effect_size': effect_size,
                'n_per_group': int(n_per_group),
                'total_n': int(n_per_group * 2)
            })
    
    return pd.DataFrame(results)

power_results = power_analysis_example()

print("Power Analysis Results")
print(power_results.to_string(index=False))

# Visualize power analysis
plt.figure(figsize=(15, 5))

# Sample size vs effect size
plt.subplot(1, 3, 1)
for power in power_results['power'].unique():
    subset = power_results[power_results['power'] == power]
    plt.plot(subset['effect_size'], subset['n_per_group'], 'o-', 
             label=f'Power = {power}', linewidth=2, markersize=8)

plt.xlabel("Cohen's d Effect Size")
plt.ylabel('Sample Size per Group')
plt.title('Sample Size vs Effect Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Power curves
plt.subplot(1, 3, 2)
sample_sizes = [20, 50, 100, 200]
effect_sizes = np.linspace(0.1, 1.0, 50)

for n in sample_sizes:
    powers = []
    for effect_size in effect_sizes:
        power = power.tt_ind_power(effect_size, n, n, alpha=0.05)
        powers.append(power)
    plt.plot(effect_sizes, powers, label=f'n = {n}', linewidth=2)

plt.xlabel("Cohen's d Effect Size")
plt.ylabel('Power')
plt.title('Power Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Effect size detection
plt.subplot(1, 3, 3)
n_values = np.arange(10, 201, 10)
min_effect_sizes = []

for n in n_values:
    # Find minimum detectable effect size for 80% power
    min_effect = power.tt_ind_solve_power(
        effect_size=None,
        alpha=0.05,
        power=0.8,
        nobs1=n
    )
    min_effect_sizes.append(min_effect)

plt.plot(n_values, min_effect_sizes, 'b-', linewidth=2)
plt.xlabel('Sample Size per Group')
plt.ylabel('Minimum Detectable Effect Size')
plt.title('Minimum Detectable Effect Size')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Practical example
def sample_size_calculation_example():
    """Calculate sample size for a practical example"""
    
    # Scenario: Testing a new drug vs placebo
    # Expected effect size: 0.5 (medium effect)
    # Desired power: 0.9
    # Significance level: 0.05
    
    effect_size = 0.5
    desired_power = 0.9
    alpha = 0.05
    
    n_per_group = power.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=desired_power,
        ratio=1.0
    )
    
    total_n = n_per_group * 2
    
    print(f"Sample Size Calculation Example:")
    print(f"Effect size (Cohen's d): {effect_size}")
    print(f"Desired power: {desired_power}")
    print(f"Significance level: {alpha}")
    print(f"Required sample size per group: {int(n_per_group)}")
    print(f"Total sample size: {int(total_n)}")
    
    return int(n_per_group), int(total_n)

sample_size_example = sample_size_calculation_example()
```

## A/B Testing

### A/B Test Design and Analysis

```python
def ab_test_simulation(n_a=1000, n_b=1000, true_rate_a=0.10, true_rate_b=0.12):
    """Simulate A/B test with conversion rates"""
    
    # Generate data
    conversions_a = np.random.binomial(n_a, true_rate_a)
    conversions_b = np.random.binomial(n_b, true_rate_b)
    
    # Calculate observed rates
    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b
    
    # Statistical test
    from scipy.stats import proportions_ztest
    z_stat, p_value = proportions_ztest([conversions_a, conversions_b], [n_a, n_b])
    
    # Confidence interval for difference
    pooled_rate = (conversions_a + conversions_b) / (n_a + n_b)
    se_diff = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n_a + 1/n_b))
    diff = rate_b - rate_a
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    
    return {
        'conversions_a': conversions_a,
        'conversions_b': conversions_b,
        'rate_a': rate_a,
        'rate_b': rate_b,
        'difference': diff,
        'z_statistic': z_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper)
    }

ab_results = ab_test_simulation()

print("A/B Test Results")
print(f"Group A: {ab_results['conversions_a']}/{1000} conversions ({ab_results['rate_a']:.3f})")
print(f"Group B: {ab_results['conversions_b']}/{1000} conversions ({ab_results['rate_b']:.3f})")
print(f"Difference: {ab_results['difference']:.4f}")
print(f"Z-statistic: {ab_results['z_statistic']:.3f}")
print(f"P-value: {ab_results['p_value']:.4f}")
print(f"95% CI: [{ab_results['confidence_interval'][0]:.4f}, {ab_results['confidence_interval'][1]:.4f}]")

# Sequential A/B testing
def sequential_ab_test(n_max=2000, true_rate_a=0.10, true_rate_b=0.12, alpha=0.05):
    """Simulate sequential A/B testing"""
    
    conversions_a = []
    conversions_b = []
    decisions = []
    sample_sizes = []
    
    for n in range(100, n_max + 1, 100):
        # Generate data up to current n
        conv_a = np.random.binomial(n//2, true_rate_a)
        conv_b = np.random.binomial(n//2, true_rate_b)
        
        conversions_a.append(conv_a)
        conversions_b.append(conv_b)
        
        # Statistical test
        z_stat, p_value = proportions_ztest([conv_a, conv_b], [n//2, n//2])
        
        # Decision rule
        if p_value < alpha:
            decision = 'reject' if z_stat > 0 else 'reject'
        else:
            decision = 'continue'
        
        decisions.append(decision)
        sample_sizes.append(n)
    
    return sample_sizes, conversions_a, conversions_b, decisions

sample_sizes, conv_a_seq, conv_b_seq, decisions = sequential_ab_test()

# Visualize sequential testing
plt.figure(figsize=(15, 5))

# Conversion rates over time
plt.subplot(1, 3, 1)
rates_a = [conv_a / (n//2) for conv_a, n in zip(conv_a_seq, sample_sizes)]
rates_b = [conv_b / (n//2) for conv_b, n in zip(conv_b_seq, sample_sizes)]

plt.plot(sample_sizes, rates_a, 'b-', label='Group A', linewidth=2)
plt.plot(sample_sizes, rates_b, 'r-', label='Group B', linewidth=2)
plt.xlabel('Sample Size')
plt.ylabel('Conversion Rate')
plt.title('Sequential A/B Test - Conversion Rates')
plt.legend()

# P-values over time
plt.subplot(1, 3, 2)
p_values = []
for conv_a, conv_b, n in zip(conv_a_seq, conv_b_seq, sample_sizes):
    z_stat, p_val = proportions_ztest([conv_a, conv_b], [n//2, n//2])
    p_values.append(p_val)

plt.plot(sample_sizes, p_values, 'g-', linewidth=2)
plt.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
plt.xlabel('Sample Size')
plt.ylabel('P-value')
plt.title('Sequential A/B Test - P-values')
plt.legend()

# Decision timeline
plt.subplot(1, 3, 3)
decision_colors = {'continue': 'yellow', 'reject': 'red'}
colors = [decision_colors[decision] for decision in decisions]

plt.scatter(sample_sizes, [1] * len(sample_sizes), c=colors, alpha=0.7, s=50)
plt.xlabel('Sample Size')
plt.ylabel('Decision')
plt.title('Sequential A/B Test - Decisions')
plt.yticks([])

plt.tight_layout()
plt.show()

# A/B test power analysis
def ab_test_power_analysis():
    """Power analysis for A/B testing"""
    
    # Parameters
    baseline_rate = 0.10
    effect_sizes = [0.01, 0.02, 0.05, 0.10]  # Absolute differences
    sample_sizes = [500, 1000, 2000, 5000]
    
    results = []
    
    for effect_size in effect_sizes:
        for n in sample_sizes:
            # Calculate power
            power_val = power.proportion_2ind_power(
                diff=effect_size,
                prop2=baseline_rate,
                nobs1=n//2,
                alpha=0.05
            )
            
            results.append({
                'effect_size': effect_size,
                'sample_size': n,
                'power': power_val
            })
    
    return pd.DataFrame(results)

ab_power_results = ab_test_power_analysis()

print("\nA/B Test Power Analysis")
print(ab_power_results.to_string(index=False))
```

## Practical Applications

### Clinical Trial Design

```python
def clinical_trial_simulation():
    """Simulate a clinical trial with multiple endpoints"""
    
    # Trial parameters
    n_patients = 200
    treatment_effect_bp = 5    # Blood pressure reduction (mmHg)
    treatment_effect_chol = 10 # Cholesterol reduction (mg/dL)
    
    # Generate patient data
    np.random.seed(42)
    
    # Baseline characteristics
    age = np.random.normal(60, 10, n_patients)
    gender = np.random.binomial(1, 0.5, n_patients)
    
    # Randomize to treatment groups
    treatment = np.random.binomial(1, 0.5, n_patients)
    
    # Generate outcomes
    baseline_bp = 140 + 0.5 * age + np.random.normal(0, 10, n_patients)
    baseline_chol = 200 + 0.3 * age + np.random.normal(0, 20, n_patients)
    
    # Treatment effects
    bp_reduction = treatment * treatment_effect_bp + np.random.normal(0, 5, n_patients)
    chol_reduction = treatment * treatment_effect_chol + np.random.normal(0, 15, n_patients)
    
    # Final outcomes
    final_bp = baseline_bp - bp_reduction
    final_chol = baseline_chol - chol_reduction
    
    # Create DataFrame
    df_trial = pd.DataFrame({
        'patient_id': range(n_patients),
        'age': age,
        'gender': gender,
        'treatment': treatment,
        'baseline_bp': baseline_bp,
        'final_bp': final_bp,
        'baseline_chol': baseline_chol,
        'final_chol': final_chol,
        'bp_change': -bp_reduction,
        'chol_change': -chol_reduction
    })
    
    return df_trial

df_trial = clinical_trial_simulation()

print("Clinical Trial Simulation")
print(f"Number of patients: {len(df_trial)}")
print(f"Treatment group size: {df_trial['treatment'].sum()}")
print(f"Control group size: {(1 - df_trial['treatment']).sum()}")

# Analyze clinical trial
def analyze_clinical_trial(data):
    """Analyze clinical trial results"""
    
    # Primary endpoint: Blood pressure
    control_bp = data[data['treatment'] == 0]['bp_change']
    treatment_bp = data[data['treatment'] == 1]['bp_change']
    
    bp_effect = treatment_bp.mean() - control_bp.mean()
    bp_t_stat, bp_p_value = stats.ttest_ind(treatment_bp, control_bp)
    
    # Secondary endpoint: Cholesterol
    control_chol = data[data['treatment'] == 0]['chol_change']
    treatment_chol = data[data['treatment'] == 1]['chol_change']
    
    chol_effect = treatment_chol.mean() - control_chol.mean()
    chol_t_stat, chol_p_value = stats.ttest_ind(treatment_chol, control_chol)
    
    # Subgroup analysis
    male_data = data[data['gender'] == 1]
    female_data = data[data['gender'] == 0]
    
    male_effect = (male_data[male_data['treatment'] == 1]['bp_change'].mean() - 
                  male_data[male_data['treatment'] == 0]['bp_change'].mean())
    female_effect = (female_data[female_data['treatment'] == 1]['bp_change'].mean() - 
                    female_data[female_data['treatment'] == 0]['bp_change'].mean())
    
    return {
        'bp_effect': bp_effect,
        'bp_p_value': bp_p_value,
        'chol_effect': chol_effect,
        'chol_p_value': chol_p_value,
        'male_effect': male_effect,
        'female_effect': female_effect
    }

trial_results = analyze_clinical_trial(df_trial)

print(f"\nClinical Trial Results:")
print(f"Blood pressure effect: {trial_results['bp_effect']:.2f} mmHg (p={trial_results['bp_p_value']:.4f})")
print(f"Cholesterol effect: {trial_results['chol_effect']:.2f} mg/dL (p={trial_results['chol_p_value']:.4f})")
print(f"Male subgroup effect: {trial_results['male_effect']:.2f} mmHg")
print(f"Female subgroup effect: {trial_results['female_effect']:.2f} mmHg")

# Visualize clinical trial results
plt.figure(figsize=(15, 10))

# Primary endpoint
plt.subplot(2, 3, 1)
control_bp = df_trial[df_trial['treatment'] == 0]['bp_change']
treatment_bp = df_trial[df_trial['treatment'] == 1]['bp_change']
plt.boxplot([control_bp, treatment_bp], labels=['Control', 'Treatment'])
plt.ylabel('Blood Pressure Change (mmHg)')
plt.title('Primary Endpoint - Blood Pressure')

# Secondary endpoint
plt.subplot(2, 3, 2)
control_chol = df_trial[df_trial['treatment'] == 0]['chol_change']
treatment_chol = df_trial[df_trial['treatment'] == 1]['chol_change']
plt.boxplot([control_chol, treatment_chol], labels=['Control', 'Treatment'])
plt.ylabel('Cholesterol Change (mg/dL)')
plt.title('Secondary Endpoint - Cholesterol')

# Subgroup analysis
plt.subplot(2, 3, 3)
male_control = df_trial[(df_trial['treatment'] == 0) & (df_trial['gender'] == 1)]['bp_change']
male_treatment = df_trial[(df_trial['treatment'] == 1) & (df_trial['gender'] == 1)]['bp_change']
female_control = df_trial[(df_trial['treatment'] == 0) & (df_trial['gender'] == 0)]['bp_change']
female_treatment = df_trial[(df_trial['treatment'] == 1) & (df_trial['gender'] == 0)]['bp_change']

plt.boxplot([male_control, male_treatment, female_control, female_treatment], 
           labels=['Male\nControl', 'Male\nTreatment', 'Female\nControl', 'Female\nTreatment'])
plt.ylabel('Blood Pressure Change (mmHg)')
plt.title('Subgroup Analysis')

# Baseline characteristics
plt.subplot(2, 3, 4)
plt.hist(df_trial[df_trial['treatment'] == 0]['age'], alpha=0.7, label='Control', bins=15)
plt.hist(df_trial[df_trial['treatment'] == 1]['age'], alpha=0.7, label='Treatment', bins=15)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.legend()

# Treatment effects
plt.subplot(2, 3, 5)
effects = ['BP Effect', 'Chol Effect', 'Male Effect', 'Female Effect']
effect_values = [trial_results['bp_effect'], trial_results['chol_effect'], 
                trial_results['male_effect'], trial_results['female_effect']]
colors = ['red' if effect < 0 else 'green' for effect in effect_values]

plt.bar(effects, effect_values, color=colors, alpha=0.7)
plt.axhline(0, color='black', linestyle='-', alpha=0.7)
plt.ylabel('Treatment Effect')
plt.title('Treatment Effects')
plt.xticks(rotation=45)

# P-values
plt.subplot(2, 3, 6)
endpoints = ['Blood Pressure', 'Cholesterol']
p_values = [trial_results['bp_p_value'], trial_results['chol_p_value']]
colors = ['red' if p < 0.05 else 'blue' for p in p_values]

plt.bar(endpoints, p_values, color=colors, alpha=0.7)
plt.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
plt.ylabel('P-value')
plt.title('Statistical Significance')
plt.legend()

plt.tight_layout()
plt.show()
```

## Practice Problems

1. **RCT Design**: Create functions to design and analyze different types of randomized controlled trials.

2. **Factorial Analysis**: Implement comprehensive factorial design analysis with interaction testing.

3. **Power Analysis**: Build power analysis tools for different experimental designs and effect sizes.

4. **Sequential Testing**: Develop sequential testing frameworks for early stopping in experiments.

## Further Reading

- "Design and Analysis of Experiments" by Douglas C. Montgomery
- "Statistics for Experimenters" by Box, Hunter, and Hunter
- "Experimental Design" by Roger E. Kirk
- "A/B Testing: The Most Powerful Way to Turn Clicks Into Customers" by Dan Siroker and Pete Koomen

## Key Takeaways

- **Randomized controlled trials** are the gold standard for establishing causality
- **Factorial designs** efficiently test multiple factors and their interactions
- **Blocking** reduces variability and increases statistical power
- **Sample size determination** ensures adequate power to detect effects
- **A/B testing** provides practical frameworks for online experiments
- **Proper randomization** is essential for valid statistical inference
- **Multiple endpoints** require careful consideration of multiple testing
- **Subgroup analysis** can reveal important treatment effect heterogeneity

In the next chapter, we'll explore statistical learning, including cross-validation, model selection, and ensemble methods. 