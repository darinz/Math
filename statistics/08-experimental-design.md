# Experimental Design

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## Introduction

Experimental design is crucial for establishing causal relationships and making valid inferences. This chapter covers randomized controlled trials, factorial designs, blocking, and their applications in AI/ML.

### Why Experimental Design Matters

Experimental design provides the foundation for causal inference, allowing us to:

1. **Establish Causality**: Distinguish correlation from causation
2. **Control Confounding**: Eliminate bias from extraneous variables
3. **Maximize Power**: Detect effects with minimal sample size
4. **Ensure Validity**: Internal and external validity of results
5. **Optimize Resources**: Efficient use of time, money, and participants

### The Challenge of Causality

Establishing causality is one of the most difficult problems in science. Correlation does not imply causation, and many factors can confound our understanding of relationships.

#### Intuitive Example: Ice Cream and Crime

Consider the correlation between ice cream sales and crime rates:
- **Observation**: Both increase in summer months
- **Confounder**: Temperature (causes both ice cream sales and outdoor activity)
- **Solution**: Controlled experiment with random assignment

### Types of Experimental Designs

1. **Randomized Controlled Trials**: Gold standard for causal inference
2. **Factorial Designs**: Test multiple factors simultaneously
3. **Blocking Designs**: Control for known sources of variation
4. **Crossover Designs**: Subjects receive multiple treatments
5. **Sequential Designs**: Adaptive experimentation

## Table of Contents
- [Randomized Controlled Trials](#randomized-controlled-trials)
- [Factorial Designs](#factorial-designs)
- [Blocking and Randomization](#blocking-and-randomization)
- [Sample Size Determination](#sample-size-determination)
- [A/B Testing](#ab-testing)
- [Practical Applications](#practical-applications)

## Setup

The examples in this chapter use Python libraries for experimental design and analysis, including statistical tests, power analysis, and visualization tools.

## Randomized Controlled Trials

Randomized Controlled Trials (RCTs) are the gold standard for establishing causal relationships. The key principle is random assignment, which ensures that treatment and control groups are comparable on average across all observed and unobserved characteristics.

### Understanding RCTs

Think of RCTs as the "scientific method" for establishing causality. By randomly assigning subjects to treatment and control groups, we create a fair comparison that allows us to isolate the effect of the treatment.

#### Intuitive Example: Drug Efficacy Study

Consider testing a new drug for blood pressure:
- **Treatment Group**: Receives the new drug
- **Control Group**: Receives placebo
- **Randomization**: Ensures groups are comparable in age, health, etc.
- **Outcome**: Blood pressure reduction
- **Causal Inference**: Difference in outcomes attributed to drug

### Mathematical Foundation

#### Potential Outcomes Framework

For each subject i, we define:
- $`Y_i(1)`$: outcome if subject receives treatment
- $`Y_i(0)`$: outcome if subject receives control

**Individual Treatment Effect**:
```math
\tau_i = Y_i(1) - Y_i(0)
```

**Average Treatment Effect (ATE)**:
```math
\tau = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]
```

#### Randomization Properties

**1. Unconfoundedness**:
```math
(Y_i(1), Y_i(0)) \perp T_i
```

Where $`T_i`$ is treatment assignment. This means treatment assignment is independent of potential outcomes.

**2. Overlap**:
```math
0 < P(T_i = 1) < 1
```

For all subjects, ensuring both treatment and control groups exist.

**3. SUTVA**: Stable Unit Treatment Value Assumption
- No interference between units
- No different versions of treatment

#### Estimation of ATE

**Simple Difference-in-Means Estimator**:
```math
\hat{\tau} = \bar{Y}_1 - \bar{Y}_0
```

Where $`\bar{Y}_1`$ and $`\bar{Y}_0`$ are sample means of treated and control groups.

**Standard Error**:
```math
SE(\hat{\tau}) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}
```

Where $`s_1^2, s_0^2`$ are sample variances and $`n_1, n_0`$ are sample sizes.

**Confidence Interval**:
```math
\hat{\tau} \pm t_{\alpha/2, df} \cdot SE(\hat{\tau})
```

#### Example: Educational Intervention

**Study**: Testing new teaching method
**Sample**: 100 students randomly assigned
**Treatment**: New method (n₁ = 50)
**Control**: Standard method (n₀ = 50)
**Outcome**: Test scores

**Results**:
- $`\bar{Y}_1 = 85.2`$ (treatment mean)
- $`\bar{Y}_0 = 78.4`$ (control mean)
- $`s_1^2 = 64`$, $`s_0^2 = 72`$

**ATE Estimate**:
```math
\hat{\tau} = 85.2 - 78.4 = 6.8
```

**Standard Error**:
```math
SE(\hat{\tau}) = \sqrt{\frac{64}{50} + \frac{72}{50}} = \sqrt{2.72} = 1.65
```

**95% Confidence Interval**:
```math
6.8 \pm 1.96 \cdot 1.65 = [3.57, 10.03]
```

### Basic RCT Design

#### Design Principles

**1. Randomization**: Ensures comparability of groups
**2. Blinding**: Reduces bias from expectations
**3. Control**: Provides baseline for comparison
**4. Replication**: Increases reliability of results

#### Implementation Steps

1. **Define Population**: Clear inclusion/exclusion criteria
2. **Random Assignment**: Use random number generator
3. **Implement Treatment**: Ensure fidelity to protocol
4. **Measure Outcomes**: Standardized measurement
5. **Analyze Results**: Appropriate statistical tests

#### Example: Weight Loss Study

**Population**: Adults aged 25-65 with BMI > 30
**Treatment**: New diet program
**Control**: Standard diet advice
**Outcome**: Weight loss (kg) after 12 weeks
**Sample Size**: 200 participants (100 per group)

**Randomization**: Computer-generated random numbers
**Blinding**: Outcome assessors blinded to group assignment
**Analysis**: t-test for difference in means

### Stratified Randomization

Stratified randomization ensures balance across important covariates by performing separate randomizations within each stratum.

#### Mathematical Concept

**Stratified ATE Estimation**:
```math
\hat{\tau}_{stratified} = \sum_{s=1}^{S} w_s \hat{\tau}_s
```

Where:
- $`w_s`$ is the weight for stratum s (usually proportional to stratum size)
- $`\hat{\tau}_s`$ is the estimated treatment effect in stratum s

**Variance of Stratified Estimator**:
```math
Var(\hat{\tau}_{stratified}) = \sum_{s=1}^{S} w_s^2 Var(\hat{\tau}_s)
```

#### Example: Clinical Trial by Age Group

**Strata**: Age groups (18-30, 31-50, 51-70)
**Weights**: $`w_1 = 0.3`$, $`w_2 = 0.4`$, $`w_3 = 0.3`$
**Treatment Effects**: $`\hat{\tau}_1 = 5.2`$, $`\hat{\tau}_2 = 4.8`$, $`\hat{\tau}_3 = 3.1`$

**Stratified ATE**:
```math
\hat{\tau}_{stratified} = 0.3(5.2) + 0.4(4.8) + 0.3(3.1) = 4.41
```

#### Benefits

1. **Reduced Variance**: More precise estimates when strata are homogeneous
2. **Guaranteed Balance**: Ensures treatment groups are balanced on stratifying variables
3. **Subgroup Analysis**: Enables analysis of treatment effects within strata

#### Implementation

1. **Identify Strata**: Choose variables that predict outcome
2. **Allocate Sample**: Determine sample size per stratum
3. **Randomize Within**: Perform separate randomization in each stratum
4. **Analyze**: Use stratified estimator

### Cluster Randomization

When individual randomization is not feasible, we randomize groups (clusters) instead of individuals.

#### Mathematical Framework

**Cluster-Level Analysis**:
```math
\hat{\tau} = \bar{Y}_{1,cluster} - \bar{Y}_{0,cluster}
```

Where $`\bar{Y}_{1,cluster}`$ and $`\bar{Y}_{0,cluster}`$ are cluster-level means.

**Intraclass Correlation Coefficient (ICC)**:
```math
ICC = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2}
```

Where $`\sigma_b^2`$ is between-cluster variance and $`\sigma_w^2`$ is within-cluster variance.

#### Example: School-Based Intervention

**Clusters**: Schools (20 treatment, 20 control)
**Individuals**: Students within schools
**Outcome**: Academic performance
**ICC**: 0.1 (moderate clustering)

**Design Effect**:
```math
DE = 1 + (m-1)ICC = 1 + (25-1)(0.1) = 3.4
```

Where m is average cluster size.

## Factorial Designs

Factorial designs efficiently test multiple factors and their interactions simultaneously.

### Understanding Factorial Designs

Factorial designs allow us to study the effects of multiple factors and their interactions in a single experiment, making efficient use of resources.

#### Intuitive Example: Website Optimization

Consider optimizing a website:
- **Factor A**: Button color (red vs. blue)
- **Factor B**: Button size (small vs. large)
- **Combinations**: 4 treatment combinations
- **Efficiency**: Test both factors simultaneously

### 2x2 Factorial Design

#### Mathematical Framework

**Model**:
```math
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
```

Where:
- $`\mu`$ = overall mean
- $`\alpha_i`$ = effect of factor A at level i
- $`\beta_j`$ = effect of factor B at level j
- $`(\alpha\beta)_{ij}`$ = interaction effect
- $`\epsilon_{ijk}`$ = random error

#### Main Effects

**Factor A Main Effect**:
```math
\alpha_1 = \frac{Y_{1..} - Y_{2..}}{2}
```

**Factor B Main Effect**:
```math
\beta_1 = \frac{Y_{.1.} - Y_{.2.}}{2}
```

#### Interaction Effect

**AB Interaction**:
```math
(\alpha\beta)_{11} = \frac{Y_{11.} - Y_{12.} - Y_{21.} + Y_{22.}}{4}
```

#### Example: Drug Efficacy Study

**Factor A**: Drug dose (low vs. high)
**Factor B**: Administration time (morning vs. evening)
**Outcome**: Blood pressure reduction

**Results**:
- $`Y_{11} = 8`$ (low dose, morning)
- $`Y_{12} = 6`$ (low dose, evening)
- $`Y_{21} = 12`$ (high dose, morning)
- $`Y_{22} = 10`$ (high dose, evening)

**Main Effects**:
- **Dose**: $`\alpha_1 = \frac{(8+6)-(12+10)}{2} = -4`$ (high dose better)
- **Time**: $`\beta_1 = \frac{(8+12)-(6+10)}{2} = 2`$ (morning better)

**Interaction**: $`(\alpha\beta)_{11} = \frac{8-6-12+10}{4} = 0`$ (no interaction)

### Higher-Order Factorial Designs

#### 2³ Factorial Design

**Three factors**: A, B, C each at 2 levels
**Treatment combinations**: 8 total
**Effects**: 3 main effects, 3 two-way interactions, 1 three-way interaction

**Model**:
```math
Y_{ijkl} = \mu + \alpha_i + \beta_j + \gamma_k + (\alpha\beta)_{ij} + (\alpha\gamma)_{ik} + (\beta\gamma)_{jk} + (\alpha\beta\gamma)_{ijk} + \epsilon_{ijkl}
```

#### Fractional Factorial Designs

When full factorial designs are too expensive, we use fractional designs.

**2^(k-p) Design**: k factors, p generators
**Resolution**: Minimum number of factors in defining relation

**Example**: 2^(5-2) design
- **Factors**: A, B, C, D, E
- **Generators**: D = AB, E = AC
- **Defining relation**: I = ABD = ACE = BCDE
- **Resolution**: III (some main effects confounded with two-way interactions)

### Analysis of Factorial Designs

#### ANOVA Table

**Source** | **SS** | **df** | **MS** | **F**
--- | --- | --- | --- | ---
Factor A | $`SS_A`$ | $`a-1`$ | $`MS_A`$ | $`MS_A/MS_E`$
Factor B | $`SS_B`$ | $`b-1`$ | $`MS_B`$ | $`MS_B/MS_E`$
AB Interaction | $`SS_{AB}`$ | $`(a-1)(b-1)`$ | $`MS_{AB}`$ | $`MS_{AB}/MS_E`$
Error | $`SS_E`$ | $`ab(n-1)`$ | $`MS_E`$ |
Total | $`SS_T`$ | $`abn-1`$ |

#### Example: 2x2 ANOVA

**Data**: 4 observations per cell
**F-test**: Compare mean squares to error mean square
**p-values**: Determine statistical significance

## Blocking and Randomization

Blocking reduces variability and increases statistical power by controlling for known sources of variation.

### Understanding Blocking

Blocking is like "controlling what you can, randomizing what you can't." We group similar experimental units together to reduce within-block variability.

#### Intuitive Example: Agricultural Experiment

Consider testing fertilizer effectiveness:
- **Blocks**: Different fields (soil quality varies)
- **Treatments**: Different fertilizers
- **Analysis**: Compare fertilizers within each field
- **Benefit**: Controls for soil quality differences

### Randomized Block Design

#### Mathematical Framework

**Model**:
```math
Y_{ij} = \mu + \tau_i + \beta_j + \epsilon_{ij}
```

Where:
- $`\mu`$ = overall mean
- $`\tau_i`$ = treatment effect i
- $`\beta_j`$ = block effect j
- $`\epsilon_{ij}`$ = random error

#### Treatment Effect Estimation

**Adjusted Treatment Means**:
```math
\bar{Y}_{i.} = \frac{1}{b}\sum_{j=1}^{b} Y_{ij}
```

**Treatment Effect**:
```math
\hat{\tau}_i = \bar{Y}_{i.} - \bar{Y}_{..}
```

Where $`\bar{Y}_{..}`$ is the grand mean.

#### Example: Drug Trial by Hospital

**Blocks**: 5 hospitals
**Treatments**: 3 drugs (A, B, C)
**Outcome**: Patient recovery time

**Data**:
```
Hospital | Drug A | Drug B | Drug C
1        | 15     | 12     | 18
2        | 14     | 13     | 17
3        | 16     | 11     | 19
4        | 15     | 12     | 18
5        | 14     | 13     | 17
```

**Analysis**: Remove block effects to isolate treatment effects

### Latin Square Design

Latin square designs control for two sources of variation simultaneously.

#### Structure

**3x3 Latin Square**:
```
A B C
B C A
C A B
```

**Properties**:
- Each treatment appears once in each row
- Each treatment appears once in each column
- Orthogonal blocking

#### Mathematical Model

**Model**:
```math
Y_{ijk} = \mu + \tau_i + \rho_j + \gamma_k + \epsilon_{ijk}
```

Where:
- $`\tau_i`$ = treatment effect
- $`\rho_j`$ = row effect
- $`\gamma_k`$ = column effect

#### Example: Car Testing

**Rows**: Drivers
**Columns**: Days
**Treatments**: Car models (A, B, C)
**Outcome**: Fuel efficiency

### Incomplete Block Designs

When block size is smaller than number of treatments.

#### Balanced Incomplete Block (BIB) Design

**Properties**:
- Each treatment appears in r blocks
- Each block contains k treatments
- Each pair of treatments appears together in λ blocks

**Parameters**:
```math
\lambda = \frac{r(k-1)}{t-1}
```

Where t is number of treatments.

#### Example: Taste Testing

**Treatments**: 6 food products
**Block size**: 3 (can only taste 3 at once)
**Design**: Each product tasted 5 times, each pair appears together twice

## Sample Size Determination

Sample size determination ensures adequate power to detect effects of interest.

### Understanding Power

Power is the probability of correctly rejecting a false null hypothesis. It depends on effect size, sample size, significance level, and variability.

#### Intuitive Example: Coin Flipping

Consider testing if a coin is fair:
- **Null hypothesis**: p = 0.5
- **Alternative**: p ≠ 0.5
- **Effect size**: How far from 0.5
- **Power**: Probability of detecting unfairness

### Power Analysis

#### Mathematical Framework

**Power Function**:
```math
Power = P(\text{Reject } H_0 | H_1 \text{ is true})
```

**For t-test**:
```math
Power = P\left(|t| > t_{\alpha/2, df} | \delta = \frac{\mu_1 - \mu_0}{\sigma/\sqrt{n}}\right)
```

Where $`\delta`$ is the standardized effect size.

#### Effect Size Measures

**Cohen's d**:
```math
d = \frac{\mu_1 - \mu_0}{\sigma}
```

**Interpretation**:
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8

#### Sample Size Calculation

**For Two-Sample t-test**:
```math
n = \frac{2(z_{\alpha/2} + z_{\beta})^2}{d^2}
```

Where:
- $`z_{\alpha/2}`$ = critical value for significance level
- $`z_{\beta}`$ = critical value for power (1-β)
- $`d`$ = standardized effect size

#### Example: Clinical Trial

**Effect size**: d = 0.5 (medium)
**Significance level**: α = 0.05
**Power**: 1-β = 0.8
**Sample size per group**:
```math
n = \frac{2(1.96 + 0.84)^2}{0.5^2} = \frac{2(7.84)}{0.25} = 63
```

### Multiple Testing Considerations

#### Bonferroni Correction

**Adjusted significance level**:
```math
\alpha_{adjusted} = \frac{\alpha}{m}
```

Where m is number of tests.

#### False Discovery Rate (FDR)

**Benjamini-Hochberg procedure**:
1. Order p-values: $`p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}`$
2. Find largest k where $`p_{(k)} \leq \frac{k\alpha}{m}`$
3. Reject hypotheses 1 through k

#### Example: Multiple Endpoints

**Study**: 5 different outcome measures
**Original α**: 0.05
**Bonferroni α**: 0.01
**FDR**: Control false discovery rate at 0.05

## A/B Testing

A/B testing provides practical frameworks for online experiments and digital optimization.

### Understanding A/B Testing

A/B testing is the application of RCT principles to digital environments, allowing systematic optimization of user experiences.

#### Intuitive Example: Website Optimization

Consider testing a new website design:
- **Variant A**: Current design (control)
- **Variant B**: New design (treatment)
- **Metric**: Conversion rate
- **Goal**: Determine if new design improves conversions

### A/B Test Design and Analysis

#### Statistical Framework

**Hypothesis Test**:
- $`H_0`$: $`p_A = p_B`$ (no difference)
- $`H_1`$: $`p_A \neq p_B`$ (difference exists)

**Test Statistic**:
```math
z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_A} + \frac{1}{n_B})}}
```

Where $`\hat{p} = \frac{n_A\hat{p}_A + n_B\hat{p}_B}{n_A + n_B}`$ is the pooled proportion.

#### Sample Size Calculation

**For Proportion Test**:
```math
n = \frac{(z_{\alpha/2} + z_{\beta})^2(p_A(1-p_A) + p_B(1-p_B))}{(p_B - p_A)^2}
```

#### Example: Email Campaign

**Current conversion rate**: 2.5%
**Expected improvement**: 3.5%
**Significance level**: 0.05
**Power**: 0.8

**Sample size per group**:
```math
n = \frac{(1.96 + 0.84)^2(0.025(0.975) + 0.035(0.965))}{(0.035 - 0.025)^2} = 2,847
```

### Sequential Testing

Sequential testing allows early stopping when sufficient evidence is accumulated.

#### Sequential Probability Ratio Test (SPRT)

**Boundaries**:
```math
A = \log\left(\frac{1-\beta}{\alpha}\right), \quad B = \log\left(\frac{\beta}{1-\alpha}\right)
```

**Test statistic**:
```math
S_n = \sum_{i=1}^n \log\left(\frac{f_1(x_i)}{f_0(x_i)}\right)
```

**Decision rules**:
- Continue if $`B < S_n < A`$
- Reject $`H_0`$ if $`S_n \geq A`$
- Accept $`H_0`$ if $`S_n \leq B`$

#### Example: Website Testing

**Null hypothesis**: No improvement in conversion rate
**Alternative**: 20% improvement
**α = 0.05, β = 0.1**

**Boundaries**: A = 2.89, B = -2.20
**Analysis**: Monitor cumulative log-likelihood ratio

### Multi-Armed Bandits

Multi-armed bandits balance exploration and exploitation in adaptive experiments.

#### ε-Greedy Algorithm

**Strategy**:
- With probability ε: random exploration
- With probability 1-ε: exploit best arm

**Regret**:
```math
R(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})
```

Where $`\mu^*`$ is the best arm's mean reward.

#### Thompson Sampling

**Bayesian approach**:
1. Sample from posterior for each arm
2. Choose arm with highest sampled value
3. Update posterior with observed reward

**Advantages**:
- Natural uncertainty quantification
- Automatic exploration-exploitation balance
- No tuning parameters

## Practical Applications

### Clinical Trial Design

Clinical trials are the foundation of evidence-based medicine.

#### Phase I Trials

**Purpose**: Safety and dose finding
**Design**: Dose escalation
**Sample size**: 20-80 patients
**Analysis**: Maximum tolerated dose (MTD)

#### Phase II Trials

**Purpose**: Efficacy and safety
**Design**: Single-arm or randomized
**Sample size**: 100-300 patients
**Analysis**: Response rate, progression-free survival

#### Phase III Trials

**Purpose**: Confirmatory efficacy
**Design**: Randomized controlled trial
**Sample size**: 300-3000 patients
**Analysis**: Primary and secondary endpoints

#### Example: Cancer Drug Trial

**Phase I**: Dose escalation (20 patients)
**Phase II**: Efficacy in specific cancer type (150 patients)
**Phase III**: Confirmatory trial (500 patients per arm)

**Endpoints**:
- **Primary**: Overall survival
- **Secondary**: Progression-free survival, quality of life

### Agricultural Experiments

Agricultural experiments test new varieties, fertilizers, and management practices.

#### Split-Plot Design

**Whole plots**: Large areas (e.g., irrigation methods)
**Subplots**: Smaller areas within whole plots (e.g., varieties)

**Model**:
```math
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \gamma_k + (\alpha\gamma)_{ik} + \epsilon_{ijk}
```

Where:
- $`\alpha_i`$ = whole plot effect
- $`\beta_j`$ = subplot effect
- $`\gamma_k`$ = block effect

#### Example: Crop Yield Study

**Whole plots**: Irrigation methods (drip, sprinkler)
**Subplots**: Varieties (A, B, C, D)
**Blocks**: Fields
**Outcome**: Yield (tons/hectare)

### Industrial Experiments

Industrial experiments optimize manufacturing processes.

#### Response Surface Methodology

**First-order model**:
```math
Y = \beta_0 + \sum_{i=1}^k \beta_i X_i + \epsilon
```

**Second-order model**:
```math
Y = \beta_0 + \sum_{i=1}^k \beta_i X_i + \sum_{i=1}^k \beta_{ii} X_i^2 + \sum_{i=1}^k \sum_{j=i+1}^k \beta_{ij} X_i X_j + \epsilon
```

#### Example: Chemical Process

**Factors**: Temperature, pressure, catalyst concentration
**Response**: Yield percentage
**Design**: Central composite design
**Analysis**: Response surface optimization

### Social Science Experiments

Social science experiments study human behavior and decision-making.

#### Field Experiments

**Natural settings**: Real-world environments
**Randomization**: Natural or artificial
**Outcomes**: Behavioral measures

#### Laboratory Experiments

**Controlled environment**: Laboratory settings
**Randomization**: Computer-generated
**Outcomes**: Response times, choices

#### Example: Behavioral Economics

**Study**: Nudge interventions for retirement savings
**Treatment**: Automatic enrollment vs. opt-in
**Control**: Standard enrollment process
**Outcome**: Participation rate

## Practice Problems

### Problem 1: RCT Design

**Objective**: Create functions to design and analyze different types of randomized controlled trials.

**Tasks**:
1. Implement simple randomization algorithms
2. Create stratified randomization functions
3. Add cluster randomization analysis
4. Include power analysis for different designs
5. Add covariate adjustment methods

**Example Implementation**:
```python
def design_rct(n_treatment, n_control, stratification_vars=None):
    """
    Design a randomized controlled trial.
    
    Returns: treatment assignments, analysis plan, power calculations
    """
    # Implementation here
```

### Problem 2: Factorial Analysis

**Objective**: Implement comprehensive factorial design analysis.

**Tasks**:
1. Create factorial design generators
2. Add ANOVA analysis for factorial designs
3. Implement interaction testing
4. Include fractional factorial designs
5. Add response surface methodology

### Problem 3: Power Analysis

**Objective**: Build power analysis tools for different experimental designs.

**Tasks**:
1. Implement power calculations for t-tests
2. Add power analysis for proportions
3. Create power analysis for factorial designs
4. Include multiple testing corrections
5. Add sequential testing methods

### Problem 4: Sequential Testing

**Objective**: Develop sequential testing frameworks for early stopping.

**Tasks**:
1. Implement SPRT algorithm
2. Add group sequential methods
3. Create multi-armed bandit algorithms
4. Include Bayesian sequential testing
5. Add monitoring and stopping rules

### Problem 5: Real-World Experimental Design

**Objective**: Apply experimental design principles to real problems.

**Tasks**:
1. Choose application area (clinical, agricultural, industrial)
2. Design appropriate experiment
3. Perform power analysis
4. Create analysis plan
5. Write comprehensive design report

## Further Reading

### Books
- **"Design and Analysis of Experiments"** by Douglas C. Montgomery
- **"Statistics for Experimenters"** by Box, Hunter, and Hunter
- **"Experimental Design"** by Roger E. Kirk
- **"A/B Testing: The Most Powerful Way to Turn Clicks Into Customers"** by Dan Siroker and Pete Koomen
- **"Clinical Trials: A Methodologic Perspective"** by Steven Piantadosi

### Online Resources
- **RCT Registry**: ClinicalTrials.gov
- **Experimental Design Software**: JMP, Minitab, R packages
- **A/B Testing Platforms**: Optimizely, Google Optimize
- **Statistical Computing**: R, Python, SAS

### Advanced Topics
- **Adaptive Designs**: Response-adaptive randomization
- **Bayesian Experimental Design**: Optimal design under uncertainty
- **Causal Inference**: Rubin's potential outcomes framework
- **Machine Learning**: Experimental design for ML systems
- **Multi-Objective Optimization**: Balancing multiple outcomes

## Key Takeaways

### Fundamental Concepts
- **Randomized controlled trials** are the gold standard for establishing causality
- **Factorial designs** efficiently test multiple factors and their interactions
- **Blocking** reduces variability and increases statistical power
- **Sample size determination** ensures adequate power to detect effects
- **A/B testing** provides practical frameworks for online experiments
- **Proper randomization** is essential for valid statistical inference
- **Multiple endpoints** require careful consideration of multiple testing
- **Subgroup analysis** can reveal important treatment effect heterogeneity

### Mathematical Tools
- **Potential outcomes framework** provides foundation for causal inference
- **ANOVA** analyzes factorial designs and interactions
- **Power analysis** determines required sample sizes
- **Sequential testing** enables early stopping in experiments
- **Multi-armed bandits** balance exploration and exploitation

### Applications
- **Clinical trials** establish safety and efficacy of medical treatments
- **Agricultural experiments** optimize crop production and management
- **Industrial experiments** improve manufacturing processes
- **Social science experiments** study human behavior and decision-making
- **Digital optimization** improves user experiences and business outcomes

### Best Practices
- **Always randomize** when possible to ensure comparability
- **Use appropriate blocking** to control for known sources of variation
- **Calculate power** before conducting experiments
- **Plan for multiple testing** when analyzing multiple endpoints
- **Monitor experiments** for early stopping opportunities
- **Document procedures** for reproducibility and transparency

### Next Steps
In the following chapters, we'll build on experimental design foundations to explore:
- **Statistical Learning**: Cross-validation, model selection, and ensemble methods
- **Causal Inference**: Advanced methods for establishing causality
- **Machine Learning**: Experimental design for ML systems
- **Advanced Topics**: Specialized methods for complex experimental scenarios

Remember that experimental design is not just about statistical methods—it's about creating fair comparisons that allow us to make valid causal inferences. The principles and methods covered in this chapter provide the foundation for rigorous scientific investigation and evidence-based decision making. 