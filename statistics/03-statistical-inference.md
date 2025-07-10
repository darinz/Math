# Statistical Inference

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)

## Introduction

Statistical inference allows us to draw conclusions about populations based on sample data. This chapter covers hypothesis testing, confidence intervals, and p-values - essential tools for making data-driven decisions in AI/ML.

### Why Statistical Inference Matters

Statistical inference is the bridge between sample data and population conclusions. In the real world, we rarely have access to entire populations, so we must rely on samples to make informed decisions. This process is fundamental to:

1. **Scientific Research**: Testing theories and hypotheses
2. **Business Decisions**: Evaluating marketing campaigns, product changes
3. **Medical Studies**: Assessing treatment effectiveness
4. **Quality Control**: Monitoring manufacturing processes
5. **Machine Learning**: Validating model performance and feature importance

### The Inference Process

Statistical inference follows a systematic approach:

1. **Data Collection**: Gather a representative sample from the population
2. **Model Specification**: Choose appropriate statistical models and assumptions
3. **Parameter Estimation**: Calculate point estimates and intervals
4. **Hypothesis Testing**: Evaluate specific claims about population parameters
5. **Interpretation**: Draw conclusions and assess practical significance

### Types of Statistical Inference

1. **Point Estimation**: Single best guess for a population parameter
2. **Interval Estimation**: Range of plausible values (confidence intervals)
3. **Hypothesis Testing**: Evaluating specific claims about parameters
4. **Prediction**: Estimating future observations or outcomes

## Table of Contents
- [Hypothesis Testing Fundamentals](#hypothesis-testing-fundamentals)
- [One-Sample Tests](#one-sample-tests)
- [Two-Sample Tests](#two-sample-tests)
- [Confidence Intervals](#confidence-intervals)
- [P-Values and Significance](#p-values-and-significance)
- [Multiple Testing](#multiple-testing)
- [Practical Applications](#practical-applications)

## Setup

The examples in this chapter use Python libraries for statistical analysis and inference. We'll work with both theoretical concepts and practical implementations to build intuition and computational skills.

## Hypothesis Testing Fundamentals

Hypothesis testing is a formal procedure for making decisions about population parameters based on sample data. It provides a structured framework for evaluating claims and making data-driven decisions.

### Understanding Hypothesis Testing

Hypothesis testing is like a scientific trial where we:
1. **State a claim** about a population parameter
2. **Collect evidence** (sample data)
3. **Evaluate the evidence** against the claim
4. **Make a decision** based on the strength of evidence

#### The Scientific Method Analogy

Think of hypothesis testing as a courtroom trial:
- **Null Hypothesis (H₀)**: The defendant is innocent (default assumption)
- **Alternative Hypothesis (H₁)**: The defendant is guilty (what we want to prove)
- **Evidence**: Sample data
- **Verdict**: Reject or fail to reject H₀ based on evidence strength

### Mathematical Foundation

**Statistical Hypothesis Testing** is a formal procedure for making decisions about population parameters based on sample data. The process involves:

1. **Formulating Hypotheses**: 
   - **Null Hypothesis (H₀)**: A statement about the population parameter that we assume to be true
   - **Alternative Hypothesis (H₁)**: A statement that contradicts the null hypothesis

2. **Test Statistic**: A function of the sample data that follows a known probability distribution under H₀

3. **Decision Rule**: Based on the test statistic and significance level α, we either reject or fail to reject H₀

**Mathematical Framework:**
For a population parameter $`\theta`$, we test:
- $`H_0: \theta = \theta_0`$ (null hypothesis)
- $`H_1: \theta \neq \theta_0`$ (two-sided alternative) or $`\theta > \theta_0`$, $`\theta < \theta_0`$ (one-sided alternatives)

The test statistic $`T`$ is calculated from sample data and compared to critical values or used to compute p-values.

#### Intuitive Understanding

The key insight is that we assume the null hypothesis is true and ask: "How unusual would our sample data be if this assumption were correct?" If the data is very unusual under the null hypothesis, we have evidence against it.

#### Example: Coin Toss

Suppose we want to test if a coin is fair:
- $`H_0: p = 0.5`$ (coin is fair)
- $`H_1: p \neq 0.5`$ (coin is biased)
- We flip the coin 100 times and get 65 heads
- Test statistic: $`Z = \frac{65 - 50}{\sqrt{100 \times 0.5 \times 0.5}} = 3`$
- This is very unusual if the coin is fair (p-value ≈ 0.003)

### Null and Alternative Hypotheses

The null hypothesis represents the "status quo" or default assumption, while the alternative hypothesis represents what we want to demonstrate.

#### Formulating Hypotheses

**Null Hypothesis (H₀):**
- Represents the default or conservative position
- Usually contains equality ($`=`, $`\leq`, $`\geq`$)
- We assume it's true unless evidence suggests otherwise

**Alternative Hypothesis (H₁):**
- Represents the research hypothesis or claim
- Contains inequality ($`\neq`, $`>`, $`<`$)
- What we want to demonstrate with evidence

#### Types of Tests

1. **Two-Sided Test**: $`H_0: \theta = \theta_0`$ vs $`H_1: \theta \neq \theta_0`$
   - Used when we want to detect any difference from the hypothesized value
   - Most common in scientific research

2. **One-Sided Test**: $`H_0: \theta \leq \theta_0`$ vs $`H_1: \theta > \theta_0`$ (or $`<`$)
   - Used when we have a directional hypothesis
   - More powerful for detecting effects in the specified direction

#### Example: Drug Effectiveness

Testing a new drug for blood pressure reduction:
- **Two-sided**: $`H_0: \mu = 0`$ vs $`H_1: \mu \neq 0`$ (any change in blood pressure)
- **One-sided**: $`H_0: \mu \leq 0`$ vs $`H_1: \mu > 0`$ (only interested in reduction)

#### Guidelines for Hypothesis Formulation

1. **Be Specific**: Hypotheses should be precise and testable
2. **Consider Direction**: Decide if you need one-sided or two-sided tests
3. **Practical Significance**: Consider what magnitude of effect is meaningful
4. **Prior Knowledge**: Use domain expertise to inform hypothesis choice

### Type I and Type II Errors

Understanding the two types of errors is crucial for interpreting test results and making decisions.

#### Mathematical Definition

- **Type I Error ($`\alpha`$)**: Rejecting $`H_0`$ when it's true
- **Type II Error ($`\beta`$)**: Failing to reject $`H_0`$ when it's false
- **Power ($`1-\beta`$)**: Probability of correctly rejecting $`H_0`$ when it's false

#### Intuitive Understanding

Think of these errors in terms of a medical test:
- **Type I Error**: False positive - test says you have a disease when you don't
- **Type II Error**: False negative - test says you don't have a disease when you do
- **Power**: Ability of the test to detect the disease when it's present

#### Error Trade-off

As $`\alpha`$ decreases, $`\beta`$ increases, and vice versa. The relationship is:

```math
\beta = \Phi(z_{\alpha/2} - \frac{\delta}{\sigma/\sqrt{n}}) + \Phi(z_{\alpha/2} + \frac{\delta}{\sigma/\sqrt{n}})
```

Where $`\delta`$ is the effect size, $`\sigma`$ is the standard deviation, and $`n`$ is the sample size.

#### Example: Medical Test

Consider a medical test with:
- $`\alpha = 0.05`$ (5% false positive rate)
- $`\beta = 0.20`$ (20% false negative rate)
- Power = $`1 - 0.20 = 0.80`$ (80% chance of detecting disease when present)

#### Factors Affecting Error Rates

1. **Sample Size**: Larger samples reduce both error rates
2. **Effect Size**: Larger effects are easier to detect
3. **Variability**: Less variable data improves power
4. **Significance Level**: Lower $`\alpha`$ increases $`\beta`$

#### Power Analysis

Power analysis helps determine the sample size needed to detect a specified effect size with desired power:

```math
n = \frac{(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}
```

Where $`z_{\alpha/2}`$ and $`z_{\beta}`$ are critical values from the standard normal distribution.

#### Example: Sample Size Calculation

To detect a difference of 5 units with 80% power and 5% significance level:
- $`\alpha = 0.05`$, $`\beta = 0.20`$
- $`z_{0.025} = 1.96`$, $`z_{0.20} = 0.84`$
- $`\delta = 5`$, $`\sigma = 10`$
- $`n = \frac{(1.96 + 0.84)^2 \times 100}{25} = 31.36`$ ≈ 32 subjects per group

## One-Sample Tests

One-sample tests compare a sample statistic to a hypothesized population parameter. These are fundamental building blocks for more complex statistical analyses.

### Z-Test for Population Mean

The Z-test is used when we know the population standard deviation and want to test hypotheses about the population mean.

#### Mathematical Foundation

When we know the population standard deviation $`\sigma`$, we use the Z-test:

```math
Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}}
```

Where:
- $`\bar{X}`$ is the sample mean
- $`\mu_0`$ is the hypothesized population mean
- $`\sigma`$ is the population standard deviation
- $`n`$ is the sample size

#### Intuitive Understanding

The Z-test standardizes the difference between the sample mean and hypothesized mean by the standard error of the mean. This tells us how many standard errors the sample mean is from the hypothesized value.

#### Assumptions

1. **Normality**: Data is normally distributed (or $`n > 30`$ by CLT)
2. **Known Variance**: Population standard deviation is known
3. **Independence**: Observations are independent
4. **Random Sampling**: Sample is representative of population

#### Example: IQ Testing

A psychologist wants to test if a group of students has above-average IQ (population mean = 100, SD = 15):
- Sample: $`n = 25`$ students, $`\bar{x} = 108`$
- $`H_0: \mu = 100`$ vs $`H_1: \mu > 100`$
- $`Z = \frac{108 - 100}{15/\sqrt{25}} = \frac{8}{3} = 2.67`$
- p-value = $`P(Z > 2.67) = 0.0038`$
- Conclusion: Reject $`H_0`$ at $`\alpha = 0.05`$

#### Critical Value Approach

For a given significance level $`\alpha`$:
- **Two-sided**: Reject if $`|Z| > z_{\alpha/2}`$
- **One-sided**: Reject if $`Z > z_{\alpha}`$ (or $`Z < -z_{\alpha}`$)

#### Example: Critical Values

For $`\alpha = 0.05`$:
- Two-sided: $`z_{0.025} = 1.96`$
- One-sided: $`z_{0.05} = 1.645`$

### T-Test for Population Mean

The t-test is used when the population standard deviation is unknown and must be estimated from the sample data.

#### Mathematical Foundation

When population standard deviation is unknown, we use the t-test:

```math
t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}}
```

Where:
- $`s`$ is the sample standard deviation
- Degrees of freedom = $`n - 1`$

#### Key Differences from Z-test

1. **Unknown Variance**: Uses sample standard deviation instead of population standard deviation
2. **t-Distribution**: Follows t-distribution instead of normal distribution
3. **Degrees of Freedom**: Shape depends on sample size
4. **More Conservative**: t-distribution has heavier tails than normal distribution

#### Intuitive Understanding

The t-test accounts for the additional uncertainty introduced by estimating the population standard deviation from the sample. The t-distribution is more spread out than the normal distribution, making it harder to reject the null hypothesis.

#### Example: Battery Life

Testing if a new battery lasts longer than 10 hours:
- Sample: $`n = 16`$ batteries, $`\bar{x} = 11.2`$ hours, $`s = 1.5`$ hours
- $`H_0: \mu = 10`$ vs $`H_1: \mu > 10`$
- $`t = \frac{11.2 - 10}{1.5/\sqrt{16}} = \frac{1.2}{0.375} = 3.2`$
- Degrees of freedom = $`16 - 1 = 15`$
- p-value = $`P(t_{15} > 3.2) = 0.003`$
- Conclusion: Reject $`H_0`$ at $`\alpha = 0.05`$

#### Degrees of Freedom

The degrees of freedom represent the number of independent pieces of information available for estimating the parameter. For the t-test:
- $`df = n - 1`$ (we lose one degree of freedom by estimating the mean)

#### t-Distribution Properties

1. **Symmetric**: Like the normal distribution
2. **Heavier Tails**: More probability in the tails than normal
3. **Convergence**: As $`df \rightarrow \infty`$, t-distribution approaches normal
4. **Shape**: Depends on degrees of freedom

#### Example: Critical Values

For $`\alpha = 0.05`$ and $`df = 15`$:
- Two-sided: $`t_{0.025, 15} = 2.131`$
- One-sided: $`t_{0.05, 15} = 1.753`$

### Chi-Square Test for Variance

The chi-square test is used to test hypotheses about population variance.

#### Mathematical Foundation

```math
\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}
```

Where:
- $`s^2`$ is the sample variance
- $`\sigma_0^2`$ is the hypothesized population variance
- Degrees of freedom = $`n - 1`$

#### Example: Quality Control

Testing if a manufacturing process has variance less than 4:
- Sample: $`n = 20`$, $`s^2 = 2.5`$
- $`H_0: \sigma^2 = 4`$ vs $`H_1: \sigma^2 < 4`$
- $`\chi^2 = \frac{19 \times 2.5}{4} = 11.875`$
- p-value = $`P(\chi^2_{19} < 11.875) = 0.15`$
- Conclusion: Fail to reject $`H_0`$ at $`\alpha = 0.05`$

## Two-Sample Tests

Two-sample tests compare parameters between two different populations or groups. These are essential for experimental designs and comparative studies.

### Independent t-Test

The independent t-test compares means between two independent groups.

#### Mathematical Foundation

```math
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

Where:
- $`\bar{X}_1, \bar{X}_2`$ are sample means
- $`s_1^2, s_2^2`$ are sample variances
- $`n_1, n_2`$ are sample sizes

#### Degrees of Freedom

For unequal variances (Welch's t-test):
```math
df = \frac{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

For equal variances (pooled t-test):
```math
df = n_1 + n_2 - 2
```

#### Example: Drug Comparison

Comparing effectiveness of two drugs:
- Drug A: $`n_1 = 30`$, $`\bar{x}_1 = 85`$, $`s_1 = 12`$
- Drug B: $`n_2 = 25``, $`\bar{x}_2 = 78`$, $`s_2 = 15`$
- $`H_0: \mu_1 = \mu_2`$ vs $`H_1: \mu_1 \neq \mu_2`$
- $`t = \frac{85 - 78}{\sqrt{\frac{144}{30} + \frac{225}{25}}} = \frac{7}{\sqrt{4.8 + 9}} = 1.85`$
- p-value ≈ 0.07
- Conclusion: Fail to reject $`H_0`$ at $`\alpha = 0.05`$

#### Assumptions

1. **Independence**: Groups are independent
2. **Normality**: Data in each group is normally distributed
3. **Equal Variances**: Population variances are equal (for pooled test)
4. **Random Sampling**: Samples are representative

### Paired t-Test

The paired t-test compares means when observations are related (before/after, matched pairs).

#### Mathematical Foundation

```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

Where:
- $`\bar{d}`$ is the mean of differences
- $`s_d`$ is the standard deviation of differences
- $`n`$ is the number of pairs

#### Example: Before/After Study

Testing weight loss program effectiveness:
- Before: $`[180, 165, 200, 175, 190]`$
- After: $`[175, 160, 195, 170, 185]`$
- Differences: $`[-5, -5, -5, -5, -5]`$
- $`\bar{d} = -5`$, $`s_d = 0`$
- $`t = \frac{-5}{0/\sqrt{5}}`$ (undefined due to zero variance)
- In practice, there would be some variation in differences

#### Advantages of Paired Tests

1. **Reduced Variability**: Controls for individual differences
2. **Higher Power**: More sensitive to detect effects
3. **Fewer Assumptions**: Doesn't require equal variances

### F-Test for Equality of Variances

The F-test compares variances between two groups.

#### Mathematical Foundation

```math
F = \frac{s_1^2}{s_2^2}
```

Where $`s_1^2 \geq s_2^2`$ (larger variance in numerator).

#### Example: Process Comparison

Comparing variability of two manufacturing processes:
- Process A: $`n_1 = 15`$, $`s_1^2 = 4.2`$
- Process B: $`n_2 = 12`$, $`s_2^2 = 2.8`$
- $`F = \frac{4.2}{2.8} = 1.5`$
- Degrees of freedom: $`(14, 11)`$
- p-value ≈ 0.25
- Conclusion: Fail to reject equality of variances

## Confidence Intervals

Confidence intervals provide a range of plausible values for population parameters, complementing hypothesis tests by quantifying uncertainty.

### Confidence Interval Construction

Confidence intervals are constructed using the sampling distribution of the statistic.

#### General Formula

```math
\text{Statistic} \pm \text{Critical Value} \times \text{Standard Error}
```

#### Interpretation

A $`100(1-\alpha)\%`$ confidence interval means that if we repeated the sampling process many times, $`100(1-\alpha)\%`$ of the intervals would contain the true parameter value.

#### Example: Mean Estimation

For a sample mean with known population standard deviation:
```math
\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
```

#### Example: t-Interval

For a sample mean with unknown population standard deviation:
```math
\bar{X} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
```

#### Example: Proportion

For a sample proportion:
```math
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
```

### One-Sample Confidence Intervals

#### Mean with Known Variance

```math
\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
```

#### Mean with Unknown Variance

```math
\bar{X} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
```

#### Example: Battery Life

From the previous example:
- $`\bar{x} = 11.2`$ hours, $`s = 1.5`$ hours, $`n = 16`$
- 95% CI: $`11.2 \pm 2.131 \times \frac{1.5}{\sqrt{16}} = 11.2 \pm 0.8 = [10.4, 12.0]`$
- We are 95% confident that the true mean battery life is between 10.4 and 12.0 hours

### Two-Sample Confidence Intervals

#### Difference in Means (Independent)

```math
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, df} \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

#### Difference in Means (Paired)

```math
\bar{d} \pm t_{\alpha/2, n-1} \frac{s_d}{\sqrt{n}}
```

#### Example: Drug Comparison

From the previous example:
- $`\bar{x}_1 - \bar{x}_2 = 7`$
- 95% CI: $`7 \pm 2.01 \times \sqrt{4.8 + 9} = 7 \pm 7.4 = [-0.4, 14.4]`$
- We are 95% confident that the true difference in means is between -0.4 and 14.4

### Sample Size Determination

To achieve a desired margin of error $`E`$:

#### For Mean (Known Variance)

```math
n = \left(\frac{z_{\alpha/2} \sigma}{E}\right)^2
```

#### For Mean (Unknown Variance)

Use pilot study to estimate $`\sigma`$, then use the formula above.

#### Example: Sample Size

To estimate mean with margin of error 2 and 95% confidence:
- $`\sigma = 10`$, $`E = 2`$, $`z_{0.025} = 1.96`$
- $`n = \left(\frac{1.96 \times 10}{2}\right)^2 = 96.04`$ ≈ 97

## P-Values and Significance

P-values are the most commonly used measure of evidence against the null hypothesis, but they are often misunderstood.

### P-Value Interpretation

The p-value is the probability of observing a test statistic as extreme as or more extreme than the one observed, assuming the null hypothesis is true.

#### Mathematical Definition

For a test statistic $`T`$ with observed value $`t_{obs}`$:

- **Two-sided test**: $`p = P(|T| \geq |t_{obs}| | H_0)`$
- **One-sided test**: $`p = P(T \geq t_{obs} | H_0)`$ or $`P(T \leq t_{obs} | H_0)`$

#### Intuitive Understanding

The p-value answers: "If the null hypothesis were true, how likely would it be to see data as extreme as what we observed?" A small p-value suggests the data is unlikely under the null hypothesis.

#### Example: Coin Toss

From the earlier example:
- Observed: 65 heads in 100 tosses
- Expected under $`H_0`$: 50 heads
- Test statistic: $`Z = 3`$
- p-value = $`P(|Z| \geq 3) = 2 \times 0.0013 = 0.0026`$
- Interpretation: If the coin were fair, there's only a 0.26% chance of seeing such extreme results

#### Common Misinterpretations

1. **p-value ≠ Probability of H₀ being true**
2. **p-value ≠ Probability of making a Type I error**
3. **p-value ≠ Strength of evidence for H₁**

#### Guidelines for Interpretation

- **p < 0.001**: Very strong evidence against H₀
- **p < 0.01**: Strong evidence against H₀
- **p < 0.05**: Moderate evidence against H₀
- **p < 0.10**: Weak evidence against H₀
- **p ≥ 0.10**: Little or no evidence against H₀

### Significance Levels

The significance level $`\alpha`$ is the threshold for rejecting the null hypothesis.

#### Common Significance Levels

- **α = 0.001**: Very strict (0.1% false positive rate)
- **α = 0.01**: Strict (1% false positive rate)
- **α = 0.05**: Standard (5% false positive rate)
- **α = 0.10**: Liberal (10% false positive rate)

#### Decision Rule

- **Reject H₀** if p-value ≤ α
- **Fail to reject H₀** if p-value > α

#### Example: Multiple Significance Levels

For a test with p-value = 0.03:
- α = 0.01: Fail to reject (p > α)
- α = 0.05: Reject (p ≤ α)
- α = 0.10: Reject (p ≤ α)

### Effect Size

Effect size measures the practical significance of a result, complementing the statistical significance measured by p-values.

#### Cohen's d (Standardized Mean Difference)

```math
d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}
```

Where $`s_{pooled}`$ is the pooled standard deviation.

#### Interpretation Guidelines

- **|d| < 0.2**: Small effect
- **0.2 ≤ |d| < 0.5**: Medium effect
- **|d| ≥ 0.5**: Large effect

#### Example: Effect Size

From the drug comparison example:
- $`\bar{x}_1 - \bar{x}_2 = 7`$
- $`s_{pooled} = 13.5`$
- $`d = \frac{7}{13.5} = 0.52`$ (large effect)

## Multiple Testing

When conducting multiple hypothesis tests, the probability of making at least one Type I error increases. Multiple testing corrections help control this inflation.

### Multiple Testing Problem

If we conduct $`m`$ independent tests at significance level $`\alpha`$, the probability of at least one Type I error is:

```math
P(\text{at least one Type I error}) = 1 - (1-\alpha)^m
```

#### Example: Multiple Tests

For $`m = 10`$ tests at $`\alpha = 0.05`$:
- $`P(\text{at least one Type I error}) = 1 - (0.95)^{10} = 0.401`$
- There's a 40% chance of at least one false positive!

### Multiple Testing Correction

#### Bonferroni Correction

The simplest correction adjusts the significance level:

```math
\alpha_{adjusted} = \frac{\alpha}{m}
```

#### Example: Bonferroni Correction

For $`m = 10`$ tests and $`\alpha = 0.05`$:
- $`\alpha_{adjusted} = \frac{0.05}{10} = 0.005`$
- Each test must have p-value ≤ 0.005 to be significant

#### False Discovery Rate (FDR)

FDR controls the expected proportion of false positives among rejected hypotheses:

```math
\text{FDR} = E\left[\frac{\text{False Positives}}{\text{Rejected Hypotheses}}\right]
```

#### Benjamini-Hochberg Procedure

1. Order p-values: $`p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}`$
2. Find largest $`k`$ such that $`p_{(k)} \leq \frac{k}{m}\alpha`$
3. Reject hypotheses 1 through $`k`$

#### Example: FDR Control

For $`m = 10`$ tests with p-values: $`[0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]`$ and $`\alpha = 0.05`$:

- $`\frac{1}{10} \times 0.05 = 0.005`$: $`p_{(1)} = 0.001 \leq 0.005`$ ✓
- $`\frac{2}{10} \times 0.05 = 0.01`$: $`p_{(2)} = 0.01 \leq 0.01`$ ✓
- $`\frac{3}{10} \times 0.05 = 0.015`$: $`p_{(3)} = 0.02 > 0.015`$ ✗

Reject first 2 hypotheses.

### Comparison of Methods

| Method | Controls | Power | Complexity |
|--------|----------|-------|------------|
| **Bonferroni** | Family-wise error rate | Low | Simple |
| **FDR** | False discovery rate | Higher | Moderate |
| **Holm** | Family-wise error rate | Higher | Moderate |

## Practical Applications

Statistical inference finds applications across numerous fields and industries.

### A/B Testing Example

A/B testing is a common application of statistical inference in business and marketing.

#### Example: Website Conversion Rate

Testing two website designs:
- **Design A**: $`n_1 = 1000`$ visitors, $`x_1 = 50`$ conversions
- **Design B**: $`n_2 = 1000`$ visitors, $`x_2 = 65`$ conversions

**Hypothesis Test:**
- $`H_0: p_1 = p_2`$ vs $`H_1: p_1 \neq p_2`$
- $`\hat{p}_1 = 0.05`$, $`\hat{p}_2 = 0.065`$
- Test statistic: $`Z = \frac{0.065 - 0.05}{\sqrt{0.0575(1-0.0575)(\frac{1}{1000} + \frac{1}{1000})}} = 2.15`$
- p-value = $`2 \times P(Z > 2.15) = 0.032`$
- Conclusion: Reject $`H_0`$ at $`\alpha = 0.05`$

**Confidence Interval:**
- 95% CI for difference: $`(0.065 - 0.05) \pm 1.96 \times 0.007 = 0.015 \pm 0.014 = [0.001, 0.029]`$
- We are 95% confident that Design B increases conversion rate by 0.1% to 2.9%

### Medical Research Example

Testing a new drug for blood pressure reduction:
- **Placebo**: $`n_1 = 50`$, $`\bar{x}_1 = 140`$ mmHg, $`s_1 = 15`$
- **Drug**: $`n_2 = 50`$, $`\bar{x}_2 = 130`$ mmHg, $`s_2 = 12`$

**Hypothesis Test:**
- $`H_0: \mu_1 = \mu_2`$ vs $`H_1: \mu_1 > \mu_2`$
- $`t = \frac{140 - 130}{\sqrt{\frac{225}{50} + \frac{144}{50}}} = \frac{10}{\sqrt{7.38}} = 3.68`$
- p-value ≈ 0.0002
- Conclusion: Reject $`H_0`$ at $`\alpha = 0.05`$

**Effect Size:**
- $`s_{pooled} = \sqrt{\frac{49 \times 225 + 49 \times 144}{98}} = 13.6`$
- $`d = \frac{10}{13.6} = 0.74`$ (large effect)

### Quality Control Example

Monitoring manufacturing process:
- **Target**: Mean weight = 100g, SD = 5g
- **Sample**: $`n = 25`$, $`\bar{x} = 102g`$, $`s = 4.5g`$

**Hypothesis Test:**
- $`H_0: \mu = 100`$ vs $`H_1: \mu \neq 100`$
- $`t = \frac{102 - 100}{4.5/\sqrt{25}} = 2.22`$
- p-value ≈ 0.036
- Conclusion: Reject $`H_0`$ at $`\alpha = 0.05`$

**Confidence Interval:**
- 95% CI: $`102 \pm 2.064 \times \frac{4.5}{\sqrt{25}} = 102 \pm 1.86 = [100.14, 103.86]`$
- Process mean is significantly different from target

## Practice Problems

### Problem 1: Hypothesis Testing Implementation

**Objective**: Create comprehensive functions for hypothesis testing with proper reporting.

**Tasks**:
1. Implement one-sample t-test with effect size calculation
2. Implement two-sample t-test (independent and paired)
3. Add confidence interval calculation
4. Create standardized output format with all relevant statistics
5. Include power analysis capabilities

**Example Implementation**:
```python
def one_sample_ttest(data, mu0, alpha=0.05, alternative='two-sided'):
    """
    Perform one-sample t-test with comprehensive output.
    
    Returns: test_statistic, p_value, effect_size, confidence_interval, decision
    """
    # Implementation here
```

### Problem 2: Power Analysis

**Objective**: Implement power analysis for various test types.

**Tasks**:
1. Calculate power for given sample size and effect size
2. Determine required sample size for desired power
3. Create power curves for different effect sizes
4. Implement power analysis for t-tests, proportions, and correlations
5. Add visualization of power curves

### Problem 3: Multiple Testing Correction

**Objective**: Implement and compare multiple testing correction methods.

**Tasks**:
1. Implement Bonferroni correction
2. Implement Benjamini-Hochberg FDR control
3. Implement Holm's step-down procedure
4. Compare methods on simulated data
5. Create visualization of correction effects

### Problem 4: Effect Size Analysis

**Objective**: Calculate and interpret various effect size measures.

**Tasks**:
1. Implement Cohen's d for different scenarios
2. Calculate eta-squared for ANOVA
3. Compute correlation-based effect sizes
4. Create effect size interpretation guidelines
5. Add confidence intervals for effect sizes

### Problem 5: Real-World Data Analysis

**Objective**: Apply statistical inference to real datasets.

**Tasks**:
1. Choose a dataset (medical, business, social science)
2. Formulate relevant hypotheses
3. Conduct appropriate statistical tests
4. Calculate effect sizes and confidence intervals
5. Write comprehensive analysis report

## Further Reading

### Books
- **"Statistical Inference"** by George Casella and Roger L. Berger
- **"The Practice of Statistics"** by David S. Moore
- **"Statistics in Plain English"** by Timothy C. Urdan
- **"Multiple Testing Procedures"** by Jason Hsu
- **"Applied Linear Statistical Models"** by Kutner et al.

### Online Resources
- **StatQuest**: YouTube channel with clear statistical explanations
- **Khan Academy**: Statistics and probability courses
- **Coursera**: Statistics with R Specialization
- **edX**: Statistical Learning

### Advanced Topics
- **Bayesian Hypothesis Testing**: Alternative to frequentist methods
- **Nonparametric Tests**: When assumptions are violated
- **Bootstrap Methods**: Resampling-based inference
- **Sequential Testing**: Adaptive experimental designs
- **Meta-Analysis**: Combining results from multiple studies

## Key Takeaways

### Fundamental Concepts
- **Hypothesis testing** provides a framework for making decisions about population parameters
- **P-values** measure evidence against the null hypothesis, not probability of hypothesis being true
- **Confidence intervals** provide a range of plausible values for population parameters
- **Multiple testing** requires correction to control false positive rates
- **Effect sizes** complement p-values by measuring practical significance
- **Type I and Type II errors** are fundamental concepts in statistical decision making

### Mathematical Tools
- **Test statistics** follow known distributions under null hypothesis
- **Sampling distributions** provide the foundation for inference
- **Standard errors** quantify uncertainty in estimates
- **Critical values** determine rejection regions
- **Power analysis** helps design studies with adequate sensitivity

### Applications
- **A/B testing** uses hypothesis testing to evaluate interventions
- **Medical research** relies on statistical inference for treatment evaluation
- **Quality control** uses statistical methods to monitor processes
- **Machine learning** uses statistical inference for model validation
- **Business analytics** applies inference for data-driven decisions

### Best Practices
- **Always report effect sizes** along with p-values
- **Use confidence intervals** to quantify uncertainty
- **Consider multiple testing** when conducting multiple comparisons
- **Check assumptions** before applying tests
- **Interpret results** in context of practical significance

### Next Steps
In the following chapters, we'll build on these inferential foundations to explore:
- **Regression Analysis**: Modeling relationships between variables
- **Analysis of Variance**: Comparing means across multiple groups
- **Nonparametric Methods**: When assumptions are violated
- **Time Series Analysis**: Modeling temporal dependencies
- **Advanced Topics**: Specialized methods for complex data structures

Remember that statistical inference is not just about calculating p-values—it's about making informed decisions in the face of uncertainty. The methods and concepts covered in this chapter provide the foundation for rigorous data analysis and evidence-based decision making. 