# Bayesian Statistics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![PyMC3](https://img.shields.io/badge/PyMC3-3.11+-blue.svg)](https://docs.pymc.io/)
[![ArviZ](https://img.shields.io/badge/ArviZ-0.12+-orange.svg)](https://python.arviz.org/)

## Introduction

Bayesian statistics provides a framework for updating beliefs with data. This chapter covers Bayesian inference, MCMC methods, and their applications in AI/ML.

### Why Bayesian Statistics Matters

Bayesian statistics offers a coherent framework for learning from data that naturally incorporates uncertainty and prior knowledge. It helps us:

1. **Update Beliefs**: Systematically combine prior knowledge with new evidence
2. **Quantify Uncertainty**: Provide probability statements about parameters and predictions
3. **Incorporate Prior Information**: Use domain knowledge to improve inference
4. **Model Comparison**: Compare competing models using evidence
5. **Decision Making**: Make optimal decisions under uncertainty

### Bayesian vs Frequentist Philosophy

**Frequentist Approach:**
- Parameters are fixed, unknown constants
- Probability is long-run frequency
- Inference based on sampling distribution
- Confidence intervals: "95% of intervals contain true parameter"

**Bayesian Approach:**
- Parameters are random variables with distributions
- Probability is degree of belief
- Inference based on posterior distribution
- Credible intervals: "95% probability parameter is in interval"

### Intuitive Example: Medical Diagnosis

Consider a medical test for a disease:
- **Prior**: 1% of population has disease
- **Test sensitivity**: 95% (P(positive|disease))
- **Test specificity**: 90% (P(negative|no disease))
- **Question**: Given positive test, what's probability of disease?

**Bayesian solution**: Update prior with likelihood to get posterior probability.

## Table of Contents
- [Bayesian Inference Fundamentals](#bayesian-inference-fundamentals)
- [Conjugate Priors](#conjugate-priors)
- [MCMC Methods](#mcmc-methods)
- [Bayesian Regression](#bayesian-regression)
- [Model Comparison](#model-comparison)
- [Practical Applications](#practical-applications)

## Setup

The examples in this chapter use Python libraries for Bayesian analysis, particularly PyMC3 for probabilistic programming and ArviZ for diagnostics and visualization.

## Bayesian Inference Fundamentals

Bayesian inference provides a coherent framework for updating beliefs about parameters based on observed data, combining prior knowledge with new evidence.

### Understanding Bayesian Inference

Think of Bayesian inference as a learning process where you start with initial beliefs (prior) and update them with new information (data) to arrive at updated beliefs (posterior).

#### Intuitive Example: Coin Flipping

Consider flipping a coin to determine if it's fair:
- **Prior belief**: Coin is likely fair (θ ≈ 0.5)
- **Data**: 7 heads in 10 flips
- **Posterior**: Updated belief about θ given the data
- **Result**: Probability distribution over possible values of θ

### Mathematical Foundation

**Bayes' Theorem:**
```math
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
```

Where:
- $`P(\theta | D)`$ = **Posterior distribution** (updated belief about θ given data)
- $`P(D | \theta)`$ = **Likelihood function** (probability of data given θ)
- $`P(\theta)`$ = **Prior distribution** (initial belief about θ)
- $`P(D)`$ = **Evidence/Marginal likelihood** (normalizing constant)

**Continuous Case:**
```math
f(\theta | D) = \frac{f(D | \theta) f(\theta)}{\int f(D | \theta) f(\theta) d\theta}
```

**Log-Posterior:**
```math
\log f(\theta | D) = \log f(D | \theta) + \log f(\theta) - \log \int f(D | \theta) f(\theta) d\theta
```

#### Example: Normal Mean Estimation

**Data**: $`X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)`$ (σ² known)
**Prior**: $`\mu \sim N(\mu_0, \tau_0^2)`$
**Likelihood**: $`f(D | \mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)`$

**Posterior**: $`\mu | D \sim N(\mu_n, \tau_n^2)`$

Where:
```math
\mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}
```

```math
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
```

### Prior Distributions

The prior distribution represents our initial beliefs about parameters before seeing the data.

#### Types of Priors

**1. Informative Priors:**
Based on previous studies, expert knowledge, or theoretical considerations.

**Example**: Drug efficacy study
- **Prior**: Based on previous clinical trials
- **Mean**: 0.3 (30% improvement)
- **Standard deviation**: 0.1 (uncertainty)

**2. Weakly Informative Priors:**
Provide some structure but are not strongly constraining.

**Example**: Normal prior with large variance
- **Prior**: $`\mu \sim N(0, 100)`$
- **Interpretation**: Weak belief centered at 0

**3. Non-Informative Priors:**
Designed to have minimal influence on the posterior.

**Example**: Uniform prior
- **Prior**: $`f(\theta) \propto 1`$ for $`\theta \in [a, b]`$
- **Interpretation**: Equal probability for all values

#### Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior belongs to the same family as the prior.

**Common Conjugate Pairs:**

**1. Normal-Normal:**
- Likelihood: $`X_i \sim N(\mu, \sigma^2)`$ (σ² known)
- Prior: $`\mu \sim N(\mu_0, \tau_0^2)`$
- Posterior: $`\mu | D \sim N(\mu_n, \tau_n^2)`$

Where:
```math
\mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}
```

```math
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
```

**2. Beta-Binomial:**
- Likelihood: $`X \sim \text{Binomial}(n, \theta)`$
- Prior: $`\theta \sim \text{Beta}(\alpha, \beta)`$
- Posterior: $`\theta | D \sim \text{Beta}(\alpha + x, \beta + n - x)`$

**Example**: Coin flipping
- **Prior**: $`\theta \sim \text{Beta}(2, 2)`$ (slightly favoring fair coin)
- **Data**: 7 heads in 10 flips
- **Posterior**: $`\theta | D \sim \text{Beta}(9, 5)`$

**3. Gamma-Poisson:**
- Likelihood: $`X_i \sim \text{Poisson}(\lambda)`$
- Prior: $`\lambda \sim \text{Gamma}(\alpha, \beta)`$
- Posterior: $`\lambda | D \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)`$

**Example**: Accident rate estimation
- **Prior**: $`\lambda \sim \text{Gamma}(2, 1)`$ (mean = 2 accidents/year)
- **Data**: 15 accidents in 5 years
- **Posterior**: $`\lambda | D \sim \text{Gamma}(17, 6)`$

**4. Inverse Gamma-Normal:**
- Likelihood: $`X_i \sim N(\mu, \sigma^2)`$ (μ known)
- Prior: $`\sigma^2 \sim \text{InvGamma}(\alpha, \beta)`$
- Posterior: $`\sigma^2 | D \sim \text{InvGamma}(\alpha + n/2, \beta + \frac{1}{2}\sum(x_i - \mu)^2)`$

#### Non-Informative Priors

**1. Jeffreys Prior:**
```math
f(\theta) \propto \sqrt{I(\theta)}
```

Where $`I(\theta)`$ is the Fisher information:
```math
I(\theta) = -E\left[\frac{\partial^2}{\partial \theta^2} \log f(X | \theta)\right]
```

**2. Uniform Prior:**
```math
f(\theta) \propto 1
```

**3. Reference Prior:**
Maximizes the expected Kullback-Leibler divergence between prior and posterior.

### Posterior Analysis

Once we have the posterior distribution, we can extract various summaries and make inferences.

#### Point Estimates

**Posterior Mean (Bayes Estimator):**
```math
\hat{\theta}_{Bayes} = E[\theta | D] = \int \theta f(\theta | D) d\theta
```

**Posterior Mode (Maximum A Posteriori):**
```math
\hat{\theta}_{MAP} = \arg\max_{\theta} f(\theta | D)
```

**Posterior Median:**
```math
\hat{\theta}_{median} = \text{median of } f(\theta | D)
```

#### Uncertainty Quantification

**Posterior Variance:**
```math
\text{Var}(\theta | D) = E[(\theta - \hat{\theta}_{Bayes})^2 | D]
```

**Posterior Standard Deviation:**
```math
\sigma_{\theta|D} = \sqrt{\text{Var}(\theta | D)}
```

#### Credible Intervals

A $`(1-\alpha)`$ credible interval satisfies:
```math
P(\theta_L \leq \theta \leq \theta_U | D) = 1 - \alpha
```

**Highest Posterior Density (HPD) Interval:**
The shortest interval containing $`(1-\alpha)`$ of the posterior probability.

**Equal-Tailed Interval:**
The interval where $`P(\theta < \theta_L | D) = \alpha/2`$ and $`P(\theta > \theta_U | D) = \alpha/2`$.

#### Example: Normal Mean with Conjugate Prior

**Data**: $`X_1, \ldots, X_{10} \sim N(\mu, 4)`$
**Sample mean**: $`\bar{x} = 5.2`$
**Prior**: $`\mu \sim N(0, 9)`$

**Posterior**: $`\mu | D \sim N(4.68, 0.36)`$

**95% Credible Interval**: [3.91, 5.45]

### Predictive Distributions

Bayesian prediction naturally incorporates parameter uncertainty.

#### Posterior Predictive Distribution

For new observation $`X_{new}`$:
```math
f(x_{new} | D) = \int f(x_{new} | \theta) f(\theta | D) d\theta
```

**Example**: Normal case
If $`X_{new} | \mu \sim N(\mu, \sigma^2)`$ and $`\mu | D \sim N(\mu_n, \tau_n^2)`$, then:
```math
X_{new} | D \sim N(\mu_n, \sigma^2 + \tau_n^2)
```

#### Prior Predictive Distribution

Before seeing any data:
```math
f(x) = \int f(x | \theta) f(\theta) d\theta
```

### Model Comparison

Bayesian model comparison uses the evidence (marginal likelihood) to compare models.

#### Bayes Factor

For comparing models $`M_1`$ and $`M_2`$:
```math
BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int f(D | \theta_1, M_1) f(\theta_1 | M_1) d\theta_1}{\int f(D | \theta_2, M_2) f(\theta_2 | M_2) d\theta_2}
```

**Interpretation**:
- $`BF_{12} > 1`$: Evidence favors $`M_1`$
- $`BF_{12} < 1`$: Evidence favors $`M_2`$
- $`BF_{12} = 1`$: Equal evidence for both models

#### Posterior Model Probabilities

```math
P(M_i | D) = \frac{P(D | M_i) P(M_i)}{\sum_j P(D | M_j) P(M_j)}
```

Where $`P(M_i)`$ is the prior probability of model $`M_i`$.

### Computational Methods

When analytical solutions are not available, we use computational methods.

#### 1. Markov Chain Monte Carlo (MCMC)

**Metropolis-Hastings Algorithm:**
- Propose new parameter values
- Accept/reject based on acceptance ratio
- Converges to posterior distribution

**Gibbs Sampling:**
- Sample each parameter conditional on others
- Useful for high-dimensional problems
- Requires known conditional distributions

**Hamiltonian Monte Carlo:**
- Uses gradient information for efficient sampling
- Particularly effective for continuous parameters
- Implemented in PyMC3 and Stan

#### 2. Variational Inference

Approximate the posterior with a simpler distribution:
```math
q(\theta) \approx f(\theta | D)
```

**Advantages**: Fast, scalable
**Disadvantages**: Approximate, may miss posterior structure

#### 3. Laplace Approximation

Approximate the posterior as a normal distribution around the MAP:
```math
f(\theta | D) \approx N(\hat{\theta}_{MAP}, [H(\hat{\theta}_{MAP})]^{-1})
```

Where $`H(\theta)`$ is the Hessian of the log-posterior. 

## Conjugate Priors

Conjugate priors provide analytical solutions for posterior distributions, making Bayesian inference computationally tractable.

### Understanding Conjugate Priors

A prior distribution is **conjugate** to a likelihood function if the posterior distribution belongs to the same family as the prior. This property simplifies calculations and provides closed-form solutions.

#### Intuitive Example: Beta-Binomial

Consider estimating a probability θ from binomial data:
- **Likelihood**: $`X \sim \text{Binomial}(n, \theta)`$
- **Prior**: $`\theta \sim \text{Beta}(\alpha, \beta)`$
- **Posterior**: $`\theta | D \sim \text{Beta}(\alpha + x, \beta + n - x)`$

The Beta distribution is conjugate to the Binomial likelihood because the posterior is also Beta.

### Common Conjugate Prior Families

#### 1. Normal-Normal (Mean Unknown, Variance Known)

**Likelihood**: $`X_i \sim N(\mu, \sigma^2)`$ where σ² is known
**Prior**: $`\mu \sim N(\mu_0, \tau_0^2)`$
**Posterior**: $`\mu | D \sim N(\mu_n, \tau_n^2)`$

**Parameters**:
```math
\mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}
```

```math
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
```

**Interpretation**:
- Posterior mean is weighted average of prior mean and sample mean
- Weights depend on prior precision and data precision
- Posterior precision is sum of prior and data precisions

**Example**: IQ Testing
- **Prior**: $`\mu \sim N(100, 25)`$ (mean IQ = 100, uncertainty = 5)
- **Data**: Sample of 20 students, mean = 105, σ = 15
- **Posterior**: $`\mu | D \sim N(104.2, 4.8)`$

#### 2. Beta-Binomial (Proportion)

**Likelihood**: $`X \sim \text{Binomial}(n, \theta)`$
**Prior**: $`\theta \sim \text{Beta}(\alpha, \beta)`$
**Posterior**: $`\theta | D \sim \text{Beta}(\alpha + x, \beta + n - x)`$

**Interpretation**:
- α + β represents prior sample size
- α represents prior successes
- β represents prior failures

**Example**: Coin Flipping
- **Prior**: $`\theta \sim \text{Beta}(2, 2)`$ (slightly favoring fair coin)
- **Data**: 7 heads in 10 flips
- **Posterior**: $`\theta | D \sim \text{Beta}(9, 5)`$
- **Posterior mean**: $`\frac{9}{14} = 0.643`$

#### 3. Gamma-Poisson (Rate)

**Likelihood**: $`X_i \sim \text{Poisson}(\lambda)`$
**Prior**: $`\lambda \sim \text{Gamma}(\alpha, \beta)`$
**Posterior**: $`\lambda | D \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)`$

**Interpretation**:
- α represents prior number of events
- β represents prior time period
- Posterior mean: $`\frac{\alpha + \sum x_i}{\beta + n}`$

**Example**: Accident Rate
- **Prior**: $`\lambda \sim \text{Gamma}(2, 1)`$ (2 accidents per year)
- **Data**: 15 accidents in 5 years
- **Posterior**: $`\lambda | D \sim \text{Gamma}(17, 6)`$
- **Posterior mean**: $`\frac{17}{6} = 2.83`$ accidents per year

#### 4. Inverse Gamma-Normal (Variance Unknown, Mean Known)

**Likelihood**: $`X_i \sim N(\mu, \sigma^2)`$ where μ is known
**Prior**: $`\sigma^2 \sim \text{InvGamma}(\alpha, \beta)`$
**Posterior**: $`\sigma^2 | D \sim \text{InvGamma}(\alpha + n/2, \beta + \frac{1}{2}\sum(x_i - \mu)^2)`$

**Example**: Measurement Precision
- **Prior**: $`\sigma^2 \sim \text{InvGamma}(3, 10)`$
- **Data**: 10 measurements with known mean μ = 100
- **Sum of squares**: $`\sum(x_i - 100)^2 = 50`$
- **Posterior**: $`\sigma^2 | D \sim \text{InvGamma}(8, 35)`$

#### 5. Normal-Inverse Gamma (Both Mean and Variance Unknown)

**Likelihood**: $`X_i \sim N(\mu, \sigma^2)`$
**Prior**: $`\mu | \sigma^2 \sim N(\mu_0, \sigma^2/\kappa_0)`$ and $`\sigma^2 \sim \text{InvGamma}(\alpha_0, \beta_0)`$
**Posterior**: 
- $`\mu | \sigma^2, D \sim N(\mu_n, \sigma^2/\kappa_n)`$
- $`\sigma^2 | D \sim \text{InvGamma}(\alpha_n, \beta_n)`$

Where:
```math
\mu_n = \frac{\kappa_0 \mu_0 + n\bar{x}}{\kappa_0 + n}
```

```math
\kappa_n = \kappa_0 + n
```

```math
\alpha_n = \alpha_0 + n/2
```

```math
\beta_n = \beta_0 + \frac{1}{2}\sum(x_i - \bar{x})^2 + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2(\kappa_0 + n)}
```

### Multivariate Conjugate Priors

#### Normal-Normal (Multivariate)

**Likelihood**: $`\mathbf{X}_i \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$ where $`\mathbf{\Sigma}`$ is known
**Prior**: $`\mathbf{\mu} \sim N_p(\mathbf{\mu}_0, \mathbf{\Sigma}_0)`$
**Posterior**: $`\mathbf{\mu} | D \sim N_p(\mathbf{\mu}_n, \mathbf{\Sigma}_n)`$

Where:
```math
\mathbf{\Sigma}_n^{-1} = \mathbf{\Sigma}_0^{-1} + n\mathbf{\Sigma}^{-1}
```

```math
\mathbf{\mu}_n = \mathbf{\Sigma}_n(\mathbf{\Sigma}_0^{-1}\mathbf{\mu}_0 + n\mathbf{\Sigma}^{-1}\bar{\mathbf{x}})
```

#### Wishart-Normal (Covariance Matrix)

**Likelihood**: $`\mathbf{X}_i \sim N_p(\mathbf{\mu}, \mathbf{\Sigma})`$ where $`\mathbf{\mu}`$ is known
**Prior**: $`\mathbf{\Sigma} \sim \text{Wishart}(\nu_0, \mathbf{V}_0)`$
**Posterior**: $`\mathbf{\Sigma} | D \sim \text{Wishart}(\nu_0 + n, \mathbf{V}_0 + \mathbf{S})`$

Where $`\mathbf{S} = \sum_{i=1}^n (\mathbf{x}_i - \mathbf{\mu})(\mathbf{x}_i - \mathbf{\mu})^T`$.

## MCMC Methods

Markov Chain Monte Carlo (MCMC) methods enable sampling from complex posterior distributions when analytical solutions are not available.

### Understanding MCMC

MCMC methods construct a Markov chain that converges to the target posterior distribution. The key insight is that we can learn about a distribution by sampling from it.

#### Intuitive Example: Exploring a Landscape

Think of the posterior as a landscape:
- **Goal**: Visit locations proportional to their height (probability)
- **Method**: Take random steps, accepting uphill moves and sometimes downhill moves
- **Result**: After many steps, time spent at each location ≈ posterior probability

### Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is the most general MCMC method.

#### Algorithm Steps

1. **Initialize**: Start at $`\theta^{(0)}`$
2. **Propose**: Generate candidate $`\theta^*`$ from proposal distribution $`q(\theta^* | \theta^{(t)})`$
3. **Accept/Reject**: Calculate acceptance probability:
```math
\alpha = \min\left(1, \frac{f(\theta^* | D) q(\theta^{(t)} | \theta^*)}{f(\theta^{(t)} | D) q(\theta^* | \theta^{(t)})}\right)
```
4. **Update**: $`\theta^{(t+1)} = \theta^*`$ with probability α, otherwise $`\theta^{(t+1)} = \theta^{(t)}`$
5. **Repeat**: Return to step 2

#### Example: Normal Mean Estimation

**Target**: $`f(\mu | D) \propto \exp\left(-\frac{n(\mu - \bar{x})^2}{2\sigma^2}\right)`$
**Proposal**: $`q(\mu^* | \mu^{(t)}) = N(\mu^{(t)}, \tau^2)`$
**Acceptance ratio**: $`\alpha = \min\left(1, \exp\left(-\frac{n}{2\sigma^2}[(\mu^* - \bar{x})^2 - (\mu^{(t)} - \bar{x})^2]\right)\right)`$

#### Tuning Parameters

**Proposal Variance**: 
- Too small: Slow mixing, high autocorrelation
- Too large: Low acceptance rate
- Optimal: 20-50% acceptance rate

**Burn-in Period**: Initial samples to discard while chain converges
**Thinning**: Keep every k-th sample to reduce autocorrelation

### Gibbs Sampling

Gibbs sampling is useful when we can sample from conditional distributions.

#### Algorithm

For parameters $`\theta = (\theta_1, \ldots, \theta_p)`$:

1. **Initialize**: $`\theta^{(0)} = (\theta_1^{(0)}, \ldots, \theta_p^{(0)})`$
2. **Sample**: For each i = 1, ..., p:
```math
\theta_i^{(t+1)} \sim f(\theta_i | \theta_1^{(t+1)}, \ldots, \theta_{i-1}^{(t+1)}, \theta_{i+1}^{(t)}, \ldots, \theta_p^{(t)}, D)
```
3. **Repeat**: Return to step 2

#### Example: Normal-Normal Model

**Model**: $`X_i \sim N(\mu, \sigma^2)`$, $`\mu \sim N(\mu_0, \tau_0^2)`$, $`\sigma^2 \sim \text{InvGamma}(\alpha_0, \beta_0)`$

**Conditional distributions**:
- $`\mu | \sigma^2, D \sim N(\mu_n, \sigma^2/\kappa_n)`$
- $`\sigma^2 | \mu, D \sim \text{InvGamma}(\alpha_0 + n/2, \beta_0 + \frac{1}{2}\sum(x_i - \mu)^2)`$

### Hamiltonian Monte Carlo (HMC)

HMC uses gradient information to propose more efficient moves.

#### Intuition

Think of a particle moving on a frictionless surface:
- **Position**: Current parameter values
- **Momentum**: Gradient information
- **Energy**: Negative log-posterior
- **Trajectory**: Hamiltonian dynamics

#### Algorithm

1. **Initialize**: $`(\theta^{(0)}, \mathbf{p}^{(0)})`$
2. **Leapfrog**: Integrate Hamiltonian dynamics for L steps
3. **Accept/Reject**: Based on energy conservation
4. **Repeat**: Return to step 1

#### Advantages

- **Efficient**: Uses gradient information
- **Scalable**: Works well in high dimensions
- **No tuning**: Automatic step size adaptation

### MCMC Diagnostics

#### Convergence Diagnostics

**1. Trace Plots**: Visualize parameter chains over time
**2. Gelman-Rubin Statistic**: Compare multiple chains
**3. Effective Sample Size**: Measure of independent samples
**4. Autocorrelation**: Measure of chain dependence

#### Example: Convergence Assessment

**Multiple chains**: Run 4 chains from different starting points
**Gelman-Rubin**: $`\hat{R} < 1.1`$ indicates convergence
**Effective sample size**: Should be > 1000 for reliable inference

## Bayesian Regression

Bayesian regression naturally incorporates uncertainty in both parameters and predictions.

### Linear Regression

#### Model Specification

**Likelihood**: $`Y_i \sim N(\mathbf{x}_i^T \mathbf{\beta}, \sigma^2)`$
**Prior**: $`\mathbf{\beta} \sim N(\mathbf{\beta}_0, \mathbf{\Sigma}_0)`$, $`\sigma^2 \sim \text{InvGamma}(\alpha_0, \beta_0)`$

#### Posterior Distribution

**Conjugate case** (normal-inverse gamma prior):
- $`\mathbf{\beta} | \sigma^2, D \sim N(\mathbf{\beta}_n, \sigma^2(\mathbf{X}^T\mathbf{X} + \mathbf{\Sigma}_0^{-1})^{-1})`$
- $`\sigma^2 | D \sim \text{InvGamma}(\alpha_n, \beta_n)`$

Where:
```math
\mathbf{\beta}_n = (\mathbf{X}^T\mathbf{X} + \mathbf{\Sigma}_0^{-1})^{-1}(\mathbf{X}^T\mathbf{y} + \mathbf{\Sigma}_0^{-1}\mathbf{\beta}_0)
```

#### Prediction

For new observation $`\mathbf{x}_{new}`$:
```math
Y_{new} | D \sim N(\mathbf{x}_{new}^T \mathbf{\beta}_n, \sigma^2 + \mathbf{x}_{new}^T(\mathbf{X}^T\mathbf{X} + \mathbf{\Sigma}_0^{-1})^{-1}\mathbf{x}_{new})
```

### Logistic Regression

#### Model Specification

**Likelihood**: $`Y_i \sim \text{Bernoulli}(\pi_i)`$ where $`\pi_i = \frac{1}{1 + e^{-\mathbf{x}_i^T \mathbf{\beta}}}`$
**Prior**: $`\mathbf{\beta} \sim N(\mathbf{0}, \sigma^2\mathbf{I})`$

#### Posterior

No conjugate prior exists, so we use MCMC:
```math
f(\mathbf{\beta} | D) \propto \prod_{i=1}^n \pi_i^{y_i}(1-\pi_i)^{1-y_i} \exp\left(-\frac{1}{2\sigma^2}\mathbf{\beta}^T\mathbf{\beta}\right)
```

### Hierarchical Models

Hierarchical models allow for partial pooling between groups.

#### Example: School Effects

**Model**:
- $`Y_{ij} \sim N(\mu_j, \sigma^2)`$ (student i in school j)
- $`\mu_j \sim N(\mu_0, \tau^2)`$ (school effects)
- $`\mu_0 \sim N(0, 100)`$ (overall mean)
- $`\sigma^2, \tau^2 \sim \text{InvGamma}(1, 1)`$ (variances)

**Interpretation**:
- Complete pooling: All schools have same mean
- No pooling: Each school estimated independently
- Partial pooling: Shrinkage toward overall mean

## Model Comparison

Bayesian model comparison uses the evidence to compare competing models.

### Bayes Factors

#### Definition

For models $`M_1`$ and $`M_2`$:
```math
BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int f(D | \theta_1, M_1) f(\theta_1 | M_1) d\theta_1}{\int f(D | \theta_2, M_2) f(\theta_2 | M_2) d\theta_2}
```

#### Interpretation

**Jeffreys' Scale**:
- $`BF_{12} > 100`$: Decisive evidence for $`M_1`$
- $`10 < BF_{12} < 100`$: Strong evidence for $`M_1`$
- $`3 < BF_{12} < 10`$: Moderate evidence for $`M_1`$
- $`1 < BF_{12} < 3`$: Weak evidence for $`M_1`$
- $`BF_{12} = 1`$: Equal evidence

#### Example: Model Selection

**Model 1**: Linear regression $`Y = \beta_0 + \beta_1 X`$
**Model 2**: Quadratic regression $`Y = \beta_0 + \beta_1 X + \beta_2 X^2`$
**Data**: 50 observations
**Bayes factor**: $`BF_{21} = 15.3`$ (moderate evidence for quadratic)

### Information Criteria

#### Deviance Information Criterion (DIC)

```math
DIC = \bar{D} + p_D
```

Where:
- $`\bar{D}`$ = mean deviance over posterior samples
- $`p_D`$ = effective number of parameters

#### Widely Applicable Information Criterion (WAIC)

```math
WAIC = -2\sum_{i=1}^n \log \left(\frac{1}{S}\sum_{s=1}^S f(y_i | \theta^{(s)})\right) + 2\sum_{i=1}^n \text{Var}(\log f(y_i | \theta))
```

#### Leave-One-Out Cross-Validation (LOO-CV)

```math
LOO = \sum_{i=1}^n \log f(y_i | y_{-i})
```

Where $`f(y_i | y_{-i})`$ is the predictive density for observation i given all other observations.

### Model Averaging

Instead of selecting one model, average over multiple models:

```math
f(y_{new} | D) = \sum_{k=1}^K f(y_{new} | D, M_k) P(M_k | D)
```

**Advantages**:
- Accounts for model uncertainty
- More robust predictions
- Avoids overfitting to single model

## Practical Applications

### Bayesian A/B Testing

#### Problem Setup

Compare two variants (A and B) to determine which performs better.

**Model**:
- $`X_A \sim \text{Binomial}(n_A, \theta_A)`$
- $`X_B \sim \text{Binomial}(n_B, \theta_B)`$
- $`\theta_A, \theta_B \sim \text{Beta}(1, 1)`$ (uniform priors)

#### Analysis

**Posterior distributions**:
- $`\theta_A | D \sim \text{Beta}(1 + x_A, 1 + n_A - x_A)`$
- $`\theta_B | D \sim \text{Beta}(1 + x_B, 1 + n_B - x_B)`$

**Probability B is better**:
```math
P(\theta_B > \theta_A | D) = \int_0^1 \int_0^{\theta_B} f(\theta_A | D) f(\theta_B | D) d\theta_A d\theta_B
```

#### Example: Website Conversion

**Variant A**: 150 conversions out of 1000 visitors
**Variant B**: 180 conversions out of 1000 visitors
**Result**: $`P(\theta_B > \theta_A | D) = 0.92`$ (92% probability B is better)

### Medical Diagnosis

#### Problem Setup

Estimate disease probability given test results.

**Model**:
- $`D`$ = disease status (0/1)
- $`T`$ = test result (0/1)
- $`P(D=1) = \pi`$ (prevalence)
- $`P(T=1 | D=1) = \text{sens}`$ (sensitivity)
- $`P(T=0 | D=0) = \text{spec}`$ (specificity)

#### Bayesian Analysis

**Prior**: $`\pi, \text{sens}, \text{spec} \sim \text{Beta}(1, 1)`$
**Posterior**: Update with test data
**Result**: Posterior probability of disease given test result

### Recommendation Systems

#### Collaborative Filtering

**Model**: User-item ratings matrix
**Latent factors**: User preferences, item characteristics
**Bayesian approach**: Uncertainty in latent factors

**Example**: Movie recommendations
- **Users**: 1000 users
- **Movies**: 500 movies
- **Latent factors**: 20 dimensions
- **Result**: Personalized recommendations with uncertainty

### Time Series Forecasting

#### Bayesian Structural Time Series

**Components**:
- **Trend**: $`\mu_t = \mu_{t-1} + \delta_{t-1} + \eta_t`$
- **Seasonal**: $`\gamma_t = -\sum_{s=1}^{S-1} \gamma_{t-s} + \omega_t`$
- **Regression**: $`\mathbf{x}_t^T \mathbf{\beta}`$

**Advantages**:
- Uncertainty quantification
- Automatic feature selection
- Handling of missing data

## Practice Problems

### Problem 1: Bayesian Inference

**Objective**: Implement Bayesian updating for different likelihood-prior combinations.

**Tasks**:
1. Create conjugate prior-likelihood pairs (normal-normal, beta-binomial, gamma-poisson)
2. Implement posterior calculation functions
3. Add visualization tools for prior, likelihood, and posterior
4. Include credible interval calculation
5. Add predictive distribution sampling

**Example Implementation**:
```python
def bayesian_update(likelihood, prior, data):
    """
    Perform Bayesian updating for conjugate pairs.
    
    Returns: posterior parameters, credible intervals, predictions
    """
    # Implementation here
```

### Problem 2: MCMC Diagnostics

**Objective**: Create comprehensive MCMC diagnostic tools.

**Tasks**:
1. Implement trace plot visualization
2. Add Gelman-Rubin convergence diagnostics
3. Calculate effective sample sizes
4. Create autocorrelation plots
5. Include Geweke diagnostics for stationarity

### Problem 3: Model Comparison

**Objective**: Build Bayesian model comparison frameworks.

**Tasks**:
1. Implement Bayes factor calculation
2. Add information criteria (DIC, WAIC, LOO-CV)
3. Create model averaging methods
4. Include posterior predictive checks
5. Add cross-validation utilities

### Problem 4: Hierarchical Models

**Objective**: Implement hierarchical Bayesian models.

**Tasks**:
1. Create multi-level regression models
2. Add partial pooling implementations
3. Implement random effects models
4. Include shrinkage estimation
5. Add model comparison for hierarchical structures

### Problem 5: Real-World Bayesian Analysis

**Objective**: Apply Bayesian methods to real datasets.

**Tasks**:
1. Choose dataset (medical, marketing, scientific)
2. Perform exploratory data analysis
3. Specify appropriate Bayesian models
4. Conduct MCMC sampling and diagnostics
5. Write comprehensive analysis report

## Further Reading

### Books
- **"Bayesian Data Analysis"** by Andrew Gelman et al.
- **"Doing Bayesian Data Analysis"** by John K. Kruschke
- **"Statistical Rethinking"** by Richard McElreath
- **"Bayesian Methods for Hackers"** by Cameron Davidson-Pilon
- **"Bayesian Analysis with Python"** by Osvaldo Martin

### Online Resources
- **PyMC Documentation**: Comprehensive tutorials and examples
- **Stan User Guide**: Advanced Bayesian modeling
- **ArviZ Documentation**: Diagnostics and visualization
- **Statistical Rethinking Course**: Richard McElreath's course materials

### Advanced Topics
- **Variational Inference**: Fast approximate Bayesian inference
- **Sequential Monte Carlo**: Particle filtering for dynamic models
- **Gaussian Processes**: Non-parametric Bayesian modeling
- **Deep Bayesian Networks**: Bayesian neural networks
- **Causal Inference**: Bayesian approaches to causality

## Key Takeaways

### Fundamental Concepts
- **Bayesian inference** provides a coherent framework for updating beliefs with data
- **Conjugate priors** simplify posterior calculations and have analytical solutions
- **MCMC methods** enable sampling from complex posterior distributions
- **Bayesian regression** naturally incorporates uncertainty in predictions
- **Model comparison** uses information criteria like DIC, WAIC, and LOO-CV
- **Credible intervals** provide intuitive uncertainty quantification
- **Bayesian methods** are particularly valuable for small datasets and complex models
- **Real-world applications** include A/B testing, medical diagnosis, and recommendation systems

### Mathematical Tools
- **Bayes' theorem** provides the foundation for all Bayesian inference
- **Conjugate families** enable analytical posterior calculations
- **MCMC algorithms** sample from complex posterior distributions
- **Information criteria** balance model fit and complexity
- **Predictive distributions** incorporate parameter uncertainty

### Applications
- **A/B testing** uses Bayesian methods for hypothesis testing
- **Medical diagnosis** applies Bayesian inference to clinical decision making
- **Recommendation systems** use Bayesian approaches for personalization
- **Time series forecasting** incorporates uncertainty in predictions
- **Causal inference** uses Bayesian methods for causal discovery

### Best Practices
- **Always check convergence** for MCMC methods
- **Use multiple diagnostics** to assess model fit
- **Consider model uncertainty** in predictions
- **Validate results** with domain knowledge
- **Communicate uncertainty** clearly to stakeholders

### Next Steps
In the following chapters, we'll build on Bayesian foundations to explore:
- **Experimental Design**: Bayesian approaches to experimental planning
- **Nonparametric Methods**: When parametric assumptions are violated
- **Advanced Topics**: Specialized methods for complex data structures
- **Machine Learning**: Bayesian approaches to predictive modeling

Remember that Bayesian statistics is not just a collection of methods—it's a coherent philosophy for learning from data that naturally incorporates uncertainty and prior knowledge. The methods and concepts covered in this chapter provide the foundation for sophisticated data analysis and evidence-based decision making. 