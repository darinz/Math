# Bayesian Statistics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![PyMC3](https://img.shields.io/badge/PyMC3-3.11+-blue.svg)](https://docs.pymc.io/)
[![ArviZ](https://img.shields.io/badge/ArviZ-0.12+-orange.svg)](https://python.arviz.org/)

Bayesian statistics provides a framework for updating beliefs with data. This chapter covers Bayesian inference, MCMC methods, and their applications in AI/ML.

## Table of Contents
- [Bayesian Inference Fundamentals](#bayesian-inference-fundamentals)
- [Conjugate Priors](#conjugate-priors)
- [MCMC Methods](#mcmc-methods)
- [Bayesian Regression](#bayesian-regression)
- [Model Comparison](#model-comparison)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import beta, gamma, norm, bernoulli
import pymc3 as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Bayesian Inference Fundamentals

### Bayes' Theorem Implementation

```python
def bayesian_update_example():
    """Demonstrate Bayesian updating with coin flips"""
    
    # Prior: Beta distribution (conjugate prior for Bernoulli)
    prior_alpha, prior_beta = 2, 2  # Beta(2,2) - slightly favoring heads
    
    # Data: sequence of coin flips
    flips = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # 7 heads, 3 tails
    
    # Calculate posterior
    posterior_alpha = prior_alpha + sum(flips)
    posterior_beta = prior_beta + len(flips) - sum(flips)
    
    # Generate points for plotting
    theta = np.linspace(0, 1, 1000)
    prior_pdf = beta.pdf(theta, prior_alpha, prior_beta)
    posterior_pdf = beta.pdf(theta, posterior_alpha, posterior_beta)
    
    return theta, prior_pdf, posterior_pdf, flips

theta, prior_pdf, posterior_pdf, flips = bayesian_update_example()

print("Bayesian Coin Flip Example")
print(f"Prior: Beta({2}, {2})")
print(f"Data: {flips}")
print(f"Posterior: Beta({2 + sum(flips)}, {2 + len(flips) - sum(flips)})")

# Visualize Bayesian updating
plt.figure(figsize=(15, 5))

# Prior distribution
plt.subplot(1, 3, 1)
plt.plot(theta, prior_pdf, 'b-', linewidth=2, label='Prior')
plt.fill_between(theta, prior_pdf, alpha=0.3, color='blue')
plt.xlabel('θ (Probability of Heads)')
plt.ylabel('Density')
plt.title('Prior Distribution')
plt.legend()

# Likelihood
plt.subplot(1, 3, 2)
n_heads = sum(flips)
n_tails = len(flips) - n_heads
likelihood = theta**n_heads * (1-theta)**n_tails
likelihood = likelihood / np.trapz(likelihood, theta)  # Normalize
plt.plot(theta, likelihood, 'g-', linewidth=2, label='Likelihood')
plt.fill_between(theta, likelihood, alpha=0.3, color='green')
plt.xlabel('θ (Probability of Heads)')
plt.ylabel('Density')
plt.title('Likelihood')
plt.legend()

# Posterior distribution
plt.subplot(1, 3, 3)
plt.plot(theta, posterior_pdf, 'r-', linewidth=2, label='Posterior')
plt.fill_between(theta, posterior_pdf, alpha=0.3, color='red')
plt.xlabel('θ (Probability of Heads)')
plt.ylabel('Density')
plt.title('Posterior Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Credible intervals
def calculate_credible_intervals(alpha, beta, confidence=0.95):
    """Calculate credible intervals for Beta distribution"""
    lower = beta.ppf((1-confidence)/2, alpha, beta)
    upper = beta.ppf((1+confidence)/2, alpha, beta)
    mean = alpha / (alpha + beta)
    mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else mean
    return lower, upper, mean, mode

prior_ci = calculate_credible_intervals(2, 2)
posterior_ci = calculate_credible_intervals(2 + sum(flips), 2 + len(flips) - sum(flips))

print(f"\nCredible Intervals (95%):")
print(f"Prior: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], Mean: {prior_ci[2]:.3f}")
print(f"Posterior: [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], Mean: {posterior_ci[2]:.3f}")
```

## Conjugate Priors

### Common Conjugate Prior Families

```python
def conjugate_prior_examples():
    """Demonstrate common conjugate prior families"""
    
    # 1. Normal-Normal (known variance)
    mu_0, sigma_0 = 0, 2  # Prior mean and std
    sigma = 1  # Known data std
    data = np.random.normal(3, sigma, 10)  # Data
    
    # Posterior parameters
    n = len(data)
    x_bar = np.mean(data)
    mu_post = (mu_0/sigma_0**2 + n*x_bar/sigma**2) / (1/sigma_0**2 + n/sigma**2)
    sigma_post = np.sqrt(1 / (1/sigma_0**2 + n/sigma**2))
    
    # 2. Beta-Bernoulli
    alpha_0, beta_0 = 1, 1  # Prior (uniform)
    bernoulli_data = np.random.binomial(1, 0.7, 20)  # Data
    alpha_post = alpha_0 + sum(bernoulli_data)
    beta_post = beta_0 + len(bernoulli_data) - sum(bernoulli_data)
    
    # 3. Gamma-Poisson
    alpha_0, beta_0 = 2, 1  # Prior
    poisson_data = np.random.poisson(5, 15)  # Data
    alpha_post_pois = alpha_0 + sum(poisson_data)
    beta_post_pois = beta_0 + len(poisson_data)
    
    return {
        'normal': {'prior': (mu_0, sigma_0), 'posterior': (mu_post, sigma_post), 'data': data},
        'bernoulli': {'prior': (alpha_0, beta_0), 'posterior': (alpha_post, beta_post), 'data': bernoulli_data},
        'poisson': {'prior': (alpha_0, beta_0), 'posterior': (alpha_post_pois, beta_post_pois), 'data': poisson_data}
    }

conjugate_examples = conjugate_prior_examples()

# Visualize conjugate prior examples
plt.figure(figsize=(15, 10))

# Normal-Normal
plt.subplot(3, 3, 1)
x = np.linspace(-5, 5, 1000)
prior_norm = norm.pdf(x, conjugate_examples['normal']['prior'][0], conjugate_examples['normal']['prior'][1])
posterior_norm = norm.pdf(x, conjugate_examples['normal']['posterior'][0], conjugate_examples['normal']['posterior'][1])
plt.plot(x, prior_norm, 'b-', linewidth=2, label='Prior')
plt.plot(x, posterior_norm, 'r-', linewidth=2, label='Posterior')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Normal-Normal Conjugate')
plt.legend()

# Data histogram
plt.subplot(3, 3, 2)
plt.hist(conjugate_examples['normal']['data'], bins=10, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Data')
plt.ylabel('Frequency')
plt.title('Normal Data')

# Beta-Bernoulli
plt.subplot(3, 3, 3)
theta = np.linspace(0, 1, 1000)
prior_beta = beta.pdf(theta, conjugate_examples['bernoulli']['prior'][0], conjugate_examples['bernoulli']['prior'][1])
posterior_beta = beta.pdf(theta, conjugate_examples['bernoulli']['posterior'][0], conjugate_examples['bernoulli']['posterior'][1])
plt.plot(theta, prior_beta, 'b-', linewidth=2, label='Prior')
plt.plot(theta, posterior_beta, 'r-', linewidth=2, label='Posterior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Beta-Bernoulli Conjugate')
plt.legend()

# Bernoulli data
plt.subplot(3, 3, 4)
plt.hist(conjugate_examples['bernoulli']['data'], bins=[0, 1, 2], alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Data')
plt.ylabel('Frequency')
plt.title('Bernoulli Data')
plt.xticks([0.5], ['0/1'])

# Gamma-Poisson
plt.subplot(3, 3, 5)
x_gamma = np.linspace(0, 10, 1000)
prior_gamma = gamma.pdf(x_gamma, conjugate_examples['poisson']['prior'][0], scale=1/conjugate_examples['poisson']['prior'][1])
posterior_gamma = gamma.pdf(x_gamma, conjugate_examples['poisson']['posterior'][0], scale=1/conjugate_examples['poisson']['posterior'][1])
plt.plot(x_gamma, prior_gamma, 'b-', linewidth=2, label='Prior')
plt.plot(x_gamma, posterior_gamma, 'r-', linewidth=2, label='Posterior')
plt.xlabel('λ')
plt.ylabel('Density')
plt.title('Gamma-Poisson Conjugate')
plt.legend()

# Poisson data
plt.subplot(3, 3, 6)
plt.hist(conjugate_examples['poisson']['data'], bins=range(0, max(conjugate_examples['poisson']['data'])+2), 
         alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Data')
plt.ylabel('Frequency')
plt.title('Poisson Data')

plt.tight_layout()
plt.show()

print("Conjugate Prior Examples:")
print(f"Normal-Normal: Prior μ={conjugate_examples['normal']['prior'][0]:.2f}, Posterior μ={conjugate_examples['normal']['posterior'][0]:.2f}")
print(f"Beta-Bernoulli: Prior α={conjugate_examples['bernoulli']['prior'][0]}, Posterior α={conjugate_examples['bernoulli']['posterior'][0]}")
print(f"Gamma-Poisson: Prior α={conjugate_examples['poisson']['prior'][0]}, Posterior α={conjugate_examples['poisson']['posterior'][0]}")
```

## MCMC Methods

### Metropolis-Hastings Algorithm

```python
def metropolis_hastings(target_dist, proposal_dist, n_samples=10000, initial_state=0):
    """Metropolis-Hastings MCMC algorithm"""
    
    samples = [initial_state]
    accepted = 0
    
    for i in range(n_samples):
        current_state = samples[-1]
        
        # Propose new state
        proposed_state = proposal_dist(current_state)
        
        # Calculate acceptance probability
        acceptance_ratio = target_dist(proposed_state) / target_dist(current_state)
        acceptance_prob = min(1, acceptance_ratio)
        
        # Accept or reject
        if np.random.random() < acceptance_prob:
            samples.append(proposed_state)
            accepted += 1
        else:
            samples.append(current_state)
    
    acceptance_rate = accepted / n_samples
    return np.array(samples), acceptance_rate

# Example: Sample from a mixture of normals
def target_distribution(x):
    """Target distribution: mixture of two normals"""
    return 0.3 * norm.pdf(x, -2, 1) + 0.7 * norm.pdf(x, 3, 1.5)

def proposal_distribution(current_state):
    """Proposal distribution: normal centered at current state"""
    return np.random.normal(current_state, 1.0)

# Run MCMC
mcmc_samples, acceptance_rate = metropolis_hastings(target_distribution, proposal_distribution)

print("Metropolis-Hastings MCMC Results")
print(f"Number of samples: {len(mcmc_samples)}")
print(f"Acceptance rate: {acceptance_rate:.3f}")

# Visualize MCMC results
plt.figure(figsize=(15, 5))

# Target distribution
plt.subplot(1, 3, 1)
x = np.linspace(-6, 8, 1000)
target_pdf = target_distribution(x)
plt.plot(x, target_pdf, 'b-', linewidth=2, label='Target Distribution')
plt.hist(mcmc_samples, bins=50, density=True, alpha=0.7, color='red', edgecolor='black', label='MCMC Samples')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Target Distribution vs MCMC Samples')
plt.legend()

# Trace plot
plt.subplot(1, 3, 2)
plt.plot(mcmc_samples[:1000], 'g-', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Sample Value')
plt.title('Trace Plot (First 1000 iterations)')

# Autocorrelation
plt.subplot(1, 3, 3)
from statsmodels.tsa.stattools import acf
acf_values = acf(mcmc_samples, nlags=50)
plt.bar(range(len(acf_values)), acf_values, alpha=0.7, color='purple')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')

plt.tight_layout()
plt.show()

# Effective sample size
def effective_sample_size(samples):
    """Calculate effective sample size"""
    n = len(samples)
    acf_values = acf(samples, nlags=min(n//2, 1000))
    # Sum autocorrelations up to first negative value
    cutoff = np.where(acf_values < 0)[0]
    if len(cutoff) > 0:
        cutoff = cutoff[0]
    else:
        cutoff = len(acf_values)
    
    ess = n / (1 + 2 * np.sum(acf_values[1:cutoff]))
    return ess

ess = effective_sample_size(mcmc_samples)
print(f"Effective sample size: {ess:.0f}")
```

## Bayesian Regression

### Linear Regression with PyMC3

```python
def bayesian_linear_regression():
    """Perform Bayesian linear regression using PyMC3"""
    
    # Generate synthetic data
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 10, n)
    true_slope = 2.5
    true_intercept = 1.0
    true_sigma = 1.0
    y = true_intercept + true_slope * x + np.random.normal(0, true_sigma, n)
    
    # Bayesian model with PyMC3
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sd=10)
        slope = pm.Normal('slope', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        # Likelihood
        mu = intercept + slope * x
        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=y)
        
        # Sample from posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=False)
    
    return model, trace, x, y

model, trace, x_reg, y_reg = bayesian_linear_regression()

print("Bayesian Linear Regression Results")
print(f"True parameters: intercept={1.0}, slope={2.5}, sigma={1.0}")

# Visualize Bayesian regression results
plt.figure(figsize=(15, 10))

# Posterior distributions
plt.subplot(2, 3, 1)
plt.hist(trace['intercept'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='True value')
plt.xlabel('Intercept')
plt.ylabel('Frequency')
plt.title('Posterior Distribution - Intercept')
plt.legend()

plt.subplot(2, 3, 2)
plt.hist(trace['slope'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(2.5, color='red', linestyle='--', linewidth=2, label='True value')
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.title('Posterior Distribution - Slope')
plt.legend()

plt.subplot(2, 3, 3)
plt.hist(trace['sigma'], bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='True value')
plt.xlabel('Sigma')
plt.ylabel('Frequency')
plt.title('Posterior Distribution - Sigma')
plt.legend()

# Regression plot with uncertainty
plt.subplot(2, 3, 4)
plt.scatter(x_reg, y_reg, alpha=0.7, color='blue', label='Data')

# Plot regression lines from posterior samples
x_plot = np.linspace(0, 10, 100)
for i in range(0, len(trace['intercept']), 100):  # Plot every 100th sample
    y_plot = trace['intercept'][i] + trace['slope'][i] * x_plot
    plt.plot(x_plot, y_plot, 'r-', alpha=0.1)

# Plot mean regression line
mean_intercept = np.mean(trace['intercept'])
mean_slope = np.mean(trace['slope'])
y_mean = mean_intercept + mean_slope * x_plot
plt.plot(x_plot, y_mean, 'r-', linewidth=3, label='Mean prediction')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Regression with Uncertainty')
plt.legend()

# Credible intervals
plt.subplot(2, 3, 5)
# Calculate credible intervals for predictions
y_predictions = []
for x_val in x_plot:
    y_pred = trace['intercept'] + trace['slope'] * x_val
    y_predictions.append(y_pred)

y_predictions = np.array(y_predictions)
lower_ci = np.percentile(y_predictions, 2.5, axis=1)
upper_ci = np.percentile(y_predictions, 97.5, axis=1)

plt.scatter(x_reg, y_reg, alpha=0.7, color='blue', label='Data')
plt.plot(x_plot, y_mean, 'r-', linewidth=2, label='Mean prediction')
plt.fill_between(x_plot, lower_ci, upper_ci, alpha=0.3, color='red', label='95% Credible Interval')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression with Credible Intervals')
plt.legend()

# Trace plots
plt.subplot(2, 3, 6)
plt.plot(trace['slope'][:1000], 'g-', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Slope')
plt.title('Trace Plot - Slope')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\nPosterior Summary:")
print(f"Intercept: {np.mean(trace['intercept']):.3f} ± {np.std(trace['intercept']):.3f}")
print(f"Slope: {np.mean(trace['slope']):.3f} ± {np.std(trace['slope']):.3f}")
print(f"Sigma: {np.mean(trace['sigma']):.3f} ± {np.std(trace['sigma']):.3f}")
```

## Model Comparison

### Bayes Factors and Model Selection

```python
def bayesian_model_comparison():
    """Compare Bayesian models using different criteria"""
    
    # Generate data from a quadratic model
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-3, 3, n)
    y = 1 + 2*x + 0.5*x**2 + np.random.normal(0, 0.5, n)
    
    # Model 1: Linear
    with pm.Model() as linear_model:
        intercept = pm.Normal('intercept', mu=0, sd=10)
        slope = pm.Normal('slope', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        mu = intercept + slope * x
        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=y)
        
        linear_trace = pm.sample(1000, tune=500, return_inferencedata=False)
    
    # Model 2: Quadratic
    with pm.Model() as quadratic_model:
        intercept = pm.Normal('intercept', mu=0, sd=10)
        slope = pm.Normal('slope', mu=0, sd=10)
        quadratic = pm.Normal('quadratic', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        mu = intercept + slope * x + quadratic * x**2
        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=y)
        
        quadratic_trace = pm.sample(1000, tune=500, return_inferencedata=False)
    
    return linear_model, quadratic_model, linear_trace, quadratic_trace, x, y

linear_model, quadratic_model, linear_trace, quadratic_trace, x_comp, y_comp = bayesian_model_comparison()

# Calculate model comparison metrics
def calculate_model_metrics(model, trace, x, y):
    """Calculate various model comparison metrics"""
    
    # DIC (Deviance Information Criterion)
    dic = pm.dic(trace, model)
    
    # WAIC (Widely Applicable Information Criterion)
    waic = pm.waic(trace, model)
    
    # LOO-CV (Leave-One-Out Cross-Validation)
    loo = pm.loo(trace, model)
    
    return {'DIC': dic, 'WAIC': waic, 'LOO': loo}

linear_metrics = calculate_model_metrics(linear_model, linear_trace, x_comp, y_comp)
quadratic_metrics = calculate_model_metrics(quadratic_model, quadratic_trace, x_comp, y_comp)

print("Bayesian Model Comparison")
print(f"Linear Model:")
print(f"  DIC: {linear_metrics['DIC']:.2f}")
print(f"  WAIC: {linear_metrics['WAIC']['waic']:.2f}")
print(f"  LOO: {linear_metrics['LOO']['loo']:.2f}")

print(f"\nQuadratic Model:")
print(f"  DIC: {quadratic_metrics['DIC']:.2f}")
print(f"  WAIC: {quadratic_metrics['WAIC']['waic']:.2f}")
print(f"  LOO: {quadratic_metrics['LOO']['loo']:.2f}")

# Visualize model comparison
plt.figure(figsize=(15, 5))

# Data and model fits
plt.subplot(1, 3, 1)
plt.scatter(x_comp, y_comp, alpha=0.7, color='blue', label='Data')

# Linear model predictions
x_plot = np.linspace(-3, 3, 100)
linear_pred = np.mean(linear_trace['intercept']) + np.mean(linear_trace['slope']) * x_plot
plt.plot(x_plot, linear_pred, 'r-', linewidth=2, label='Linear Model')

# Quadratic model predictions
quad_pred = (np.mean(quadratic_trace['intercept']) + 
             np.mean(quadratic_trace['slope']) * x_plot + 
             np.mean(quadratic_trace['quadratic']) * x_plot**2)
plt.plot(x_plot, quad_pred, 'g-', linewidth=2, label='Quadratic Model')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Fits')
plt.legend()

# Model comparison metrics
plt.subplot(1, 3, 2)
metrics = ['DIC', 'WAIC', 'LOO']
linear_values = [linear_metrics['DIC'], linear_metrics['WAIC']['waic'], linear_metrics['LOO']['loo']]
quadratic_values = [quadratic_metrics['DIC'], quadratic_metrics['WAIC']['waic'], quadratic_metrics['LOO']['loo']]

x_pos = np.arange(len(metrics))
width = 0.35

plt.bar(x_pos - width/2, linear_values, width, label='Linear Model', alpha=0.7)
plt.bar(x_pos + width/2, quadratic_values, width, label='Quadratic Model', alpha=0.7)
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Model Comparison Metrics')
plt.xticks(x_pos, metrics)
plt.legend()

# Residuals comparison
plt.subplot(1, 3, 3)
linear_residuals = y_comp - (np.mean(linear_trace['intercept']) + np.mean(linear_trace['slope']) * x_comp)
quadratic_residuals = y_comp - (np.mean(quadratic_trace['intercept']) + 
                               np.mean(quadratic_trace['slope']) * x_comp + 
                               np.mean(quadratic_trace['quadratic']) * x_comp**2)

plt.scatter(x_comp, linear_residuals, alpha=0.7, color='red', label='Linear Residuals')
plt.scatter(x_comp, quadratic_residuals, alpha=0.7, color='green', label='Quadratic Residuals')
plt.axhline(0, color='black', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('Residuals')
plt.title('Residuals Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# Model selection conclusion
print(f"\nModel Selection:")
print(f"Lower values indicate better models.")
print(f"Linear model preferred by: {sum(1 for i in range(3) if linear_values[i] < quadratic_values[i])}/3 metrics")
```

## Practical Applications

### Bayesian A/B Testing

```python
def bayesian_ab_testing():
    """Perform Bayesian A/B testing"""
    
    # Simulate A/B test data
    np.random.seed(42)
    n_a, n_b = 1000, 1000
    true_rate_a = 0.10  # 10% conversion rate
    true_rate_b = 0.12  # 12% conversion rate
    
    conversions_a = np.random.binomial(n_a, true_rate_a)
    conversions_b = np.random.binomial(n_b, true_rate_b)
    
    # Bayesian model
    with pm.Model() as ab_model:
        # Priors for conversion rates
        rate_a = pm.Beta('rate_a', alpha=1, beta=1)  # Uniform prior
        rate_b = pm.Beta('rate_b', alpha=1, beta=1)  # Uniform prior
        
        # Likelihoods
        obs_a = pm.Binomial('obs_a', n=n_a, p=rate_a, observed=conversions_a)
        obs_b = pm.Binomial('obs_b', n=n_b, p=rate_b, observed=conversions_b)
        
        # Difference between rates
        diff = pm.Deterministic('diff', rate_b - rate_a)
        
        # Sample from posterior
        ab_trace = pm.sample(2000, tune=1000, return_inferencedata=False)
    
    return ab_trace, conversions_a, conversions_b, n_a, n_b

ab_trace, conv_a, conv_b, n_a, n_b = bayesian_ab_testing()

print("Bayesian A/B Testing Results")
print(f"Group A: {conv_a}/{n_a} conversions ({conv_a/n_a:.3f})")
print(f"Group B: {conv_b}/{n_b} conversions ({conv_b/n_b:.3f})")

# Visualize A/B testing results
plt.figure(figsize=(15, 5))

# Posterior distributions
plt.subplot(1, 3, 1)
plt.hist(ab_trace['rate_a'], bins=30, alpha=0.7, color='blue', label='Group A', density=True)
plt.hist(ab_trace['rate_b'], bins=30, alpha=0.7, color='red', label='Group B', density=True)
plt.axvline(conv_a/n_a, color='blue', linestyle='--', alpha=0.7)
plt.axvline(conv_b/n_b, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.title('Posterior Distributions')
plt.legend()

# Difference distribution
plt.subplot(1, 3, 2)
plt.hist(ab_trace['diff'], bins=30, alpha=0.7, color='green', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='No difference')
plt.xlabel('Difference (B - A)')
plt.ylabel('Frequency')
plt.title('Posterior Distribution of Difference')
plt.legend()

# Probability of B being better
prob_b_better = np.mean(ab_trace['diff'] > 0)
plt.subplot(1, 3, 3)
plt.bar(['A Better', 'B Better'], [1-prob_b_better, prob_b_better], 
        alpha=0.7, color=['blue', 'red'])
plt.ylabel('Probability')
plt.title('Probability of B being Better')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Credible intervals
rate_a_ci = np.percentile(ab_trace['rate_a'], [2.5, 97.5])
rate_b_ci = np.percentile(ab_trace['rate_b'], [2.5, 97.5])
diff_ci = np.percentile(ab_trace['diff'], [2.5, 97.5])

print(f"\nCredible Intervals (95%):")
print(f"Rate A: [{rate_a_ci[0]:.4f}, {rate_a_ci[1]:.4f}]")
print(f"Rate B: [{rate_b_ci[0]:.4f}, {rate_b_ci[1]:.4f}]")
print(f"Difference: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
print(f"Probability B > A: {prob_b_better:.3f}")
```

## Practice Problems

1. **Bayesian Inference**: Implement Bayesian updating for different likelihood-prior combinations.

2. **MCMC Diagnostics**: Create comprehensive MCMC diagnostic tools for convergence assessment.

3. **Model Comparison**: Build Bayesian model comparison frameworks with multiple criteria.

4. **Hierarchical Models**: Implement hierarchical Bayesian models for grouped data.

## Further Reading

- "Bayesian Data Analysis" by Andrew Gelman et al.
- "Doing Bayesian Data Analysis" by John K. Kruschke
- "Statistical Rethinking" by Richard McElreath
- "Bayesian Methods for Hackers" by Cameron Davidson-Pilon

## Key Takeaways

- **Bayesian inference** provides a coherent framework for updating beliefs with data
- **Conjugate priors** simplify posterior calculations and have analytical solutions
- **MCMC methods** enable sampling from complex posterior distributions
- **Bayesian regression** naturally incorporates uncertainty in predictions
- **Model comparison** uses information criteria like DIC, WAIC, and LOO-CV
- **Credible intervals** provide intuitive uncertainty quantification
- **Bayesian methods** are particularly valuable for small datasets and complex models
- **Real-world applications** include A/B testing, medical diagnosis, and recommendation systems

In the next chapter, we'll explore experimental design, including randomized controlled trials, factorial designs, and A/B testing methodologies. 