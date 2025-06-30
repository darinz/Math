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

## Bayesian Inference

Bayesian inference provides a coherent framework for updating beliefs about parameters based on observed data, combining prior knowledge with new evidence.

### Mathematical Foundation

**Bayes' Theorem:**
$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

Where:
- $P(\theta | D)$ = **Posterior distribution** (updated belief about θ given data)
- $P(D | \theta)$ = **Likelihood function** (probability of data given θ)
- $P(\theta)$ = **Prior distribution** (initial belief about θ)
- $P(D)$ = **Evidence/Marginal likelihood** (normalizing constant)

**Continuous Case:**
$$f(\theta | D) = \frac{f(D | \theta) f(\theta)}{\int f(D | \theta) f(\theta) d\theta}$$

**Log-Posterior:**
$$\log f(\theta | D) = \log f(D | \theta) + \log f(\theta) - \log \int f(D | \theta) f(\theta) d\theta$$

### Prior Distributions

**Conjugate Priors:**
A prior is **conjugate** to a likelihood if the posterior belongs to the same family as the prior.

**Common Conjugate Pairs:**

**1. Normal-Normal:**
- Likelihood: $X_i \sim N(\mu, \sigma^2)$ (σ² known)
- Prior: $\mu \sim N(\mu_0, \tau_0^2)$
- Posterior: $\mu | D \sim N(\mu_n, \tau_n^2)$

Where:
$$\mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}$$
$$\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}$$

**2. Beta-Binomial:**
- Likelihood: $X \sim \text{Binomial}(n, \theta)$
- Prior: $\theta \sim \text{Beta}(\alpha, \beta)$
- Posterior: $\theta | D \sim \text{Beta}(\alpha + x, \beta + n - x)$

**3. Gamma-Poisson:**
- Likelihood: $X_i \sim \text{Poisson}(\lambda)$
- Prior: $\lambda \sim \text{Gamma}(\alpha, \beta)$
- Posterior: $\lambda | D \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$

**4. Inverse Gamma-Normal:**
- Likelihood: $X_i \sim N(\mu, \sigma^2)$ (μ known)
- Prior: $\sigma^2 \sim \text{InvGamma}(\alpha, \beta)$
- Posterior: $\sigma^2 | D \sim \text{InvGamma}(\alpha + n/2, \beta + \frac{1}{2}\sum(x_i - \mu)^2)$

**Non-Informative Priors:**

**1. Jeffreys Prior:**
$$f(\theta) \propto \sqrt{I(\theta)}$$

Where $I(\theta)$ is the Fisher information:
$$I(\theta) = -E\left[\frac{\partial^2}{\partial \theta^2} \log f(X | \theta)\right]$$

**2. Uniform Prior:**
$$f(\theta) \propto 1$$

**3. Reference Prior:**
Maximizes the expected Kullback-Leibler divergence between prior and posterior.

### Posterior Analysis

**Posterior Mean (Bayes Estimator):**
$$\hat{\theta}_{Bayes} = E[\theta | D] = \int \theta f(\theta | D) d\theta$$

**Posterior Variance:**
$$\text{Var}(\theta | D) = E[(\theta - \hat{\theta}_{Bayes})^2 | D]$$

**Posterior Mode (Maximum A Posteriori):**
$$\hat{\theta}_{MAP} = \arg\max_{\theta} f(\theta | D)$$

**Credible Intervals:**
A $(1-\alpha)$ credible interval satisfies:
$$P(\theta_L \leq \theta \leq \theta_U | D) = 1 - \alpha$$

**Highest Posterior Density (HPD) Interval:**
The shortest interval containing $(1-\alpha)$ of the posterior probability.

### Predictive Distributions

**Posterior Predictive Distribution:**
$$f(x_{new} | D) = \int f(x_{new} | \theta) f(\theta | D) d\theta$$

**Prior Predictive Distribution:**
$$f(x) = \int f(x | \theta) f(\theta) d\theta$$

### Model Comparison

**Bayes Factor:**
$$BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int f(D | \theta_1, M_1) f(\theta_1 | M_1) d\theta_1}{\int f(D | \theta_2, M_2) f(\theta_2 | M_2) d\theta_2}$$

**Posterior Model Probabilities:**
$$P(M_i | D) = \frac{P(D | M_i) P(M_i)}{\sum_j P(D | M_j) P(M_j)}$$

### Computational Methods

**1. Markov Chain Monte Carlo (MCMC):**
- **Metropolis-Hastings Algorithm**
- **Gibbs Sampling**
- **Hamiltonian Monte Carlo**

**2. Variational Inference:**
Approximate the posterior with a simpler distribution.

**3. Laplace Approximation:**
Approximate the posterior as a normal distribution around the MAP.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns

def normal_normal_conjugate(x, mu0, tau0_sq, sigma_sq):
    """
    Normal-Normal conjugate pair
    
    Mathematical implementation:
    Prior: μ ~ N(μ₀, τ₀²)
    Likelihood: Xᵢ ~ N(μ, σ²)
    Posterior: μ|D ~ N(μₙ, τₙ²)
    
    Where:
    μₙ = (μ₀/τ₀² + nẍ/σ²) / (1/τ₀² + n/σ²)
    1/τₙ² = 1/τ₀² + n/σ²
    
    Parameters:
    x: array, observed data
    mu0: float, prior mean
    tau0_sq: float, prior variance
    sigma_sq: float, known data variance
    
    Returns:
    tuple: (posterior_mean, posterior_variance)
    """
    n = len(x)
    x_bar = np.mean(x)
    
    # Posterior parameters
    tau_n_sq_inv = 1/tau0_sq + n/sigma_sq
    tau_n_sq = 1/tau_n_sq_inv
    
    mu_n = (mu0/tau0_sq + n*x_bar/sigma_sq) / tau_n_sq_inv
    
    return mu_n, tau_n_sq

def beta_binomial_conjugate(x, n, alpha, beta):
    """
    Beta-Binomial conjugate pair
    
    Mathematical implementation:
    Prior: θ ~ Beta(α, β)
    Likelihood: X ~ Binomial(n, θ)
    Posterior: θ|D ~ Beta(α + x, β + n - x)
    
    Parameters:
    x: int, number of successes
    n: int, number of trials
    alpha, beta: float, prior parameters
    
    Returns:
    tuple: (posterior_alpha, posterior_beta)
    """
    alpha_post = alpha + x
    beta_post = beta + n - x
    
    return alpha_post, beta_post

def gamma_poisson_conjugate(x, alpha, beta):
    """
    Gamma-Poisson conjugate pair
    
    Mathematical implementation:
    Prior: λ ~ Gamma(α, β)
    Likelihood: Xᵢ ~ Poisson(λ)
    Posterior: λ|D ~ Gamma(α + Σxᵢ, β + n)
    
    Parameters:
    x: array, observed data
    alpha, beta: float, prior parameters
    
    Returns:
    tuple: (posterior_alpha, posterior_beta)
    """
    n = len(x)
    sum_x = np.sum(x)
    
    alpha_post = alpha + sum_x
    beta_post = beta + n
    
    return alpha_post, beta_post

def jeffreys_prior_normal():
    """
    Jeffreys prior for normal distribution with unknown mean
    
    Mathematical implementation:
    f(μ) ∝ 1 (uniform prior)
    f(σ²) ∝ 1/σ² (scale-invariant prior)
    """
    return "f(μ) ∝ 1, f(σ²) ∝ 1/σ²"

def posterior_predictive_normal(mu_post, sigma_post_sq, sigma_data_sq):
    """
    Posterior predictive distribution for normal model
    
    Mathematical implementation:
    X_new|D ~ N(μ_post, σ_post² + σ_data²)
    
    Parameters:
    mu_post: float, posterior mean
    sigma_post_sq: float, posterior variance
    sigma_data_sq: float, data variance
    
    Returns:
    tuple: (predictive_mean, predictive_variance)
    """
    pred_mean = mu_post
    pred_var = sigma_post_sq + sigma_data_sq
    
    return pred_mean, pred_var

def bayes_factor_normal(x, mu1, sigma1_sq, mu2, sigma2_sq, prior1=0.5, prior2=0.5):
    """
    Calculate Bayes factor for two normal models
    
    Mathematical implementation:
    BF₁₂ = P(D|M₁) / P(D|M₂)
    
    Parameters:
    x: array, observed data
    mu1, sigma1_sq: parameters of model 1
    mu2, sigma2_sq: parameters of model 2
    prior1, prior2: prior model probabilities
    
    Returns:
    float: Bayes factor
    """
    n = len(x)
    
    # Calculate marginal likelihoods
    def marginal_likelihood_normal(x, mu, sigma_sq):
        # Assuming conjugate normal-normal with non-informative prior
        x_bar = np.mean(x)
        s_sq = np.var(x, ddof=1)
        
        # Marginal likelihood for normal with unknown mean
        log_ml = -(n/2) * np.log(2*np.pi) - (n/2) * np.log(sigma_sq) - \
                 (1/(2*sigma_sq)) * (np.sum((x - x_bar)**2) + n*(x_bar - mu)**2)
        return np.exp(log_ml)
    
    ml1 = marginal_likelihood_normal(x, mu1, sigma1_sq)
    ml2 = marginal_likelihood_normal(x, mu2, sigma2_sq)
    
    bayes_factor = ml1 / ml2
    return bayes_factor

def credible_interval_normal(mu_post, sigma_post_sq, alpha=0.05):
    """
    Calculate credible interval for normal posterior
    
    Mathematical implementation:
    P(μ_L ≤ μ ≤ μ_U | D) = 1 - α
    
    Parameters:
    mu_post: float, posterior mean
    sigma_post_sq: float, posterior variance
    alpha: float, significance level
    
    Returns:
    tuple: (lower_bound, upper_bound)
    """
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)
    margin = z_alpha_2 * np.sqrt(sigma_post_sq)
    
    lower = mu_post - margin
    upper = mu_post + margin
    
    return lower, upper

def metropolis_hastings_normal(log_posterior, x0, n_samples=10000, proposal_std=0.1):
    """
    Metropolis-Hastings algorithm for normal posterior
    
    Mathematical implementation:
    1. Propose θ* ~ q(θ*|θₜ)
    2. Accept with probability min(1, f(θ*|D)/f(θₜ|D) × q(θₜ|θ*)/q(θ*|θₜ))
    
    Parameters:
    log_posterior: function, log posterior density
    x0: float, initial value
    n_samples: int, number of samples
    proposal_std: float, proposal standard deviation
    
    Returns:
    array: MCMC samples
    """
    samples = np.zeros(n_samples)
    samples[0] = x0
    
    accepted = 0
    
    for i in range(1, n_samples):
        # Propose new value
        proposal = np.random.normal(samples[i-1], proposal_std)
        
        # Calculate acceptance probability
        log_alpha = log_posterior(proposal) - log_posterior(samples[i-1])
        alpha = min(1, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.random() < alpha:
            samples[i] = proposal
            accepted += 1
        else:
            samples[i] = samples[i-1]
    
    acceptance_rate = accepted / (n_samples - 1)
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    return samples

def laplace_approximation(log_posterior, x0):
    """
    Laplace approximation to posterior
    
    Mathematical implementation:
    f(θ|D) ≈ N(θ̂, -1/H(θ̂))
    where θ̂ is the MAP and H(θ̂) is the Hessian at θ̂
    
    Parameters:
    log_posterior: function, log posterior density
    x0: float, initial guess
    
    Returns:
    tuple: (map_estimate, laplace_variance)
    """
    # Find MAP
    result = minimize(lambda x: -log_posterior(x), x0, method='BFGS')
    map_estimate = result.x[0]
    
    # Calculate Hessian (second derivative)
    h = 1e-6
    hessian = -(log_posterior(map_estimate + h) - 2*log_posterior(map_estimate) + 
                log_posterior(map_estimate - h)) / (h**2)
    
    laplace_variance = 1 / hessian
    
    return map_estimate, laplace_variance

# Example: Normal-Normal conjugate analysis
np.random.seed(42)

# True parameters
true_mu = 5.0
true_sigma = 2.0
n_data = 20

# Generate data
data = np.random.normal(true_mu, true_sigma, n_data)

# Prior parameters
mu0 = 0.0
tau0_sq = 10.0
sigma_sq = true_sigma**2  # Assume known

print("Bayesian Inference: Normal-Normal Conjugate Analysis")
print("=" * 60)

# Calculate posterior
mu_post, tau_post_sq = normal_normal_conjugate(data, mu0, tau0_sq, sigma_sq)

print(f"Data: n = {n_data}, x̄ = {np.mean(data):.3f}, s² = {np.var(data, ddof=1):.3f}")
print(f"Prior: μ ~ N({mu0}, {tau0_sq})")
print(f"Posterior: μ|D ~ N({mu_post:.3f}, {tau_post_sq:.3f})")

# Compare with frequentist estimate
freq_estimate = np.mean(data)
freq_var = sigma_sq / n_data

print(f"\nComparison:")
print(f"Frequentist: μ̂ = {freq_estimate:.3f}, Var(μ̂) = {freq_var:.3f}")
print(f"Bayesian: μ̂ = {mu_post:.3f}, Var(μ|D) = {tau_post_sq:.3f}")

# Credible interval
ci_lower, ci_upper = credible_interval_normal(mu_post, tau_post_sq, alpha=0.05)
print(f"95% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Posterior predictive distribution
pred_mean, pred_var = posterior_predictive_normal(mu_post, tau_post_sq, sigma_sq)
print(f"Posterior Predictive: X_new|D ~ N({pred_mean:.3f}, {pred_var:.3f})")

# MCMC sampling
def log_posterior_normal(mu):
    """Log posterior for normal model with normal prior"""
    # Log likelihood
    log_likelihood = -0.5 * np.sum((data - mu)**2) / sigma_sq
    
    # Log prior
    log_prior = -0.5 * (mu - mu0)**2 / tau0_sq
    
    return log_likelihood + log_prior

mcmc_samples = metropolis_hastings_normal(log_posterior_normal, x0=0.0, n_samples=5000)
print(f"MCMC mean: {np.mean(mcmc_samples):.3f}")
print(f"MCMC variance: {np.var(mcmc_samples):.3f}")

# Laplace approximation
map_est, laplace_var = laplace_approximation(log_posterior_normal, x0=0.0)
print(f"Laplace approximation: μ̂ = {map_est:.3f}, Var(μ|D) = {laplace_var:.3f}")

# Visualize Bayesian analysis
plt.figure(figsize=(15, 10))

# 1. Prior, likelihood, and posterior
plt.subplot(2, 3, 1)
mu_range = np.linspace(-2, 12, 1000)

# Prior
prior = stats.norm.pdf(mu_range, mu0, np.sqrt(tau0_sq))
plt.plot(mu_range, prior, 'b-', linewidth=2, label='Prior')

# Likelihood (scaled)
likelihood = stats.norm.pdf(mu_range, np.mean(data), np.sqrt(sigma_sq/n_data))
likelihood = likelihood / np.max(likelihood) * np.max(prior)  # Scale for visualization
plt.plot(mu_range, likelihood, 'g-', linewidth=2, label='Likelihood (scaled)')

# Posterior
posterior = stats.norm.pdf(mu_range, mu_post, np.sqrt(tau_post_sq))
plt.plot(mu_range, posterior, 'r-', linewidth=2, label='Posterior')

plt.axvline(true_mu, color='k', linestyle='--', alpha=0.7, label='True μ')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Prior, Likelihood, and Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Data histogram with predictive distribution
plt.subplot(2, 3, 2)
plt.hist(data, bins=10, alpha=0.7, density=True, label='Data')

# Predictive distribution
x_pred = np.linspace(min(data) - 2, max(data) + 2, 1000)
pred_pdf = stats.norm.pdf(x_pred, pred_mean, np.sqrt(pred_var))
plt.plot(x_pred, pred_pdf, 'r-', linewidth=2, label='Predictive')

plt.xlabel('X')
plt.ylabel('Density')
plt.title('Data and Predictive Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. MCMC trace plot
plt.subplot(2, 3, 3)
plt.plot(mcmc_samples[:1000], alpha=0.7)
plt.axhline(mu_post, color='r', linestyle='--', label='Analytical Posterior Mean')
plt.xlabel('Iteration')
plt.ylabel('μ')
plt.title('MCMC Trace Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. MCMC histogram
plt.subplot(2, 3, 4)
plt.hist(mcmc_samples, bins=50, alpha=0.7, density=True, label='MCMC Samples')

# Analytical posterior
posterior_mcmc = stats.norm.pdf(mu_range, mu_post, np.sqrt(tau_post_sq))
plt.plot(mu_range, posterior_mcmc, 'r-', linewidth=2, label='Analytical Posterior')

plt.xlabel('μ')
plt.ylabel('Density')
plt.title('MCMC vs Analytical Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Credible intervals comparison
plt.subplot(2, 3, 5)
intervals = []
labels = []

# Frequentist confidence interval
freq_ci = stats.norm.interval(0.95, loc=freq_estimate, scale=np.sqrt(freq_var))
intervals.append(freq_ci)
labels.append('Frequentist 95% CI')

# Bayesian credible interval
intervals.append((ci_lower, ci_upper))
labels.append('Bayesian 95% CI')

# Plot intervals
y_positions = np.arange(len(intervals))
for i, (lower, upper) in enumerate(intervals):
    plt.hlines(y_positions[i], lower, upper, linewidth=3, alpha=0.7)
    plt.plot([lower, upper], [y_positions[i], y_positions[i]], 'o', markersize=8)

plt.axvline(true_mu, color='k', linestyle='--', alpha=0.7, label='True μ')
plt.yticks(y_positions, labels)
plt.xlabel('μ')
plt.title('Confidence vs Credible Intervals')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Bayes factor analysis
plt.subplot(2, 3, 6)
# Compare two models
mu1, sigma1_sq = 3.0, 1.0
mu2, sigma2_sq = 7.0, 1.0

bf = bayes_factor_normal(data, mu1, sigma1_sq, mu2, sigma2_sq)
log_bf = np.log(bf)

plt.bar(['Model 1 vs Model 2'], [log_bf], alpha=0.7, color='skyblue')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.ylabel('log(Bayes Factor)')
plt.title('Model Comparison')
plt.grid(True, alpha=0.3)

# Add interpretation
if log_bf > 2:
    interpretation = "Strong evidence for Model 1"
elif log_bf > 1:
    interpretation = "Moderate evidence for Model 1"
elif log_bf > 0:
    interpretation = "Weak evidence for Model 1"
elif log_bf > -1:
    interpretation = "Weak evidence for Model 2"
elif log_bf > -2:
    interpretation = "Moderate evidence for Model 2"
else:
    interpretation = "Strong evidence for Model 2"

plt.text(0, log_bf + 0.1, interpretation, ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# Example: Beta-Binomial conjugate analysis
print(f"\nBeta-Binomial Conjugate Analysis")
print("=" * 40)

# Generate binomial data
n_trials = 50
true_theta = 0.3
x_successes = np.random.binomial(n_trials, true_theta)

# Prior parameters
alpha_prior = 2.0
beta_prior = 5.0

# Calculate posterior
alpha_post, beta_post = beta_binomial_conjugate(x_successes, n_trials, alpha_prior, beta_prior)

print(f"Data: {x_successes} successes out of {n_trials} trials")
print(f"Prior: θ ~ Beta({alpha_prior}, {beta_prior})")
print(f"Posterior: θ|D ~ Beta({alpha_post}, {beta_post})")

# Posterior statistics
posterior_mean = alpha_post / (alpha_post + beta_post)
posterior_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))

print(f"Posterior mean: {posterior_mean:.3f}")
print(f"Posterior variance: {posterior_var:.6f}")

# Frequentist comparison
freq_estimate = x_successes / n_trials
freq_var = freq_estimate * (1 - freq_estimate) / n_trials

print(f"Frequentist: θ̂ = {freq_estimate:.3f}, Var(θ̂) = {freq_var:.6f}")

# Credible interval for beta distribution
ci_lower_beta = stats.beta.ppf(0.025, alpha_post, beta_post)
ci_upper_beta = stats.beta.ppf(0.975, alpha_post, beta_post)
print(f"95% Credible Interval: [{ci_lower_beta:.3f}, {ci_upper_beta:.3f}]")

# Visualize beta-binomial analysis
plt.figure(figsize=(12, 8))

# Prior, likelihood, and posterior
theta_range = np.linspace(0, 1, 1000)

# Prior
prior_beta = stats.beta.pdf(theta_range, alpha_prior, beta_prior)
plt.subplot(2, 2, 1)
plt.plot(theta_range, prior_beta, 'b-', linewidth=2, label='Prior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Beta Prior')
plt.grid(True, alpha=0.3)

# Likelihood (scaled)
likelihood_binom = stats.binom.pmf(x_successes, n_trials, theta_range)
likelihood_binom = likelihood_binom / np.max(likelihood_binom) * np.max(prior_beta)
plt.subplot(2, 2, 2)
plt.plot(theta_range, likelihood_binom, 'g-', linewidth=2, label='Likelihood (scaled)')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Binomial Likelihood')
plt.grid(True, alpha=0.3)

# Posterior
posterior_beta = stats.beta.pdf(theta_range, alpha_post, beta_post)
plt.subplot(2, 2, 3)
plt.plot(theta_range, posterior_beta, 'r-', linewidth=2, label='Posterior')
plt.axvline(true_theta, color='k', linestyle='--', alpha=0.7, label='True θ')
plt.axvline(posterior_mean, color='orange', linestyle='--', alpha=0.7, label='Posterior Mean')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Beta Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

# Predictive distribution
plt.subplot(2, 2, 4)
n_new = 20
pred_successes = np.arange(0, n_new + 1)
pred_probs = stats.betabinom.pmf(pred_successes, n_new, alpha_post, beta_post)

plt.bar(pred_successes, pred_probs, alpha=0.7, color='purple')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Predictive Distribution (n={n_new})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate mathematical properties
print(f"\nMathematical Properties Verification:")

# 1. Conjugate property
print(f"1. Conjugate Property:")
print(f"   Prior: Beta({alpha_prior}, {beta_prior})")
print(f"   Likelihood: Binomial({n_trials}, θ)")
print(f"   Posterior: Beta({alpha_post}, {beta_post})")
print(f"   Conjugate property holds: {alpha_post == alpha_prior + x_successes and beta_post == beta_prior + n_trials - x_successes}")

# 2. Posterior mean as weighted average
print(f"\n2. Posterior Mean as Weighted Average:")
prior_mean = alpha_prior / (alpha_prior + beta_prior)
likelihood_mean = x_successes / n_trials
weight_prior = (alpha_prior + beta_prior) / (alpha_prior + beta_prior + n_trials)
weight_likelihood = n_trials / (alpha_prior + beta_prior + n_trials)

weighted_avg = weight_prior * prior_mean + weight_likelihood * likelihood_mean
print(f"   Prior mean: {prior_mean:.3f}")
print(f"   Likelihood mean: {likelihood_mean:.3f}")
print(f"   Weighted average: {weighted_avg:.3f}")
print(f"   Posterior mean: {posterior_mean:.3f}")
print(f"   Agreement: {abs(weighted_avg - posterior_mean) < 1e-10}")

# 3. Effect of sample size
print(f"\n3. Effect of Sample Size:")
# Compare with different sample sizes
sample_sizes = [10, 50, 100, 500]
for n in sample_sizes:
    x_n = np.random.binomial(n, true_theta)
    alpha_n, beta_n = beta_binomial_conjugate(x_n, n, alpha_prior, beta_prior)
    mean_n = alpha_n / (alpha_n + beta_n)
    var_n = (alpha_n * beta_n) / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))
    print(f"   n={n}: θ̂={mean_n:.3f}, Var(θ|D)={var_n:.6f}")

# 4. Bayes factor interpretation
print(f"\n4. Bayes Factor Interpretation:")
bf_values = [0.01, 0.1, 0.3, 1, 3, 10, 100]
interpretations = ["Very strong evidence for M2", "Strong evidence for M2", 
                   "Moderate evidence for M2", "No preference", 
                   "Moderate evidence for M1", "Strong evidence for M1", 
                   "Very strong evidence for M1"]

for bf, interpretation in zip(bf_values, interpretations):
    print(f"   BF = {bf}: {interpretation}")
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