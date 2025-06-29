# Probability Fundamentals

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)

Probability theory is the mathematical foundation for statistics and machine learning. Understanding probability concepts is essential for building probabilistic models, interpreting statistical results, and making data-driven decisions.

## Table of Contents
- [Basic Probability Concepts](#basic-probability-concepts)
- [Random Variables](#random-variables)
- [Probability Distributions](#probability-distributions)
- [Joint and Conditional Probability](#joint-and-conditional-probability)
- [Bayes' Theorem](#bayes-theorem)
- [Central Limit Theorem](#central-limit-theorem)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, expon, uniform, beta, gamma
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

np.random.seed(42)
```

## Basic Probability Concepts

### Sample Space and Events

```python
def calculate_probability(favorable_outcomes, total_outcomes):
    """Calculate probability of an event"""
    return favorable_outcomes / total_outcomes

# Example: Rolling a fair die
def roll_die_probability():
    """Calculate probabilities for rolling a fair die"""
    total_outcomes = 6
    
    # Probability of rolling a 6
    prob_six = calculate_probability(1, total_outcomes)
    
    # Probability of rolling an even number
    prob_even = calculate_probability(3, total_outcomes)  # 2, 4, 6
    
    # Probability of rolling a number greater than 4
    prob_greater_than_4 = calculate_probability(2, total_outcomes)  # 5, 6
    
    return {
        'P(6)': prob_six,
        'P(even)': prob_even,
        'P(>4)': prob_greater_than_4
    }

probabilities = roll_die_probability()
for event, prob in probabilities.items():
    print(f"{event}: {prob:.3f}")

# Simulate die rolls
def simulate_die_rolls(n_rolls=1000):
    """Simulate rolling a die multiple times"""
    rolls = np.random.randint(1, 7, n_rolls)
    
    # Calculate empirical probabilities
    empirical_probs = {}
    for i in range(1, 7):
        empirical_probs[f'P({i})'] = np.sum(rolls == i) / n_rolls
    
    return rolls, empirical_probs

rolls, emp_probs = simulate_die_rolls(10000)
print("\nEmpirical probabilities (10,000 rolls):")
for event, prob in emp_probs.items():
    print(f"{event}: {prob:.3f}")
```

### Probability Rules

```python
def probability_rules_example():
    """Demonstrate probability rules with coin flips"""
    # Sample space for 2 coin flips: HH, HT, TH, TT
    sample_space = ['HH', 'HT', 'TH', 'TT']
    
    # Event A: First flip is heads
    event_a = ['HH', 'HT']
    
    # Event B: Second flip is heads
    event_b = ['HH', 'TH']
    
    # Event C: Both flips are heads
    event_c = ['HH']
    
    # Calculate probabilities
    p_a = len(event_a) / len(sample_space)
    p_b = len(event_b) / len(sample_space)
    p_c = len(event_c) / len(sample_space)
    
    # Union: A or B (A ∪ B)
    union_ab = list(set(event_a + event_b))
    p_union_ab = len(union_ab) / len(sample_space)
    
    # Intersection: A and B (A ∩ B)
    intersection_ab = list(set(event_a) & set(event_b))
    p_intersection_ab = len(intersection_ab) / len(sample_space)
    
    # Complement: Not A (A')
    complement_a = [x for x in sample_space if x not in event_a]
    p_complement_a = len(complement_a) / len(sample_space)
    
    return {
        'P(A)': p_a,
        'P(B)': p_b,
        'P(C)': p_c,
        'P(A ∪ B)': p_union_ab,
        'P(A ∩ B)': p_intersection_ab,
        'P(A\')': p_complement_a
    }

prob_rules = probability_rules_example()
print("Probability Rules Example (2 coin flips):")
for event, prob in prob_rules.items():
    print(f"{event}: {prob:.3f}")

# Verify probability rules
print(f"\nVerification:")
print(f"P(A) + P(A') = {prob_rules['P(A)'] + prob_rules['P(A\')']:.3f} (should be 1)")
print(f"P(A ∪ B) = P(A) + P(B) - P(A ∩ B): {prob_rules['P(A)'] + prob_rules['P(B)'] - prob_rules['P(A ∩ B)']:.3f}")
```

## Random Variables

### Discrete Random Variables

```python
def discrete_random_variable_example():
    """Example with discrete random variable: number of heads in 3 coin flips"""
    # Sample space: 000, 001, 010, 011, 100, 101, 110, 111
    # X = number of heads
    
    # Probability mass function
    pmf = {
        0: 1/8,  # 000
        1: 3/8,  # 001, 010, 100
        2: 3/8,  # 011, 101, 110
        3: 1/8   # 111
    }
    
    # Expected value
    expected_value = sum(x * p for x, p in pmf.items())
    
    # Variance
    variance = sum((x - expected_value)**2 * p for x, p in pmf.items())
    
    return pmf, expected_value, variance

pmf, ev, var = discrete_random_variable_example()
print("Discrete Random Variable: Number of heads in 3 coin flips")
print("Probability Mass Function:")
for x, p in pmf.items():
    print(f"P(X = {x}) = {p:.3f}")

print(f"\nExpected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")
print(f"Standard Deviation: σ = {np.sqrt(var):.3f}")

# Simulate the random variable
def simulate_coin_flips(n_experiments=10000):
    """Simulate 3 coin flips multiple times"""
    results = []
    for _ in range(n_experiments):
        flips = np.random.choice([0, 1], size=3)  # 0 = tails, 1 = heads
        num_heads = np.sum(flips)
        results.append(num_heads)
    
    return np.array(results)

simulated_results = simulate_coin_flips()
print(f"\nSimulation Results ({len(simulated_results)} experiments):")
print(f"Empirical mean: {np.mean(simulated_results):.3f}")
print(f"Empirical variance: {np.var(simulated_results):.3f}")

# Plot PMF
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
x_values = list(pmf.keys())
y_values = list(pmf.values())
plt.bar(x_values, y_values, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Theoretical PMF')
plt.xlabel('Number of Heads')
plt.ylabel('Probability')

plt.subplot(1, 2, 2)
unique, counts = np.unique(simulated_results, return_counts=True)
empirical_pmf = counts / len(simulated_results)
plt.bar(unique, empirical_pmf, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Empirical PMF')
plt.xlabel('Number of Heads')
plt.ylabel('Probability')

plt.tight_layout()
plt.show()
```

### Continuous Random Variables

```python
def continuous_random_variable_example():
    """Example with continuous random variable: uniform distribution"""
    # Uniform distribution on [0, 1]
    a, b = 0, 1
    
    # Probability density function
    def pdf(x):
        if a <= x <= b:
            return 1 / (b - a)
        else:
            return 0
    
    # Cumulative distribution function
    def cdf(x):
        if x < a:
            return 0
        elif x <= b:
            return (x - a) / (b - a)
        else:
            return 1
    
    # Expected value and variance
    expected_value = (a + b) / 2
    variance = (b - a)**2 / 12
    
    return pdf, cdf, expected_value, variance

pdf, cdf, ev, var = continuous_random_variable_example()
print("Continuous Random Variable: Uniform Distribution U(0,1)")
print(f"Expected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")

# Generate samples and plot
samples = np.random.uniform(0, 1, 10000)

plt.figure(figsize=(15, 5))

# PDF
plt.subplot(1, 3, 1)
x = np.linspace(-0.5, 1.5, 1000)
y_pdf = [pdf(xi) for xi in x]
plt.plot(x, y_pdf, 'b-', linewidth=2, label='Theoretical PDF')
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Empirical')
plt.title('Probability Density Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# CDF
plt.subplot(1, 3, 2)
y_cdf = [cdf(xi) for xi in x]
plt.plot(x, y_cdf, 'r-', linewidth=2, label='Theoretical CDF')
plt.hist(samples, bins=50, density=True, cumulative=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical')
plt.title('Cumulative Distribution Function')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()

# Q-Q plot
plt.subplot(1, 3, 3)
stats.probplot(samples, dist="uniform", plot=plt)
plt.title('Q-Q Plot vs Uniform Distribution')

plt.tight_layout()
plt.show()
```

## Probability Distributions

### Discrete Distributions

#### Binomial Distribution

```python
def binomial_distribution_example():
    """Binomial distribution: number of successes in n trials"""
    n, p = 10, 0.3  # 10 trials, 30% success probability
    
    # Theoretical PMF
    x_values = np.arange(0, n + 1)
    pmf_theoretical = binom.pmf(x_values, n, p)
    
    # Expected value and variance
    expected_value = n * p
    variance = n * p * (1 - p)
    
    # Simulate
    samples = np.random.binomial(n, p, 10000)
    
    return x_values, pmf_theoretical, samples, expected_value, variance

x_vals, pmf_theo, samples, ev, var = binomial_distribution_example()

print("Binomial Distribution B(10, 0.3)")
print(f"Expected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")

plt.figure(figsize=(12, 4))

# PMF comparison
plt.subplot(1, 2, 1)
plt.bar(x_vals, pmf_theo, alpha=0.7, color='skyblue', edgecolor='black', label='Theoretical')
unique, counts = np.unique(samples, return_counts=True)
empirical_pmf = counts / len(samples)
plt.bar(unique, empirical_pmf, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical')
plt.title('Binomial PMF')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.legend()

# CDF
plt.subplot(1, 2, 2)
cdf_theoretical = binom.cdf(x_vals, 10, 0.3)
plt.step(x_vals, cdf_theoretical, where='post', label='Theoretical CDF', linewidth=2)
plt.hist(samples, bins=11, density=True, cumulative=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical CDF')
plt.title('Binomial CDF')
plt.xlabel('Number of Successes')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Poisson Distribution

```python
def poisson_distribution_example():
    """Poisson distribution: number of events in fixed interval"""
    lambda_param = 3  # average rate of events
    
    # Theoretical PMF
    x_values = np.arange(0, 15)
    pmf_theoretical = poisson.pmf(x_values, lambda_param)
    
    # Expected value and variance (both equal to λ for Poisson)
    expected_value = lambda_param
    variance = lambda_param
    
    # Simulate
    samples = np.random.poisson(lambda_param, 10000)
    
    return x_values, pmf_theoretical, samples, expected_value, variance

x_vals, pmf_theo, samples, ev, var = poisson_distribution_example()

print("Poisson Distribution Poi(3)")
print(f"Expected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")

plt.figure(figsize=(12, 4))

# PMF comparison
plt.subplot(1, 2, 1)
plt.bar(x_vals, pmf_theo, alpha=0.7, color='skyblue', edgecolor='black', label='Theoretical')
unique, counts = np.unique(samples, return_counts=True)
empirical_pmf = counts / len(samples)
plt.bar(unique, empirical_pmf, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical')
plt.title('Poisson PMF')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()

# CDF
plt.subplot(1, 2, 2)
cdf_theoretical = poisson.cdf(x_vals, 3)
plt.step(x_vals, cdf_theoretical, where='post', label='Theoretical CDF', linewidth=2)
plt.hist(samples, bins=15, density=True, cumulative=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical CDF')
plt.title('Poisson CDF')
plt.xlabel('Number of Events')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

### Continuous Distributions

#### Normal Distribution

```python
def normal_distribution_example():
    """Normal distribution: the most important distribution in statistics"""
    mu, sigma = 0, 1  # standard normal distribution
    
    # Generate theoretical values
    x = np.linspace(-4, 4, 1000)
    pdf_theoretical = norm.pdf(x, mu, sigma)
    cdf_theoretical = norm.cdf(x, mu, sigma)
    
    # Simulate
    samples = np.random.normal(mu, sigma, 10000)
    
    # Expected value and variance
    expected_value = mu
    variance = sigma**2
    
    return x, pdf_theoretical, cdf_theoretical, samples, expected_value, variance

x, pdf_theo, cdf_theo, samples, ev, var = normal_distribution_example()

print("Normal Distribution N(0,1)")
print(f"Expected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")

plt.figure(figsize=(15, 5))

# PDF
plt.subplot(1, 3, 1)
plt.plot(x, pdf_theo, 'b-', linewidth=2, label='Theoretical PDF')
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Empirical')
plt.title('Normal PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# CDF
plt.subplot(1, 3, 2)
plt.plot(x, cdf_theo, 'r-', linewidth=2, label='Theoretical CDF')
plt.hist(samples, bins=50, density=True, cumulative=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical')
plt.title('Normal CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()

# Q-Q plot
plt.subplot(1, 3, 3)
stats.probplot(samples, dist="norm", plot=plt)
plt.title('Q-Q Plot vs Normal Distribution')

plt.tight_layout()
plt.show()

# Demonstrate 68-95-99.7 rule
print(f"\n68-95-99.7 Rule Verification:")
print(f"P(-1 < X < 1): {norm.cdf(1) - norm.cdf(-1):.3f} (should be ~0.68)")
print(f"P(-2 < X < 2): {norm.cdf(2) - norm.cdf(-2):.3f} (should be ~0.95)")
print(f"P(-3 < X < 3): {norm.cdf(3) - norm.cdf(-3):.3f} (should be ~0.997)")
```

#### Exponential Distribution

```python
def exponential_distribution_example():
    """Exponential distribution: time between events"""
    lambda_param = 2  # rate parameter
    
    # Generate theoretical values
    x = np.linspace(0, 5, 1000)
    pdf_theoretical = expon.pdf(x, scale=1/lambda_param)
    cdf_theoretical = expon.cdf(x, scale=1/lambda_param)
    
    # Simulate
    samples = np.random.exponential(1/lambda_param, 10000)
    
    # Expected value and variance
    expected_value = 1 / lambda_param
    variance = 1 / lambda_param**2
    
    return x, pdf_theoretical, cdf_theoretical, samples, expected_value, variance

x, pdf_theo, cdf_theo, samples, ev, var = exponential_distribution_example()

print("Exponential Distribution Exp(2)")
print(f"Expected Value: E[X] = {ev:.3f}")
print(f"Variance: Var(X) = {var:.3f}")

plt.figure(figsize=(15, 5))

# PDF
plt.subplot(1, 3, 1)
plt.plot(x, pdf_theo, 'b-', linewidth=2, label='Theoretical PDF')
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Empirical')
plt.title('Exponential PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# CDF
plt.subplot(1, 3, 2)
plt.plot(x, cdf_theo, 'r-', linewidth=2, label='Theoretical CDF')
plt.hist(samples, bins=50, density=True, cumulative=True, alpha=0.7, color='lightgreen', edgecolor='black', label='Empirical')
plt.title('Exponential CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()

# Q-Q plot
plt.subplot(1, 3, 3)
stats.probplot(samples, dist="expon", plot=plt)
plt.title('Q-Q Plot vs Exponential Distribution')

plt.tight_layout()
plt.show()
```

## Joint and Conditional Probability

### Joint Probability

```python
def joint_probability_example():
    """Example: Joint probability of two dice"""
    # Create joint probability table for two dice
    dice1 = np.arange(1, 7)
    dice2 = np.arange(1, 7)
    
    # Joint probability table
    joint_prob = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            joint_prob[i, j] = 1/36  # Each outcome has equal probability
    
    # Marginal probabilities
    marginal_dice1 = np.sum(joint_prob, axis=1)  # Sum over dice2
    marginal_dice2 = np.sum(joint_prob, axis=0)  # Sum over dice1
    
    return joint_prob, marginal_dice1, marginal_dice2

joint_prob, marg1, marg2 = joint_probability_example()

print("Joint Probability Table (Two Dice)")
print("Dice1\\Dice2 | 1    2    3    4    5    6")
print("-" * 40)
for i in range(6):
    row = f"   {i+1}     |"
    for j in range(6):
        row += f" {joint_prob[i,j]:.3f}"
    print(row)

print(f"\nMarginal Probabilities:")
print(f"Dice1: {marg1}")
print(f"Dice2: {marg2}")
```

### Conditional Probability

```python
def conditional_probability_example():
    """Example: Conditional probability with cards"""
    # Standard deck of 52 cards
    # Event A: Drawing a heart
    # Event B: Drawing a face card (J, Q, K)
    
    # Total cards
    total_cards = 52
    
    # Hearts
    hearts = 13
    # Face cards
    face_cards = 12
    # Face cards that are hearts
    heart_face_cards = 3  # J♥, Q♥, K♥
    
    # Probabilities
    p_hearts = hearts / total_cards
    p_face_cards = face_cards / total_cards
    p_hearts_and_face = heart_face_cards / total_cards
    
    # Conditional probabilities
    p_hearts_given_face = p_hearts_and_face / p_face_cards
    p_face_given_hearts = p_hearts_and_face / p_hearts
    
    return {
        'P(Hearts)': p_hearts,
        'P(Face Cards)': p_face_cards,
        'P(Hearts ∩ Face Cards)': p_hearts_and_face,
        'P(Hearts|Face Cards)': p_hearts_given_face,
        'P(Face Cards|Hearts)': p_face_given_hearts
    }

cond_probs = conditional_probability_example()
print("Conditional Probability Example (Cards)")
for event, prob in cond_probs.items():
    print(f"{event}: {prob:.3f}")

# Independence check
p_hearts = cond_probs['P(Hearts)']
p_face = cond_probs['P(Face Cards)']
p_joint = cond_probs['P(Hearts ∩ Face Cards)']
print(f"\nIndependence Check:")
print(f"P(Hearts) × P(Face Cards) = {p_hearts * p_face:.3f}")
print(f"P(Hearts ∩ Face Cards) = {p_joint:.3f}")
print(f"Independent: {abs(p_hearts * p_face - p_joint) < 1e-10}")
```

## Bayes' Theorem

```python
def bayes_theorem_example():
    """Example: Medical diagnosis with Bayes' theorem"""
    # Disease prevalence: 1% of population has the disease
    p_disease = 0.01
    
    # Test accuracy
    p_positive_given_disease = 0.95    # Sensitivity
    p_negative_given_no_disease = 0.90  # Specificity
    
    # Calculate other probabilities
    p_no_disease = 1 - p_disease
    p_positive_given_no_disease = 1 - p_negative_given_no_disease
    
    # Prior probability
    prior = p_disease
    
    # Likelihood
    likelihood = p_positive_given_disease
    
    # Evidence (total probability of positive test)
    evidence = (p_positive_given_disease * p_disease + 
               p_positive_given_no_disease * p_no_disease)
    
    # Posterior probability using Bayes' theorem
    posterior = (likelihood * prior) / evidence
    
    return {
        'Prior P(Disease)': prior,
        'Likelihood P(Positive|Disease)': likelihood,
        'Evidence P(Positive)': evidence,
        'Posterior P(Disease|Positive)': posterior
    }

bayes_result = bayes_theorem_example()
print("Bayes' Theorem Example (Medical Diagnosis)")
for term, value in bayes_result.items():
    print(f"{term}: {value:.3f}")

# Demonstrate with different prior probabilities
def bayes_with_different_priors():
    """Show how posterior changes with different priors"""
    priors = [0.001, 0.01, 0.1, 0.5]
    sensitivity = 0.95
    specificity = 0.90
    
    results = []
    for prior in priors:
        likelihood = sensitivity
        evidence = (sensitivity * prior + (1 - specificity) * (1 - prior))
        posterior = (likelihood * prior) / evidence
        results.append((prior, posterior))
    
    return results

bayes_results = bayes_with_different_priors()
print(f"\nPosterior Probabilities with Different Priors:")
for prior, posterior in bayes_results:
    print(f"Prior: {prior:.3f} → Posterior: {posterior:.3f}")

# Visualize Bayes' theorem
plt.figure(figsize=(10, 6))
priors = np.linspace(0.001, 0.5, 100)
posteriors = []
sensitivity = 0.95
specificity = 0.90

for prior in priors:
    likelihood = sensitivity
    evidence = (sensitivity * prior + (1 - specificity) * (1 - prior))
    posterior = (likelihood * prior) / evidence
    posteriors.append(posterior)

plt.plot(priors, posteriors, 'b-', linewidth=2, label='Posterior')
plt.plot(priors, priors, 'r--', linewidth=2, label='Prior')
plt.xlabel('Prior Probability')
plt.ylabel('Posterior Probability')
plt.title("Bayes' Theorem: How Prior Affects Posterior")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Central Limit Theorem

```python
def central_limit_theorem_demonstration():
    """Demonstrate the Central Limit Theorem"""
    # Start with a non-normal distribution (exponential)
    population = np.random.exponential(2, 100000)
    
    # Take samples of different sizes
    sample_sizes = [1, 5, 10, 30, 100]
    sample_means = []
    
    for n in sample_sizes:
        means = []
        for _ in range(1000):
            sample = np.random.choice(population, size=n, replace=False)
            means.append(np.mean(sample))
        sample_means.append(means)
    
    return population, sample_means, sample_sizes

population, sample_means, sizes = central_limit_theorem_demonstration()

print("Central Limit Theorem Demonstration")
print(f"Population mean: {np.mean(population):.3f}")
print(f"Population std: {np.std(population):.3f}")

plt.figure(figsize=(15, 10))

# Population distribution
plt.subplot(2, 3, 1)
plt.hist(population, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Population Distribution (Exponential)')
plt.xlabel('Value')
plt.ylabel('Density')

# Sample means distributions
for i, (means, size) in enumerate(zip(sample_means, sizes)):
    plt.subplot(2, 3, i+2)
    plt.hist(means, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    
    # Overlay normal distribution
    mu = np.mean(means)
    sigma = np.std(means)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', linewidth=2, label='Normal approx.')
    
    plt.title(f'Sample Means (n={size})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()

# Verify CLT predictions
print(f"\nCLT Verification:")
for means, size in zip(sample_means, sizes):
    expected_std = np.std(population) / np.sqrt(size)
    actual_std = np.std(means)
    print(f"n={size}: Expected std = {expected_std:.3f}, Actual std = {actual_std:.3f}")
```

## Practical Applications

### Monte Carlo Simulation

```python
def monte_carlo_pi_estimation(n_points=10000):
    """Estimate π using Monte Carlo simulation"""
    # Generate random points in a 2x2 square
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Calculate distance from origin
    distances = np.sqrt(x**2 + y**2)
    
    # Points inside unit circle
    inside_circle = np.sum(distances <= 1)
    
    # Estimate π
    pi_estimate = 4 * inside_circle / n_points
    
    return x, y, distances, pi_estimate

x, y, distances, pi_est = monte_carlo_pi_estimation(100000)
print(f"Monte Carlo π Estimation: {pi_est:.6f}")
print(f"Actual π: {np.pi:.6f}")
print(f"Error: {abs(pi_est - np.pi):.6f}")

# Visualize
plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2)
plt.gca().add_patch(circle)

# Plot points
inside = distances <= 1
outside = distances > 1

plt.scatter(x[inside], y[inside], c='blue', alpha=0.6, s=1, label='Inside')
plt.scatter(x[outside], y[outside], c='gray', alpha=0.6, s=1, label='Outside')

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title(f'Monte Carlo π Estimation (π ≈ {pi_est:.4f})')
plt.legend()
plt.axis('equal')
plt.show()
```

### Confidence Intervals

```python
def confidence_interval_example():
    """Demonstrate confidence intervals"""
    # Generate population
    population = np.random.normal(100, 15, 10000)
    true_mean = np.mean(population)
    
    # Take multiple samples
    n_samples = 100
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

means, intervals, true_mean, count = confidence_interval_example()

print(f"Confidence Interval Example")
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

## Practice Problems

1. **Probability Calculations**: Create a function that calculates probabilities for various scenarios (e.g., card games, dice games).

2. **Distribution Fitting**: Implement a function that fits different distributions to data and selects the best fit using goodness-of-fit tests.

3. **Bayesian Inference**: Build a simple Bayesian classifier and compare it with frequentist approaches.

4. **Monte Carlo Methods**: Use Monte Carlo simulation to solve complex probability problems.

## Further Reading

- "Probability and Statistics for Engineering and the Sciences" by Jay L. Devore
- "Introduction to Probability" by Joseph K. Blitzstein and Jessica Hwang
- "Statistical Inference" by George Casella and Roger L. Berger
- "Bayesian Data Analysis" by Andrew Gelman et al.

## Key Takeaways

- **Probability** provides the foundation for statistical inference and machine learning
- **Random variables** can be discrete or continuous, each with their own properties
- **Probability distributions** model uncertainty in data and are essential for statistical modeling
- **Joint and conditional probabilities** help understand relationships between events
- **Bayes' theorem** is fundamental for updating beliefs with new evidence
- **Central Limit Theorem** explains why normal distributions are so common
- **Monte Carlo methods** provide powerful tools for solving complex probability problems

In the next chapter, we'll explore statistical inference, including hypothesis testing and confidence intervals. 