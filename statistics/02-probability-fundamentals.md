# Probability Fundamentals

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)

Probability theory is the mathematical foundation for statistics, machine learning, and data science. Understanding probability concepts is essential for making informed decisions under uncertainty.

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

Probability provides a mathematical framework for quantifying uncertainty. It allows us to make predictions and decisions in the face of incomplete information.

### Sample Space and Events

**Sample Space (Ω)**: The set of all possible outcomes of an experiment.

**Event**: A subset of the sample space.

**Mathematical Definition:**
- Sample Space: Ω = {ω₁, ω₂, ..., ωₙ}
- Event A: A ⊆ Ω
- Probability Function: P: 2^Ω → [0,1] satisfying:
  1. P(Ω) = 1
  2. P(A) ≥ 0 for all A ⊆ Ω
  3. P(A∪B) = P(A) + P(B) if A∩B = ∅ (additivity)

```python
def calculate_probability(event_outcomes, total_outcomes):
    """
    Calculate probability of an event
    
    Mathematical implementation:
    P(A) = |A| / |Ω|
    where |A| is the number of outcomes in event A
    and |Ω| is the total number of possible outcomes
    
    Parameters:
    event_outcomes: int, number of favorable outcomes
    total_outcomes: int, total number of possible outcomes
    
    Returns:
    float: probability between 0 and 1
    """
    if total_outcomes == 0:
        raise ValueError("Total outcomes cannot be zero")
    if event_outcomes < 0 or event_outcomes > total_outcomes:
        raise ValueError("Event outcomes must be between 0 and total outcomes")
    
    return event_outcomes / total_outcomes

# Example: Rolling a fair die
sample_space = {1, 2, 3, 4, 5, 6}  # Ω
event_even = {2, 4, 6}  # A = "rolling an even number"
event_odd = {1, 3, 5}   # B = "rolling an odd number"

p_even = calculate_probability(len(event_even), len(sample_space))
p_odd = calculate_probability(len(event_odd), len(sample_space))

print(f"Sample space: {sample_space}")
print(f"Event 'even numbers': {event_even}")
print(f"P(even) = {p_even:.3f}")
print(f"P(odd) = {p_odd:.3f}")
print(f"P(even) + P(odd) = {p_even + p_odd:.3f}")

# Verify probability axioms
print(f"P(Ω) = {calculate_probability(len(sample_space), len(sample_space)):.3f}")
print(f"P(∅) = {calculate_probability(0, len(sample_space)):.3f}")
```

### Conditional Probability

**Conditional Probability** measures the probability of an event given that another event has occurred.

**Mathematical Definition:**
$$P(A|B) = \frac{P(A \cap B)}{P(B)} \quad \text{where } P(B) > 0$$

**Interpretation:**
- P(A|B) is the probability of A occurring given that B has occurred
- This "updates" our belief about A based on the information that B occurred
- The formula can be rearranged to give: P(A∩B) = P(A|B) × P(B)

**Properties:**
1. **Range**: 0 ≤ P(A|B) ≤ 1
2. **Normalization**: P(Ω|B) = 1
3. **Additivity**: P(A∪C|B) = P(A|B) + P(C|B) if A∩C = ∅

```python
def conditional_probability(p_a_and_b, p_b):
    """
    Calculate conditional probability P(A|B)
    
    Mathematical implementation:
    P(A|B) = P(A∩B) / P(B)
    
    Parameters:
    p_a_and_b: float, P(A∩B)
    p_b: float, P(B)
    
    Returns:
    float: P(A|B)
    """
    if p_b == 0:
        raise ValueError("Cannot condition on event with zero probability")
    if p_a_and_b > p_b:
        raise ValueError("P(A∩B) cannot be greater than P(B)")
    
    return p_a_and_b / p_b

# Example: Medical test scenario
# P(disease) = 0.01 (1% of population has disease)
# P(positive test|disease) = 0.95 (95% sensitivity)
# P(negative test|no disease) = 0.90 (90% specificity)

p_disease = 0.01
p_positive_given_disease = 0.95
p_negative_given_no_disease = 0.90

# Calculate P(positive test and disease)
p_positive_and_disease = p_positive_given_disease * p_disease

# Calculate P(positive test)
p_positive = (p_positive_given_disease * p_disease + 
              (1 - p_negative_given_no_disease) * (1 - p_disease))

# Calculate P(disease|positive test) using Bayes' theorem
p_disease_given_positive = conditional_probability(p_positive_and_disease, p_positive)

print(f"P(disease) = {p_disease:.3f}")
print(f"P(positive|disease) = {p_positive_given_disease:.3f}")
print(f"P(negative|no disease) = {p_negative_given_no_disease:.3f}")
print(f"P(positive and disease) = {p_positive_and_disease:.4f}")
print(f"P(positive) = {p_positive:.4f}")
print(f"P(disease|positive) = {p_disease_given_positive:.4f}")

# Demonstrate the relationship
print(f"P(disease|positive) × P(positive) = {p_disease_given_positive * p_positive:.4f}")
print(f"P(positive|disease) × P(disease) = {p_positive_given_disease * p_disease:.4f}")
```

### Independence

Two events A and B are **independent** if the occurrence of one does not affect the probability of the other.

**Mathematical Definition:**
Events A and B are independent if and only if:
$$P(A \cap B) = P(A) \times P(B)$$

**Equivalent Definitions:**
1. P(A|B) = P(A) (if P(B) > 0)
2. P(B|A) = P(B) (if P(A) > 0)

**Properties:**
1. **Symmetry**: If A is independent of B, then B is independent of A
2. **Transitivity**: Independence is not transitive
3. **Complement**: If A and B are independent, then A and B^c are independent

```python
def check_independence(p_a, p_b, p_a_and_b, tolerance=1e-10):
    """
    Check if two events are independent
    
    Mathematical implementation:
    Events A and B are independent if P(A∩B) = P(A) × P(B)
    
    Parameters:
    p_a: float, P(A)
    p_b: float, P(B)
    p_a_and_b: float, P(A∩B)
    tolerance: float, numerical tolerance for comparison
    
    Returns:
    bool: True if events are independent
    """
    expected_p_a_and_b = p_a * p_b
    return abs(p_a_and_b - expected_p_a_and_b) < tolerance

# Example: Coin flips
p_heads_first = 0.5
p_heads_second = 0.5
p_both_heads = 0.25  # For fair coins

is_independent = check_independence(p_heads_first, p_heads_second, p_both_heads)
print(f"P(first flip heads) = {p_heads_first}")
print(f"P(second flip heads) = {p_heads_second}")
print(f"P(both heads) = {p_both_heads}")
print(f"P(first) × P(second) = {p_heads_first * p_heads_second}")
print(f"Events are independent: {is_independent}")

# Example: Dependent events (drawing cards without replacement)
# P(first card is ace) = 4/52
# P(second card is ace|first card is ace) = 3/51
# P(second card is ace|first card is not ace) = 4/51

p_first_ace = 4/52
p_second_ace_given_first_ace = 3/51
p_second_ace_given_first_not_ace = 4/51

# Calculate P(second ace)
p_second_ace = (p_second_ace_given_first_ace * p_first_ace + 
                p_second_ace_given_first_not_ace * (1 - p_first_ace))

# Calculate P(both aces)
p_both_aces = p_first_ace * p_second_ace_given_first_ace

is_independent_cards = check_independence(p_first_ace, p_second_ace, p_both_aces)
print(f"\nCard example:")
print(f"P(first ace) = {p_first_ace:.4f}")
print(f"P(second ace) = {p_second_ace:.4f}")
print(f"P(both aces) = {p_both_aces:.4f}")
print(f"P(first) × P(second) = {p_first_ace * p_second_ace:.4f}")
print(f"Events are independent: {is_independent_cards}")
```

### Law of Total Probability

The **Law of Total Probability** allows us to calculate the probability of an event by conditioning on a partition of the sample space.

**Mathematical Definition:**
If B₁, B₂, ..., Bₙ form a partition of Ω (i.e., they are mutually exclusive and exhaustive), then:
$$P(A) = \sum_{i=1}^{n} P(A|B_i) \times P(B_i)$$

**Special Case (Two Events):**
$$P(A) = P(A|B) \times P(B) + P(A|B^c) \times P(B^c)$$

**Applications:**
- Medical diagnosis
- Quality control
- Risk assessment

```python
def law_of_total_probability(conditional_probs, partition_probs):
    """
    Calculate probability using Law of Total Probability
    
    Mathematical implementation:
    P(A) = Σ P(A|B_i) × P(B_i)
    
    Parameters:
    conditional_probs: list, P(A|B_i) for each partition
    partition_probs: list, P(B_i) for each partition
    
    Returns:
    float: P(A)
    """
    if len(conditional_probs) != len(partition_probs):
        raise ValueError("Conditional and partition probabilities must have same length")
    
    if abs(sum(partition_probs) - 1.0) > 1e-10:
        raise ValueError("Partition probabilities must sum to 1")
    
    return sum(c * p for c, p in zip(conditional_probs, partition_probs))

# Example: Quality control
# Three machines produce widgets with different defect rates
# Machine 1: 60% of production, 2% defect rate
# Machine 2: 30% of production, 3% defect rate  
# Machine 3: 10% of production, 5% defect rate

production_shares = [0.60, 0.30, 0.10]  # P(B_i)
defect_rates = [0.02, 0.03, 0.05]       # P(defect|B_i)

overall_defect_rate = law_of_total_probability(defect_rates, production_shares)

print("Quality Control Example:")
for i, (share, rate) in enumerate(zip(production_shares, defect_rates)):
    print(f"Machine {i+1}: {share*100:.0f}% production, {rate*100:.1f}% defect rate")

print(f"Overall defect rate: {overall_defect_rate:.4f} ({overall_defect_rate*100:.2f}%)")

# Verification
manual_calculation = sum(rate * share for rate, share in zip(defect_rates, production_shares))
print(f"Manual calculation: {manual_calculation:.4f}")

# Example: Medical diagnosis with multiple symptoms
# P(disease) = 0.01
# P(symptom1|disease) = 0.8, P(symptom1|no disease) = 0.1
# P(symptom2|disease) = 0.6, P(symptom2|no disease) = 0.05

p_disease = 0.01
p_no_disease = 1 - p_disease

# Calculate P(symptom1)
p_symptom1_given_disease = 0.8
p_symptom1_given_no_disease = 0.1

p_symptom1 = law_of_total_probability(
    [p_symptom1_given_disease, p_symptom1_given_no_disease],
    [p_disease, p_no_disease]
)

print(f"\nMedical Diagnosis Example:")
print(f"P(disease) = {p_disease:.3f}")
print(f"P(symptom1|disease) = {p_symptom1_given_disease:.1f}")
print(f"P(symptom1|no disease) = {p_symptom1_given_no_disease:.1f}")
print(f"P(symptom1) = {p_symptom1:.4f}")
```

### Bayes' Theorem

**Bayes' Theorem** provides a way to update probabilities based on new evidence.

**Mathematical Definition:**
$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Where P(B) can be calculated using the Law of Total Probability:
$$P(B) = P(B|A) \times P(A) + P(B|A^c) \times P(A^c)$$

**Interpretation:**
- P(A): Prior probability (before evidence)
- P(A|B): Posterior probability (after evidence)
- P(B|A): Likelihood (how likely is the evidence given the hypothesis)
- P(B): Evidence (total probability of observing the evidence)

**Applications:**
- Medical diagnosis
- Spam filtering
- Machine learning (Naive Bayes)
- Bayesian inference

```python
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    """
    Calculate posterior probability using Bayes' theorem
    
    Mathematical implementation:
    P(A|B) = P(B|A) × P(A) / P(B)
    where P(B) = P(B|A) × P(A) + P(B|A^c) × P(A^c)
    
    Parameters:
    p_a: float, prior probability P(A)
    p_b_given_a: float, likelihood P(B|A)
    p_b_given_not_a: float, likelihood P(B|A^c)
    
    Returns:
    float: posterior probability P(A|B)
    """
    p_not_a = 1 - p_a
    
    # Calculate P(B) using Law of Total Probability
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    
    if p_b == 0:
        raise ValueError("Evidence probability cannot be zero")
    
    # Apply Bayes' theorem
    p_a_given_b = (p_b_given_a * p_a) / p_b
    
    return p_a_given_b

# Example: Medical test revisited
p_disease = 0.01  # Prior: 1% of population has disease
p_positive_given_disease = 0.95  # Sensitivity
p_positive_given_no_disease = 0.10  # 1 - Specificity

p_disease_given_positive = bayes_theorem(
    p_disease, p_positive_given_disease, p_positive_given_no_disease
)

print("Bayes' Theorem - Medical Test:")
print(f"Prior P(disease) = {p_disease:.3f}")
print(f"Likelihood P(positive|disease) = {p_positive_given_disease:.2f}")
print(f"Likelihood P(positive|no disease) = {p_positive_given_no_disease:.2f}")
print(f"Posterior P(disease|positive) = {p_disease_given_positive:.4f}")

# Demonstrate how prior affects posterior
priors = [0.001, 0.01, 0.1, 0.5]
print(f"\nEffect of Prior on Posterior:")
for prior in priors:
    posterior = bayes_theorem(prior, p_positive_given_disease, p_positive_given_no_disease)
    print(f"Prior: {prior:.3f} → Posterior: {posterior:.4f}")

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

## Random Variables

A **random variable** is a function that assigns a numerical value to each outcome in a sample space. Random variables are the foundation for probability distributions and statistical modeling.

### Discrete Random Variables

A **discrete random variable** takes on a countable number of distinct values.

**Mathematical Definition:**
A discrete random variable X is a function X: Ω → ℝ such that:
- The range of X is countable
- For each x ∈ ℝ, the set {ω ∈ Ω : X(ω) = x} is an event

**Probability Mass Function (PMF):**
$$p_X(x) = P(X = x)$$

**Properties of PMF:**
1. $p_X(x) \geq 0$ for all x
2. $\sum_{x} p_X(x) = 1$
3. $P(X \in A) = \sum_{x \in A} p_X(x)$

**Cumulative Distribution Function (CDF):**
$$F_X(x) = P(X \leq x) = \sum_{k \leq x} p_X(k)$$

**Expected Value (Mean):**
$$\mu = E[X] = \sum_{x} x \cdot p_X(x)$$

**Variance:**
$$\sigma^2 = \text{Var}(X) = E[(X - \mu)^2] = \sum_{x} (x - \mu)^2 \cdot p_X(x)$$

**Moment Generating Function (MGF):**
$$M_X(t) = E[e^{tX}] = \sum_{x} e^{tx} \cdot p_X(x)$$

**Properties of Expected Value:**
1. **Linearity**: E[aX + b] = aE[X] + b
2. **Additivity**: E[X + Y] = E[X] + E[Y] (if X, Y are independent)
3. **Monotonicity**: If X ≤ Y, then E[X] ≤ E[Y]

**Properties of Variance:**
1. **Non-negativity**: Var(X) ≥ 0
2. **Scale**: Var(aX + b) = a²Var(X)
3. **Additivity**: Var(X + Y) = Var(X) + Var(Y) (if X, Y are independent)

```python
def discrete_pmf(values, probabilities):
    """
    Create a discrete probability mass function
    
    Mathematical implementation:
    PMF: p_X(x) = P(X = x)
    
    Parameters:
    values: array-like, possible values of the random variable
    probabilities: array-like, corresponding probabilities
    
    Returns:
    dict: PMF as {value: probability}
    """
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have same length")
    
    if abs(sum(probabilities) - 1.0) > 1e-10:
        raise ValueError("Probabilities must sum to 1")
    
    if any(p < 0 for p in probabilities):
        raise ValueError("Probabilities must be non-negative")
    
    return dict(zip(values, probabilities))

def discrete_cdf(pmf, x):
    """
    Calculate cumulative distribution function for discrete random variable
    
    Mathematical implementation:
    CDF: F_X(x) = P(X ≤ x) = Σ_{k≤x} p_X(k)
    
    Parameters:
    pmf: dict, probability mass function
    x: float, point to evaluate CDF
    
    Returns:
    float: F_X(x)
    """
    return sum(prob for value, prob in pmf.items() if value <= x)

def discrete_expected_value(pmf):
    """
    Calculate expected value of discrete random variable
    
    Mathematical implementation:
    E[X] = Σ x × p_X(x)
    
    Parameters:
    pmf: dict, probability mass function
    
    Returns:
    float: expected value
    """
    return sum(value * prob for value, prob in pmf.items())

def discrete_variance(pmf):
    """
    Calculate variance of discrete random variable
    
    Mathematical implementation:
    Var(X) = E[(X - μ)²] = Σ (x - μ)² × p_X(x)
    
    Parameters:
    pmf: dict, probability mass function
    
    Returns:
    float: variance
    """
    mu = discrete_expected_value(pmf)
    return sum((value - mu)**2 * prob for value, prob in pmf.items())

def discrete_mgf(pmf, t):
    """
    Calculate moment generating function for discrete random variable
    
    Mathematical implementation:
    M_X(t) = E[e^(tX)] = Σ e^(tx) × p_X(x)
    
    Parameters:
    pmf: dict, probability mass function
    t: float, parameter for MGF
    
    Returns:
    float: M_X(t)
    """
    return sum(np.exp(t * value) * prob for value, prob in pmf.items())

def discrete_moments_from_mgf(pmf, k=4):
    """
    Calculate moments using moment generating function
    
    Mathematical implementation:
    E[X^k] = M_X^(k)(0) = d^k/dt^k M_X(t) |_{t=0}
    
    Parameters:
    pmf: dict, probability mass function
    k: int, order of moment
    
    Returns:
    list: moments from order 1 to k
    """
    moments = []
    for i in range(1, k + 1):
        # Numerical differentiation
        h = 1e-6
        if i == 1:
            moment = (discrete_mgf(pmf, h) - discrete_mgf(pmf, -h)) / (2 * h)
        elif i == 2:
            moment = (discrete_mgf(pmf, h) - 2 * discrete_mgf(pmf, 0) + discrete_mgf(pmf, -h)) / (h**2)
        else:
            # Higher order moments using finite differences
            moment = discrete_mgf(pmf, 0)  # Placeholder for higher orders
        moments.append(moment)
    return moments

# Example: Fair die
die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
die_pmf = discrete_pmf(die_values, die_probs)

print("Fair Die Example:")
print(f"PMF: {die_pmf}")
print(f"Expected value: {discrete_expected_value(die_pmf):.3f}")
print(f"Variance: {discrete_variance(die_pmf):.3f}")

# Calculate CDF at various points
for x in [0, 1, 3.5, 6, 7]:
    cdf_val = discrete_cdf(die_pmf, x)
    print(f"F({x}) = {cdf_val:.3f}")

# Calculate MGF at t = 0.5
mgf_val = discrete_mgf(die_pmf, 0.5)
print(f"MGF at t=0.5: {mgf_val:.4f}")

# Calculate moments from MGF
moments = discrete_moments_from_mgf(die_pmf, 2)
print(f"First moment (mean): {moments[0]:.3f}")
print(f"Second moment: {moments[1]:.3f}")

# Example: Loaded die (biased towards 6)
loaded_values = [1, 2, 3, 4, 5, 6]
loaded_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # 50% chance of rolling 6
loaded_pmf = discrete_pmf(loaded_values, loaded_probs)

print(f"\nLoaded Die Example:")
print(f"PMF: {loaded_pmf}")
print(f"Expected value: {discrete_expected_value(loaded_pmf):.3f}")
print(f"Variance: {discrete_variance(loaded_pmf):.3f}")

# Compare fair vs loaded die
print(f"\nComparison:")
print(f"Fair die E[X]: {discrete_expected_value(die_pmf):.3f}")
print(f"Loaded die E[X]: {discrete_expected_value(loaded_pmf):.3f}")
print(f"Fair die Var(X): {discrete_variance(die_pmf):.3f}")
print(f"Loaded die Var(X): {discrete_variance(loaded_pmf):.3f}")

# Example: Bernoulli random variable (coin flip)
def bernoulli_pmf(p):
    """
    Create Bernoulli PMF
    
    Mathematical implementation:
    X ~ Bernoulli(p)
    p_X(0) = 1-p, p_X(1) = p
    E[X] = p, Var(X) = p(1-p)
    """
    return {0: 1-p, 1: p}

p = 0.7  # Probability of success
bernoulli = bernoulli_pmf(p)

print(f"\nBernoulli Example (p={p}):")
print(f"PMF: {bernoulli}")
print(f"E[X] = {discrete_expected_value(bernoulli):.3f}")
print(f"Var(X) = {discrete_variance(bernoulli):.3f}")

# Theoretical values for Bernoulli
print(f"Theoretical E[X] = p = {p:.3f}")
print(f"Theoretical Var(X) = p(1-p) = {p*(1-p):.3f}")

# Calculate MGF for Bernoulli
t_values = np.linspace(-2, 2, 100)
mgf_values = [discrete_mgf(bernoulli, t) for t in t_values]

# Visualize MGF
plt.figure(figsize=(12, 8))

# Plot 1: PMF
plt.subplot(2, 2, 1)
values = list(bernoulli.keys())
probs = list(bernoulli.values())
plt.bar(values, probs, alpha=0.7, color='skyblue', edgecolor='navy')
plt.xlabel('X')
plt.ylabel('P(X = x)')
plt.title('Bernoulli PMF')
plt.grid(True, alpha=0.3)

# Plot 2: CDF
plt.subplot(2, 2, 2)
x_range = np.linspace(-0.5, 1.5, 1000)
cdf_values = [discrete_cdf(bernoulli, x) for x in x_range]
plt.step(x_range, cdf_values, where='post', linewidth=2, color='green')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Bernoulli CDF')
plt.grid(True, alpha=0.3)

# Plot 3: MGF
plt.subplot(2, 2, 3)
plt.plot(t_values, mgf_values, 'r-', linewidth=2)
plt.xlabel('t')
plt.ylabel('M_X(t)')
plt.title('Bernoulli MGF')
plt.grid(True, alpha=0.3)

# Plot 4: Comparison of distributions
plt.subplot(2, 2, 4)
x_pos = np.arange(2)
fair_probs = [1/6] * 6
loaded_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]

plt.bar(x_pos - 0.2, [discrete_expected_value(die_pmf), discrete_variance(die_pmf)], 
        0.4, label='Fair Die', alpha=0.7)
plt.bar(x_pos + 0.2, [discrete_expected_value(loaded_pmf), discrete_variance(loaded_pmf)], 
        0.4, label='Loaded Die', alpha=0.7)
plt.xticks(x_pos, ['Mean', 'Variance'])
plt.ylabel('Value')
plt.title('Comparison of Moments')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Continuous Random Variables

A **continuous random variable** takes on values in a continuous range.

**Mathematical Definition:**
A continuous random variable X is a function X: Ω → ℝ such that:
- The range of X is uncountable
- There exists a function f_X(x) ≥ 0 such that:
  $$P(X \in A) = \int_A f_X(x) dx$$

**Probability Density Function (PDF):**
$$f_X(x) = \frac{d}{dx} F_X(x)$$

**Properties of PDF:**
1. $f_X(x) \geq 0$ for all x
2. $\int_{-\infty}^{\infty} f_X(x) dx = 1$
3. $P(X \in A) = \int_A f_X(x) dx$

**Cumulative Distribution Function (CDF):**
$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt$$

**Expected Value:**
$$\mu = E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**Variance:**
$$\sigma^2 = \text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f_X(x) dx$$

**Moment Generating Function:**
$$M_X(t) = E[e^{tX}] = \int_{-\infty}^{\infty} e^{tx} \cdot f_X(x) dx$$

**Properties:**
1. **Linearity**: E[aX + b] = aE[X] + b
2. **Additivity**: E[X + Y] = E[X] + E[Y] (if X, Y are independent)
3. **Variance**: Var(aX + b) = a²Var(X)

```python
def continuous_pdf_check(pdf_func, a, b, tolerance=1e-6):
    """
    Check if a function is a valid PDF on interval [a, b]
    
    Mathematical implementation:
    Check: ∫_a^b f(x) dx = 1 and f(x) ≥ 0 for all x ∈ [a, b]
    
    Parameters:
    pdf_func: function, PDF to check
    a, b: float, interval bounds
    tolerance: float, numerical tolerance
    
    Returns:
    bool: True if valid PDF
    """
    from scipy.integrate import quad
    
    # Check if integral equals 1
    integral, _ = quad(pdf_func, a, b)
    if abs(integral - 1.0) > tolerance:
        return False
    
    # Check if function is non-negative (sample points)
    x_samples = np.linspace(a, b, 1000)
    if any(pdf_func(x) < -tolerance for x in x_samples):
        return False
    
    return True

def continuous_cdf(pdf_func, x, a, b):
    """
    Calculate CDF for continuous random variable
    
    Mathematical implementation:
    F_X(x) = ∫_a^x f_X(t) dt
    
    Parameters:
    pdf_func: function, PDF
    x: float, point to evaluate
    a, b: float, support interval
    
    Returns:
    float: F_X(x)
    """
    from scipy.integrate import quad
    
    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        integral, _ = quad(pdf_func, a, x)
        return integral

def continuous_expected_value(pdf_func, a, b):
    """
    Calculate expected value of continuous random variable
    
    Mathematical implementation:
    E[X] = ∫_a^b x × f_X(x) dx
    
    Parameters:
    pdf_func: function, PDF
    a, b: float, support interval
    
    Returns:
    float: expected value
    """
    from scipy.integrate import quad
    
    def integrand(x):
        return x * pdf_func(x)
    
    integral, _ = quad(integrand, a, b)
    return integral

def continuous_variance(pdf_func, a, b):
    """
    Calculate variance of continuous random variable
    
    Mathematical implementation:
    Var(X) = E[(X - μ)²] = ∫_a^b (x - μ)² × f_X(x) dx
    
    Parameters:
    pdf_func: function, PDF
    a, b: float, support interval
    
    Returns:
    float: variance
    """
    from scipy.integrate import quad
    
    mu = continuous_expected_value(pdf_func, a, b)
    
    def integrand(x):
        return (x - mu)**2 * pdf_func(x)
    
    integral, _ = quad(integrand, a, b)
    return integral

def continuous_mgf(pdf_func, t, a, b):
    """
    Calculate moment generating function for continuous random variable
    
    Mathematical implementation:
    M_X(t) = E[e^(tX)] = ∫_a^b e^(tx) × f_X(x) dx
    
    Parameters:
    pdf_func: function, PDF
    t: float, parameter for MGF
    a, b: float, support interval
    
    Returns:
    float: M_X(t)
    """
    from scipy.integrate import quad
    
    def integrand(x):
        return np.exp(t * x) * pdf_func(x)
    
    integral, _ = quad(integrand, a, b)
    return integral

# Example: Uniform distribution on [0, 1]
def uniform_pdf(x):
    """PDF of uniform distribution on [0, 1]"""
    if 0 <= x <= 1:
        return 1.0
    else:
        return 0.0

print("Uniform Distribution Example [0, 1]:")
print(f"Valid PDF: {continuous_pdf_check(uniform_pdf, 0, 1)}")

# Calculate CDF at various points
for x in [-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5]:
    cdf_val = continuous_cdf(uniform_pdf, x, 0, 1)
    print(f"F({x}) = {cdf_val:.3f}")

# Calculate moments
uniform_mean = continuous_expected_value(uniform_pdf, 0, 1)
uniform_var = continuous_variance(uniform_pdf, 0, 1)

print(f"E[X] = {uniform_mean:.3f}")
print(f"Var(X) = {uniform_var:.3f}")

# Theoretical values for uniform [0, 1]
print(f"Theoretical E[X] = (a+b)/2 = 0.5")
print(f"Theoretical Var(X) = (b-a)²/12 = {1/12:.3f}")

# Calculate MGF
t_values = np.linspace(-2, 2, 100)
mgf_values = [continuous_mgf(uniform_pdf, t, 0, 1) for t in t_values]

# Example: Exponential distribution
def exponential_pdf(x, lambda_param=1.0):
    """PDF of exponential distribution with rate λ"""
    if x >= 0:
        return lambda_param * np.exp(-lambda_param * x)
    else:
        return 0.0

lambda_val = 2.0
exp_pdf = lambda x: exponential_pdf(x, lambda_val)

print(f"\nExponential Distribution Example (λ={lambda_val}):")
print(f"Valid PDF: {continuous_pdf_check(exp_pdf, 0, np.inf)}")

# Calculate CDF at various points
for x in [0, 0.5, 1, 2, 3]:
    cdf_val = continuous_cdf(exp_pdf, x, 0, np.inf)
    print(f"F({x}) = {cdf_val:.3f}")

# Calculate moments
exp_mean = continuous_expected_value(exp_pdf, 0, np.inf)
exp_var = continuous_variance(exp_pdf, 0, np.inf)

print(f"E[X] = {exp_mean:.3f}")
print(f"Var(X) = {exp_var:.3f}")

# Theoretical values for exponential
print(f"Theoretical E[X] = 1/λ = {1/lambda_val:.3f}")
print(f"Theoretical Var(X) = 1/λ² = {1/lambda_val**2:.3f}")

# Example: Custom PDF (triangular distribution)
def triangular_pdf(x):
    """PDF of triangular distribution on [0, 2] with peak at 1"""
    if 0 <= x <= 1:
        return x
    elif 1 <= x <= 2:
        return 2 - x
    else:
        return 0.0

print(f"\nTriangular Distribution Example [0, 2]:")
print(f"Valid PDF: {continuous_pdf_check(triangular_pdf, 0, 2)}")

# Calculate moments
tri_mean = continuous_expected_value(triangular_pdf, 0, 2)
tri_var = continuous_variance(triangular_pdf, 0, 2)

print(f"E[X] = {tri_mean:.3f}")
print(f"Var(X) = {tri_var:.3f}")

# Calculate CDF at various points
for x in [0, 0.5, 1, 1.5, 2]:
    cdf_val = continuous_cdf(triangular_pdf, x, 0, 2)
    print(f"F({x}) = {cdf_val:.3f}")

# Visualize continuous distributions
plt.figure(figsize=(15, 10))

# Plot 1: Uniform PDF and CDF
plt.subplot(2, 3, 1)
x_uniform = np.linspace(-0.5, 1.5, 1000)
y_pdf = [uniform_pdf(x) for x in x_uniform]
y_cdf = [continuous_cdf(uniform_pdf, x, 0, 1) for x in x_uniform]

plt.plot(x_uniform, y_pdf, 'b-', linewidth=2, label='PDF')
plt.plot(x_uniform, y_cdf, 'r--', linewidth=2, label='CDF')
plt.xlabel('x')
plt.ylabel('f(x), F(x)')
plt.title('Uniform Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Exponential PDF and CDF
plt.subplot(2, 3, 2)
x_exp = np.linspace(0, 4, 1000)
y_pdf_exp = [exp_pdf(x) for x in x_exp]
y_cdf_exp = [continuous_cdf(exp_pdf, x, 0, np.inf) for x in x_exp]

plt.plot(x_exp, y_pdf_exp, 'b-', linewidth=2, label='PDF')
plt.plot(x_exp, y_cdf_exp, 'r--', linewidth=2, label='CDF')
plt.xlabel('x')
plt.ylabel('f(x), F(x)')
plt.title('Exponential Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Triangular PDF and CDF
plt.subplot(2, 3, 3)
x_tri = np.linspace(-0.5, 2.5, 1000)
y_pdf_tri = [triangular_pdf(x) for x in x_tri]
y_cdf_tri = [continuous_cdf(triangular_pdf, x, 0, 2) for x in x_tri]

plt.plot(x_tri, y_pdf_tri, 'b-', linewidth=2, label='PDF')
plt.plot(x_tri, y_cdf_tri, 'r--', linewidth=2, label='CDF')
plt.xlabel('x')
plt.ylabel('f(x), F(x)')
plt.title('Triangular Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: MGF comparison
plt.subplot(2, 3, 4)
t_range = np.linspace(-1, 1, 100)
mgf_uniform = [continuous_mgf(uniform_pdf, t, 0, 1) for t in t_range]
mgf_exp = [continuous_mgf(exp_pdf, t, 0, np.inf) for t in t_range]

plt.plot(t_range, mgf_uniform, 'b-', linewidth=2, label='Uniform')
plt.plot(t_range, mgf_exp, 'r-', linewidth=2, label='Exponential')
plt.xlabel('t')
plt.ylabel('M_X(t)')
plt.title('Moment Generating Functions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Moments comparison
plt.subplot(2, 3, 5)
distributions = ['Uniform', 'Exponential', 'Triangular']
means = [uniform_mean, exp_mean, tri_mean]
variances = [uniform_var, exp_var, tri_var]

x_pos = np.arange(len(distributions))
plt.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7)
plt.bar(x_pos + 0.2, variances, 0.4, label='Variance', alpha=0.7)
plt.xticks(x_pos, distributions)
plt.ylabel('Value')
plt.title('Moments Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Probability calculations
plt.subplot(2, 3, 6)
# P(0.3 ≤ X ≤ 0.7) for uniform
p_uniform = continuous_cdf(uniform_pdf, 0.7, 0, 1) - continuous_cdf(uniform_pdf, 0.3, 0, 1)
# P(X > 1) for exponential
p_exp = 1 - continuous_cdf(exp_pdf, 1, 0, np.inf)
# P(0.5 ≤ X ≤ 1.5) for triangular
p_tri = continuous_cdf(triangular_pdf, 1.5, 0, 2) - continuous_cdf(triangular_pdf, 0.5, 0, 2)

probabilities = [p_uniform, p_exp, p_tri]
plt.bar(distributions, probabilities, alpha=0.7, color=['blue', 'red', 'green'])
plt.ylabel('Probability')
plt.title('Example Probabilities')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate mathematical properties
print(f"\nMathematical Properties Verification:")
print(f"Uniform distribution:")
print(f"  E[X] = 0.5, Var(X) = 1/12 ≈ 0.0833")
print(f"  Calculated: E[X] = {uniform_mean:.4f}, Var(X) = {uniform_var:.4f}")

print(f"\nExponential distribution:")
print(f"  E[X] = 1/λ = {1/lambda_val:.3f}, Var(X) = 1/λ² = {1/lambda_val**2:.3f}")
print(f"  Calculated: E[X] = {exp_mean:.4f}, Var(X) = {exp_var:.4f}")

print(f"\nTriangular distribution:")
print(f"  Theoretical: E[X] = 1, Var(X) = 1/6 ≈ 0.1667")
print(f"  Calculated: E[X] = {tri_mean:.4f}, Var(X) = {tri_var:.4f}")
```

### Joint Random Variables

When we have multiple random variables, we can study their joint behavior.

**Joint PMF (Discrete):**
$$p_{X,Y}(x,y) = P(X = x, Y = y)$$

**Joint PDF (Continuous):**
$$f_{X,Y}(x,y) = \frac{\partial^2}{\partial x \partial y} F_{X,Y}(x,y)$$

**Marginal Distributions:**
- Discrete: $p_X(x) = \sum_y p_{X,Y}(x,y)$
- Continuous: $f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) dy$

**Independence:**
X and Y are independent if:
- Discrete: $p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y)$
- Continuous: $f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)$

**Covariance:**
$$\text{Cov}(X,Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

**Correlation:**
$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Properties of Covariance:**
1. **Symmetry**: Cov(X,Y) = Cov(Y,X)
2. **Linearity**: Cov(aX + b, cY + d) = ac Cov(X,Y)
3. **Additivity**: Cov(X + Y, Z) = Cov(X,Z) + Cov(Y,Z)
4. **Independence**: If X, Y independent, then Cov(X,Y) = 0

**Properties of Correlation:**
1. **Range**: -1 ≤ ρ ≤ 1
2. **Linear Relationship**: |ρ| = 1 if and only if Y = aX + b
3. **Independence**: If X, Y independent, then ρ = 0 (but not conversely)

```python
def joint_pmf_discrete(x_values, y_values, joint_probs):
    """
    Create joint PMF for discrete random variables
    
    Mathematical implementation:
    p_{X,Y}(x,y) = P(X = x, Y = y)
    
    Parameters:
    x_values: array-like, values of X
    y_values: array-like, values of Y
    joint_probs: 2D array, joint probabilities
    
    Returns:
    dict: joint PMF as {(x, y): probability}
    """
    if joint_probs.shape != (len(x_values), len(y_values)):
        raise ValueError("Joint probabilities must match dimensions")
    
    if abs(np.sum(joint_probs) - 1.0) > 1e-10:
        raise ValueError("Joint probabilities must sum to 1")
    
    joint_pmf = {}
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            joint_pmf[(x, y)] = joint_probs[i, j]
    
    return joint_pmf

def marginal_pmf_x(joint_pmf):
    """
    Calculate marginal PMF of X
    
    Mathematical implementation:
    p_X(x) = Σ_y p_{X,Y}(x,y)
    
    Parameters:
    joint_pmf: dict, joint PMF
    
    Returns:
    dict: marginal PMF of X
    """
    marginal = {}
    for (x, y), prob in joint_pmf.items():
        if x in marginal:
            marginal[x] += prob
        else:
            marginal[x] = prob
    return marginal

def marginal_pmf_y(joint_pmf):
    """
    Calculate marginal PMF of Y
    
    Mathematical implementation:
    p_Y(y) = Σ_x p_{X,Y}(x,y)
    
    Parameters:
    joint_pmf: dict, joint PMF
    
    Returns:
    dict: marginal PMF of Y
    """
    marginal = {}
    for (x, y), prob in joint_pmf.items():
        if y in marginal:
            marginal[y] += prob
        else:
            marginal[y] = prob
    return marginal

def check_independence_discrete(joint_pmf):
    """
    Check if discrete random variables are independent
    
    Mathematical implementation:
    Check if p_{X,Y}(x,y) = p_X(x) × p_Y(y) for all (x,y)
    
    Parameters:
    joint_pmf: dict, joint PMF
    
    Returns:
    bool: True if independent
    """
    marginal_x = marginal_pmf_x(joint_pmf)
    marginal_y = marginal_pmf_y(joint_pmf)
    
    for (x, y), joint_prob in joint_pmf.items():
        expected_prob = marginal_x[x] * marginal_y[y]
        if abs(joint_prob - expected_prob) > 1e-10:
            return False
    return True

def covariance_discrete(joint_pmf):
    """
    Calculate covariance of discrete random variables
    
    Mathematical implementation:
    Cov(X,Y) = E[XY] - E[X]E[Y]
    
    Parameters:
    joint_pmf: dict, joint PMF
    
    Returns:
    float: covariance
    """
    # Calculate E[XY]
    e_xy = sum(x * y * prob for (x, y), prob in joint_pmf.items())
    
    # Calculate E[X] and E[Y]
    marginal_x = marginal_pmf_x(joint_pmf)
    marginal_y = marginal_pmf_y(joint_pmf)
    
    e_x = sum(x * prob for x, prob in marginal_x.items())
    e_y = sum(y * prob for y, prob in marginal_y.items())
    
    return e_xy - e_x * e_y

def correlation_discrete(joint_pmf):
    """
    Calculate correlation coefficient of discrete random variables
    
    Mathematical implementation:
    ρ = Cov(X,Y) / (σ_X × σ_Y)
    
    Parameters:
    joint_pmf: dict, joint PMF
    
    Returns:
    float: correlation coefficient
    """
    # Calculate covariance
    cov_xy = covariance_discrete(joint_pmf)
    
    # Calculate standard deviations
    marginal_x = marginal_pmf_x(joint_pmf)
    marginal_y = marginal_pmf_y(joint_pmf)
    
    e_x = sum(x * prob for x, prob in marginal_x.items())
    e_y = sum(y * prob for y, prob in marginal_y.items())
    
    var_x = sum((x - e_x)**2 * prob for x, prob in marginal_x.items())
    var_y = sum((y - e_y)**2 * prob for y, prob in marginal_y.items())
    
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    
    if std_x == 0 or std_y == 0:
        return 0
    
    return cov_xy / (std_x * std_y)

# Example: Two dice
x_values = [1, 2, 3, 4, 5, 6]  # First die
y_values = [1, 2, 3, 4, 5, 6]  # Second die

# Independent case: fair dice
joint_probs_indep = np.ones((6, 6)) / 36  # Each outcome has probability 1/36
joint_pmf_indep = joint_pmf_discrete(x_values, y_values, joint_probs_indep)

print("Independent Dice Example:")
print(f"Independent: {check_independence_discrete(joint_pmf_indep)}")
print(f"Covariance: {covariance_discrete(joint_pmf_indep):.6f}")
print(f"Correlation: {correlation_discrete(joint_pmf_indep):.6f}")

# Dependent case: sum of dice
def create_sum_dice_pmf():
    """Create joint PMF where Y = X + 1 (mod 6)"""
    joint_probs = np.zeros((6, 6))
    for i in range(6):
        j = (i + 1) % 6
        joint_probs[i, j] = 1/6
    return joint_pmf_discrete(x_values, y_values, joint_probs)

joint_pmf_dep = create_sum_dice_pmf()

print(f"\nDependent Dice Example:")
print(f"Independent: {check_independence_discrete(joint_pmf_dep)}")
print(f"Covariance: {covariance_discrete(joint_pmf_dep):.6f}")
print(f"Correlation: {correlation_discrete(joint_pmf_dep):.6f}")

# Calculate marginal distributions
marginal_x_indep = marginal_pmf_x(joint_pmf_indep)
marginal_y_indep = marginal_pmf_y(joint_pmf_indep)

print(f"\nMarginal distributions (independent case):")
print(f"P(X=1) = {marginal_x_indep[1]:.3f}")
print(f"P(Y=1) = {marginal_y_indep[1]:.3f}")
print(f"P(X=1, Y=1) = {joint_pmf_indep[(1, 1)]:.3f}")
print(f"P(X=1) × P(Y=1) = {marginal_x_indep[1] * marginal_y_indep[1]:.3f}")

# Demonstrate covariance properties
print(f"\nCovariance Properties:")
print(f"1. Symmetry: Cov(X,Y) = Cov(Y,X)")
print(f"   Independent case: {covariance_discrete(joint_pmf_indep):.6f}")

# Create a new joint PMF with linear transformation
def create_transformed_pmf(joint_pmf, a, b, c, d):
    """Create joint PMF for aX + b and cY + d"""
    new_joint_pmf = {}
    for (x, y), prob in joint_pmf.items():
        new_x = a * x + b
        new_y = c * y + d
        new_joint_pmf[(new_x, new_y)] = prob
    return new_joint_pmf

# Test linearity property: Cov(aX + b, cY + d) = ac Cov(X,Y)
a, b, c, d = 2, 1, 3, 2
transformed_pmf = create_transformed_pmf(joint_pmf_indep, a, b, c, d)
cov_transformed = covariance_discrete(transformed_pmf)
cov_original = covariance_discrete(joint_pmf_indep)
expected_cov = a * c * cov_original

print(f"2. Linearity: Cov({a}X + {b}, {c}Y + {d}) = {a}×{c}×Cov(X,Y)")
print(f"   Expected: {expected_cov:.6f}")
print(f"   Calculated: {cov_transformed:.6f}")
print(f"   Property holds: {abs(cov_transformed - expected_cov) < 1e-10}")

# Visualize joint distributions
plt.figure(figsize=(15, 10))

# Plot 1: Independent joint PMF
plt.subplot(2, 3, 1)
joint_matrix_indep = np.array([[joint_pmf_indep[(x, y)] for y in y_values] for x in x_values])
sns.heatmap(joint_matrix_indep, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=y_values, yticklabels=x_values)
plt.title('Independent Joint PMF')
plt.xlabel('Y')
plt.ylabel('X')

# Plot 2: Dependent joint PMF
plt.subplot(2, 3, 2)
joint_matrix_dep = np.array([[joint_pmf_dep[(x, y)] for y in y_values] for x in x_values])
sns.heatmap(joint_matrix_dep, annot=True, fmt='.3f', cmap='Reds', 
            xticklabels=y_values, yticklabels=x_values)
plt.title('Dependent Joint PMF')
plt.xlabel('Y')
plt.ylabel('X')

# Plot 3: Marginal distributions comparison
plt.subplot(2, 3, 3)
x_pos = np.arange(len(x_values))
marginal_x_vals = [marginal_x_indep[x] for x in x_values]
marginal_y_vals = [marginal_y_indep[y] for y in y_values]

plt.bar(x_pos - 0.2, marginal_x_vals, 0.4, label='P(X=x)', alpha=0.7)
plt.bar(x_pos + 0.2, marginal_y_vals, 0.4, label='P(Y=y)', alpha=0.7)
plt.xticks(x_pos, x_values)
plt.ylabel('Probability')
plt.title('Marginal Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Independence verification
plt.subplot(2, 3, 4)
# Check independence for each (x,y) pair
independence_errors = []
for x in x_values:
    for y in y_values:
        joint_prob = joint_pmf_indep[(x, y)]
        marginal_product = marginal_x_indep[x] * marginal_y_indep[y]
        error = abs(joint_prob - marginal_product)
        independence_errors.append(error)

plt.hist(independence_errors, bins=10, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('|P(X,Y) - P(X)P(Y)|')
plt.ylabel('Frequency')
plt.title('Independence Verification')
plt.grid(True, alpha=0.3)

# Plot 5: Covariance vs correlation
plt.subplot(2, 3, 5)
cases = ['Independent', 'Dependent']
covariances = [covariance_discrete(joint_pmf_indep), covariance_discrete(joint_pmf_dep)]
correlations = [correlation_discrete(joint_pmf_indep), correlation_discrete(joint_pmf_dep)]

x_pos = np.arange(len(cases))
plt.bar(x_pos - 0.2, covariances, 0.4, label='Covariance', alpha=0.7)
plt.bar(x_pos + 0.2, correlations, 0.4, label='Correlation', alpha=0.7)
plt.xticks(x_pos, cases)
plt.ylabel('Value')
plt.title('Covariance vs Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Joint vs marginal relationship
plt.subplot(2, 3, 6)
# Show that sum of joint PMF equals marginal PMF
joint_sums_x = [sum(joint_pmf_indep[(x, y)] for y in y_values) for x in x_values]
marginal_x_check = [marginal_x_indep[x] for x in x_values]

plt.plot(x_values, joint_sums_x, 'bo-', label='Sum of joint PMF', linewidth=2)
plt.plot(x_values, marginal_x_check, 'ro-', label='Marginal PMF', linewidth=2)
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Joint → Marginal Relationship')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate mathematical relationships
print(f"\nMathematical Relationships:")
print(f"1. Marginal from Joint: Σ_y P(X=x, Y=y) = P(X=x)")
print(f"   Verification: {all(abs(joint_sums_x[i] - marginal_x_check[i]) < 1e-10 for i in range(len(x_values)))}")

print(f"\n2. Independence: P(X=x, Y=y) = P(X=x) × P(Y=y)")
print(f"   Independent case: {check_independence_discrete(joint_pmf_indep)}")
print(f"   Dependent case: {check_independence_discrete(joint_pmf_dep)}")

print(f"\n3. Correlation bounds: -1 ≤ ρ ≤ 1")
print(f"   Independent: ρ = {correlation_discrete(joint_pmf_indep):.6f}")
print(f"   Dependent: ρ = {correlation_discrete(joint_pmf_dep):.6f}")
```

## Probability Distributions

Probability distributions describe how probabilities are distributed over the possible values of a random variable. Understanding these distributions is crucial for statistical modeling, hypothesis testing, and machine learning.

### Discrete Distributions

#### Binomial Distribution

The **binomial distribution** models the number of successes in a fixed number of independent Bernoulli trials.

**Mathematical Definition:**
$$X \sim \text{Binomial}(n, p)$$

**Probability Mass Function:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n$$

Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

**Properties:**
1. **Support**: {0, 1, 2, ..., n}
2. **Parameters**: n (number of trials), p (probability of success)
3. **Mean**: E[X] = np
4. **Variance**: Var(X) = np(1-p)
5. **Moment Generating Function**: M(t) = (pe^t + (1-p))^n

**Derivation of Mean:**
$$E[X] = \sum_{k=0}^{n} k \binom{n}{k} p^k (1-p)^{n-k}$$
$$= \sum_{k=1}^{n} k \frac{n!}{k!(n-k)!} p^k (1-p)^{n-k}$$
$$= np \sum_{k=1}^{n} \frac{(n-1)!}{(k-1)!(n-k)!} p^{k-1} (1-p)^{n-k}$$
$$= np \sum_{j=0}^{n-1} \binom{n-1}{j} p^j (1-p)^{n-1-j} = np$$

**Applications:**
- Quality control (defective items)
- Medical trials (success/failure)
- Survey responses (yes/no)

```python
def binomial_pmf(k, n, p):
    """
    Calculate binomial PMF
    
    Mathematical implementation:
    P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
    
    Parameters:
    k: int, number of successes
    n: int, number of trials
    p: float, probability of success
    
    Returns:
    float: P(X = k)
    """
    if k < 0 or k > n:
        return 0.0
    
    # Calculate binomial coefficient
    from scipy.special import comb
    binomial_coeff = comb(n, k)
    
    return binomial_coeff * (p**k) * ((1-p)**(n-k))

def binomial_cdf(k, n, p):
    """
    Calculate binomial CDF
    
    Mathematical implementation:
    F(k) = P(X ≤ k) = Σ_{i=0}^k C(n,i) × p^i × (1-p)^(n-i)
    
    Parameters:
    k: int, number of successes
    n: int, number of trials
    p: float, probability of success
    
    Returns:
    float: P(X ≤ k)
    """
    return sum(binomial_pmf(i, n, p) for i in range(k + 1))

def binomial_moments(n, p):
    """
    Calculate binomial moments
    
    Mathematical implementation:
    E[X] = np
    Var(X) = np(1-p)
    
    Parameters:
    n: int, number of trials
    p: float, probability of success
    
    Returns:
    tuple: (mean, variance)
    """
    mean = n * p
    variance = n * p * (1 - p)
    return mean, variance

# Example: Coin flipping
n_trials = 10
p_success = 0.5

print("Binomial Distribution - Coin Flipping:")
print(f"Parameters: n={n_trials}, p={p_success}")

# Calculate PMF for all possible values
pmf_values = {}
for k in range(n_trials + 1):
    pmf_values[k] = binomial_pmf(k, n_trials, p_success)

print(f"PMF: {pmf_values}")

# Calculate moments
mean, variance = binomial_moments(n_trials, p_success)
print(f"E[X] = {mean:.3f}")
print(f"Var(X) = {variance:.3f}")

# Calculate specific probabilities
print(f"P(X = 5) = {binomial_pmf(5, n_trials, p_success):.4f}")
print(f"P(X ≤ 5) = {binomial_cdf(5, n_trials, p_success):.4f}")
print(f"P(X > 5) = {1 - binomial_cdf(5, n_trials, p_success):.4f}")

# Compare with theoretical values
theoretical_mean = n_trials * p_success
theoretical_var = n_trials * p_success * (1 - p_success)
print(f"Theoretical E[X] = {theoretical_mean:.3f}")
print(f"Theoretical Var(X) = {theoretical_var:.3f}")

# Example: Quality control
defective_rate = 0.05  # 5% defective items
sample_size = 20

print(f"\nQuality Control Example:")
print(f"Sample size: {sample_size}, Defective rate: {defective_rate:.3f}")

# Probability of finding exactly 2 defective items
prob_exactly_2 = binomial_pmf(2, sample_size, defective_rate)
print(f"P(exactly 2 defective) = {prob_exactly_2:.4f}")

# Probability of finding at most 2 defective items
prob_at_most_2 = binomial_cdf(2, sample_size, defective_rate)
print(f"P(at most 2 defective) = {prob_at_most_2:.4f}")

# Probability of finding more than 2 defective items
prob_more_than_2 = 1 - binomial_cdf(2, sample_size, defective_rate)
print(f"P(more than 2 defective) = {prob_more_than_2:.4f}")

# Expected number of defective items
expected_defective = sample_size * defective_rate
print(f"Expected defective items: {expected_defective:.2f}")

# Visualize the distribution
k_values = list(range(sample_size + 1))
probabilities = [binomial_pmf(k, sample_size, defective_rate) for k in k_values]

plt.figure(figsize=(10, 6))
plt.bar(k_values, probabilities, alpha=0.7, color='skyblue', edgecolor='navy')
plt.axvline(expected_defective, color='red', linestyle='--', 
            label=f'Mean = {expected_defective:.2f}')
plt.xlabel('Number of Defective Items')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={sample_size}, p={defective_rate})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Poisson Distribution

The **Poisson distribution** models the number of events occurring in a fixed interval of time or space.

**Mathematical Definition:**
$$X \sim \text{Poisson}(\lambda)$$

**Probability Mass Function:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

**Properties:**
1. **Support**: {0, 1, 2, ...}
2. **Parameter**: λ (rate parameter, λ > 0)
3. **Mean**: E[X] = λ
4. **Variance**: Var(X) = λ
5. **Moment Generating Function**: M(t) = e^{λ(e^t - 1)}

**Derivation of Mean:**
$$E[X] = \sum_{k=0}^{\infty} k \frac{\lambda^k e^{-\lambda}}{k!}$$
$$= \lambda e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!}$$
$$= \lambda e^{-\lambda} \sum_{j=0}^{\infty} \frac{\lambda^j}{j!} = \lambda e^{-\lambda} e^{\lambda} = \lambda$$

**Applications:**
- Arrival times (customers, calls)
- Rare events (accidents, mutations)
- Radioactive decay

```python
def poisson_pmf(k, lambda_param):
    """
    Calculate Poisson PMF
    
    Mathematical implementation:
    P(X = k) = (λ^k × e^(-λ)) / k!
    
    Parameters:
    k: int, number of events
    lambda_param: float, rate parameter
    
    Returns:
    float: P(X = k)
    """
    if k < 0:
        return 0.0
    
    from scipy.special import factorial
    return (lambda_param**k * np.exp(-lambda_param)) / factorial(k)

def poisson_cdf(k, lambda_param):
    """
    Calculate Poisson CDF
    
    Mathematical implementation:
    F(k) = P(X ≤ k) = Σ_{i=0}^k (λ^i × e^(-λ)) / i!
    
    Parameters:
    k: int, number of events
    lambda_param: float, rate parameter
    
    Returns:
    float: P(X ≤ k)
    """
    return sum(poisson_pmf(i, lambda_param) for i in range(k + 1))

def poisson_moments(lambda_param):
    """
    Calculate Poisson moments
    
    Mathematical implementation:
    E[X] = λ
    Var(X) = λ
    
    Parameters:
    lambda_param: float, rate parameter
    
    Returns:
    tuple: (mean, variance)
    """
    return lambda_param, lambda_param

# Example: Customer arrivals
arrival_rate = 3.0  # 3 customers per hour

print("Poisson Distribution - Customer Arrivals:")
print(f"Rate parameter: λ = {arrival_rate}")

# Calculate PMF for first 10 values
pmf_values = {}
for k in range(11):
    pmf_values[k] = poisson_pmf(k, arrival_rate)

print(f"PMF (first 11 values): {pmf_values}")

# Calculate moments
mean, variance = poisson_moments(arrival_rate)
print(f"E[X] = {mean:.3f}")
print(f"Var(X) = {variance:.3f}")

# Calculate specific probabilities
print(f"P(X = 0) = {poisson_pmf(0, arrival_rate):.4f}")
print(f"P(X = 1) = {poisson_pmf(1, arrival_rate):.4f}")
print(f"P(X ≤ 2) = {poisson_cdf(2, arrival_rate):.4f}")
print(f"P(X > 5) = {1 - poisson_cdf(5, arrival_rate):.4f}")

# Example: Rare disease cases
disease_rate = 0.1  # 0.1 cases per 1000 people

print(f"\nRare Disease Example:")
print(f"Disease rate: λ = {disease_rate} per 1000 people")

# Probability of exactly 0 cases
prob_no_cases = poisson_pmf(0, disease_rate)
print(f"P(no cases) = {prob_no_cases:.4f}")

# Probability of at least 1 case
prob_at_least_one = 1 - poisson_pmf(0, disease_rate)
print(f"P(at least 1 case) = {prob_at_least_one:.4f}")

# Expected number of cases
expected_cases = disease_rate
print(f"Expected cases: {expected_cases:.3f}")

# Visualize the distribution
k_values = list(range(10))
probabilities = [poisson_pmf(k, arrival_rate) for k in k_values]

plt.figure(figsize=(10, 6))
plt.bar(k_values, probabilities, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
plt.axvline(arrival_rate, color='red', linestyle='--', 
            label=f'Mean = {arrival_rate:.1f}')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ = {arrival_rate})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Continuous Distributions

#### Normal (Gaussian) Distribution

The **normal distribution** is the most important continuous distribution in statistics.

**Mathematical Definition:**
$$X \sim \mathcal{N}(\mu, \sigma^2)$$

**Probability Density Function:**
$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Properties:**
1. **Support**: (-∞, ∞)
2. **Parameters**: μ (mean), σ² (variance)
3. **Mean**: E[X] = μ
4. **Variance**: Var(X) = σ²
5. **Moment Generating Function**: M(t) = e^{μt + σ²t²/2}
6. **Symmetry**: f(μ + x) = f(μ - x)
7. **68-95-99.7 Rule**: P(μ-σ ≤ X ≤ μ+σ) ≈ 0.68, P(μ-2σ ≤ X ≤ μ+2σ) ≈ 0.95, P(μ-3σ ≤ X ≤ μ+3σ) ≈ 0.997

**Standard Normal Distribution:**
$$Z \sim \mathcal{N}(0, 1)$$
$$f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$$

**Applications:**
- Measurement errors
- Natural phenomena
- Central Limit Theorem
- Statistical inference

```python
def normal_pdf(x, mu, sigma):
    """
    Calculate normal PDF
    
    Mathematical implementation:
    f(x) = (1/(σ√(2π))) × e^(-((x-μ)²)/(2σ²))
    
    Parameters:
    x: float or array, point(s) to evaluate
    mu: float, mean
    sigma: float, standard deviation
    
    Returns:
    float or array: f(x)
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def normal_cdf(x, mu, sigma):
    """
    Calculate normal CDF
    
    Mathematical implementation:
    F(x) = P(X ≤ x) = ∫_{-∞}^x f(t) dt
    
    Parameters:
    x: float or array, point(s) to evaluate
    mu: float, mean
    sigma: float, standard deviation
    
    Returns:
    float or array: F(x)
    """
    from scipy.stats import norm
    return norm.cdf(x, mu, sigma)

def normal_moments(mu, sigma):
    """
    Calculate normal moments
    
    Mathematical implementation:
    E[X] = μ
    Var(X) = σ²
    
    Parameters:
    mu: float, mean
    sigma: float, standard deviation
    
    Returns:
    tuple: (mean, variance)
    """
    return mu, sigma**2

def standard_normal_pdf(z):
    """
    Calculate standard normal PDF
    
    Mathematical implementation:
    f(z) = (1/√(2π)) × e^(-z²/2)
    
    Parameters:
    z: float or array, point(s) to evaluate
    
    Returns:
    float or array: f(z)
    """
    return normal_pdf(z, 0, 1)

def standard_normal_cdf(z):
    """
    Calculate standard normal CDF
    
    Mathematical implementation:
    Φ(z) = P(Z ≤ z) = ∫_{-∞}^z f(t) dt
    
    Parameters:
    z: float or array, point(s) to evaluate
    
    Returns:
    float or array: Φ(z)
    """
    return normal_cdf(z, 0, 1)

# Example: Height distribution
mu_height = 170  # cm
sigma_height = 10  # cm

print("Normal Distribution - Height:")
print(f"Parameters: μ = {mu_height}, σ = {sigma_height}")

# Calculate moments
mean, variance = normal_moments(mu_height, sigma_height)
print(f"E[X] = {mean:.3f}")
print(f"Var(X) = {variance:.3f}")

# Calculate specific probabilities using 68-95-99.7 rule
prob_1sigma = normal_cdf(mu_height + sigma_height, mu_height, sigma_height) - \
              normal_cdf(mu_height - sigma_height, mu_height, sigma_height)
prob_2sigma = normal_cdf(mu_height + 2*sigma_height, mu_height, sigma_height) - \
              normal_cdf(mu_height - 2*sigma_height, mu_height, sigma_height)
prob_3sigma = normal_cdf(mu_height + 3*sigma_height, mu_height, sigma_height) - \
              normal_cdf(mu_height - 3*sigma_height, mu_height, sigma_height)

print(f"P(μ-σ ≤ X ≤ μ+σ) = {prob_1sigma:.3f}")
print(f"P(μ-2σ ≤ X ≤ μ+2σ) = {prob_2sigma:.3f}")
print(f"P(μ-3σ ≤ X ≤ μ+3σ) = {prob_3sigma:.3f}")

# Calculate specific probabilities
print(f"P(X ≤ 180) = {normal_cdf(180, mu_height, sigma_height):.4f}")
print(f"P(X > 160) = {1 - normal_cdf(160, mu_height, sigma_height):.4f}")
print(f"P(165 ≤ X ≤ 175) = {normal_cdf(175, mu_height, sigma_height) - normal_cdf(165, mu_height, sigma_height):.4f}")

# Standard normal distribution
print(f"\nStandard Normal Distribution:")
print(f"P(Z ≤ 1.96) = {standard_normal_cdf(1.96):.4f}")
print(f"P(Z > -1.96) = {1 - standard_normal_cdf(-1.96):.4f}")
print(f"P(-1.96 ≤ Z ≤ 1.96) = {standard_normal_cdf(1.96) - standard_normal_cdf(-1.96):.4f}")

# Visualize the distribution
x = np.linspace(mu_height - 4*sigma_height, mu_height + 4*sigma_height, 1000)
y = normal_pdf(x, mu_height, sigma_height)

plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Normal PDF')
plt.axvline(mu_height, color='red', linestyle='--', label=f'Mean = {mu_height}')
plt.fill_between(x, y, where=(x >= mu_height - sigma_height) & (x <= mu_height + sigma_height), 
                 alpha=0.3, color='green', label='68% (1σ)')
plt.fill_between(x, y, where=(x >= mu_height - 2*sigma_height) & (x <= mu_height + 2*sigma_height), 
                 alpha=0.2, color='yellow', label='95% (2σ)')
plt.xlabel('Height (cm)')
plt.ylabel('Probability Density')
plt.title(f'Normal Distribution (μ = {mu_height}, σ = {sigma_height})')
plt.legend()
plt.grid(True, alpha=0.3)

# Standard normal
plt.subplot(2, 1, 2)
z = np.linspace(-4, 4, 1000)
y_std = standard_normal_pdf(z)
plt.plot(z, y_std, 'g-', linewidth=2, label='Standard Normal PDF')
plt.axvline(0, color='red', linestyle='--', label='Mean = 0')
plt.fill_between(z, y_std, where=(z >= -1) & (z <= 1), 
                 alpha=0.3, color='green', label='68%')
plt.fill_between(z, y_std, where=(z >= -2) & (z <= 2), 
                 alpha=0.2, color='yellow', label='95%')
plt.xlabel('Z')
plt.ylabel('Probability Density')
plt.title('Standard Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Exponential Distribution

The **exponential distribution** models the time between events in a Poisson process.

**Mathematical Definition:**
$$X \sim \text{Exponential}(\lambda)$$

**Probability Density Function:**
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Properties:**
1. **Support**: [0, ∞)
2. **Parameter**: λ (rate parameter, λ > 0)
3. **Mean**: E[X] = 1/λ
4. **Variance**: Var(X) = 1/λ²
5. **Memoryless Property**: P(X > s + t | X > s) = P(X > t)

**Derivation of Mean:**
$$E[X] = \int_0^{\infty} x \lambda e^{-\lambda x} dx$$
$$= \lambda \int_0^{\infty} x e^{-\lambda x} dx$$
$$= \lambda \left[ -\frac{x}{\lambda} e^{-\lambda x} \right]_0^{\infty} + \lambda \int_0^{\infty} \frac{1}{\lambda} e^{-\lambda x} dx$$
$$= 0 + \int_0^{\infty} e^{-\lambda x} dx = \frac{1}{\lambda}$$

**Applications:**
- Time between arrivals
- Component lifetimes
- Radioactive decay

```python
def exponential_pdf(x, lambda_param):
    """
    Calculate exponential PDF
    
    Mathematical implementation:
    f(x) = λ × e^(-λx) for x ≥ 0
    
    Parameters:
    x: float or array, point(s) to evaluate
    lambda_param: float, rate parameter
    
    Returns:
    float or array: f(x)
    """
    if isinstance(x, (int, float)):
        return lambda_param * np.exp(-lambda_param * x) if x >= 0 else 0.0
    else:
        return np.where(x >= 0, lambda_param * np.exp(-lambda_param * x), 0.0)

def exponential_cdf(x, lambda_param):
    """
    Calculate exponential CDF
    
    Mathematical implementation:
    F(x) = P(X ≤ x) = 1 - e^(-λx) for x ≥ 0
    
    Parameters:
    x: float or array, point(s) to evaluate
    lambda_param: float, rate parameter
    
    Returns:
    float or array: F(x)
    """
    if isinstance(x, (int, float)):
        return 1 - np.exp(-lambda_param * x) if x >= 0 else 0.0
    else:
        return np.where(x >= 0, 1 - np.exp(-lambda_param * x), 0.0)

def exponential_moments(lambda_param):
    """
    Calculate exponential moments
    
    Mathematical implementation:
    E[X] = 1/λ
    Var(X) = 1/λ²
    
    Parameters:
    lambda_param: float, rate parameter
    
    Returns:
    tuple: (mean, variance)
    """
    mean = 1 / lambda_param
    variance = 1 / (lambda_param**2)
    return mean, variance

def exponential_memoryless_property(lambda_param, s, t):
    """
    Demonstrate memoryless property
    
    Mathematical implementation:
    P(X > s + t | X > s) = P(X > t)
    
    Parameters:
    lambda_param: float, rate parameter
    s: float, time already waited
    t: float, additional time
    
    Returns:
    tuple: (conditional_prob, unconditional_prob)
    """
    # P(X > s + t | X > s) = P(X > s + t) / P(X > s)
    prob_s_plus_t = 1 - exponential_cdf(s + t, lambda_param)
    prob_s = 1 - exponential_cdf(s, lambda_param)
    conditional_prob = prob_s_plus_t / prob_s
    
    # P(X > t)
    unconditional_prob = 1 - exponential_cdf(t, lambda_param)
    
    return conditional_prob, unconditional_prob

# Example: Time between customer arrivals
arrival_rate = 2.0  # 2 customers per hour

print("Exponential Distribution - Customer Arrivals:")
print(f"Rate parameter: λ = {arrival_rate}")

# Calculate moments
mean, variance = exponential_moments(arrival_rate)
print(f"E[X] = {mean:.3f} hours")
print(f"Var(X) = {variance:.3f} hours²")
print(f"Standard deviation = {np.sqrt(variance):.3f} hours")

# Calculate specific probabilities
print(f"P(X ≤ 0.5) = {exponential_cdf(0.5, arrival_rate):.4f}")
print(f"P(X > 1) = {1 - exponential_cdf(1, arrival_rate):.4f}")
print(f"P(0.5 ≤ X ≤ 1.5) = {exponential_cdf(1.5, arrival_rate) - exponential_cdf(0.5, arrival_rate):.4f}")

# Demonstrate memoryless property
s = 0.5  # Already waited 0.5 hours
t = 0.3  # Additional 0.3 hours

cond_prob, uncond_prob = exponential_memoryless_property(arrival_rate, s, t)
print(f"\nMemoryless Property:")
print(f"P(X > {s+t} | X > {s}) = {cond_prob:.4f}")
print(f"P(X > {t}) = {uncond_prob:.4f}")
print(f"Memoryless property holds: {abs(cond_prob - uncond_prob) < 1e-10}")

# Example: Component lifetime
failure_rate = 0.1  # 0.1 failures per year

print(f"\nComponent Lifetime Example:")
print(f"Failure rate: λ = {failure_rate} per year")

# Expected lifetime
expected_lifetime = 1 / failure_rate
print(f"Expected lifetime: {expected_lifetime:.1f} years")

# Probability of surviving more than 5 years
prob_survive_5 = 1 - exponential_cdf(5, failure_rate)
print(f"P(survive > 5 years) = {prob_survive_5:.4f}")

# Probability of failing within first year
prob_fail_1 = exponential_cdf(1, failure_rate)
print(f"P(fail within 1 year) = {prob_fail_1:.4f}")

# Visualize the distribution
x = np.linspace(0, 5, 1000)
y = exponential_pdf(x, arrival_rate)

plt.figure(figsize=(12, 8))

# PDF
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', linewidth=2, label='Exponential PDF')
plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.3f}')
plt.fill_between(x, y, where=(x >= 0) & (x <= 1), 
                 alpha=0.3, color='green', label='P(X ≤ 1)')
plt.xlabel('Time (hours)')
plt.ylabel('Probability Density')
plt.title(f'Exponential Distribution (λ = {arrival_rate})')
plt.legend()
plt.grid(True, alpha=0.3)

# CDF
plt.subplot(2, 1, 2)
y_cdf = exponential_cdf(x, arrival_rate)
plt.plot(x, y_cdf, 'g-', linewidth=2, label='Exponential CDF')
plt.axhline(0.632, color='red', linestyle='--', label='1 - 1/e ≈ 0.632')
plt.axvline(mean, color='red', linestyle=':', alpha=0.7)
plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Probability')
plt.title('Exponential CDF')
plt.legend()
plt.grid(True, alpha=0.3)

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