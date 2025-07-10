"""
Probability Fundamentals
=======================

This module implements core probability concepts including:
- Basic probability calculations
- Random variables (discrete and continuous)
- Probability distributions
- Bayes' theorem and Bayesian inference
- Central Limit Theorem
- Monte Carlo simulation
- Practical applications in data science

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, expon, gamma, chi2
import pandas as pd
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProbabilityCalculator:
    """
    A class for performing basic probability calculations and demonstrations.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)  # For reproducible results
    
    def coin_toss_experiment(self, n_tosses: int = 1000) -> Dict[str, Any]:
        """
        Simulate coin toss experiments to demonstrate basic probability concepts.
        
        Parameters:
        -----------
        n_tosses : int
            Number of coin tosses to simulate
            
        Returns:
        --------
        dict
            Results including probabilities and experimental outcomes
        """
        print("=== Coin Toss Experiment ===")
        print(f"Simulating {n_tosses} coin tosses...")
        
        # Simulate coin tosses (0 = tails, 1 = heads)
        tosses = self.rng.integers(0, 2, n_tosses)
        
        # Calculate experimental probabilities
        n_heads = np.sum(tosses)
        n_tails = n_tosses - n_heads
        
        p_heads_exp = n_heads / n_tosses
        p_tails_exp = n_tails / n_tosses
        
        print(f"Theoretical P(Heads) = 0.5")
        print(f"Experimental P(Heads) = {p_heads_exp:.4f}")
        print(f"Experimental P(Tails) = {p_tails_exp:.4f}")
        
        # Demonstrate law of large numbers
        cumulative_heads = np.cumsum(tosses)
        cumulative_probs = cumulative_heads / np.arange(1, n_tosses + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, n_tosses + 1), cumulative_probs, 
                alpha=0.7, label='Experimental P(Heads)')
        plt.axhline(y=0.5, color='red', linestyle='--', 
                   label='Theoretical P(Heads) = 0.5')
        plt.xlabel('Number of Tosses')
        plt.ylabel('Cumulative Probability of Heads')
        plt.title('Law of Large Numbers: Coin Toss Experiment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'n_tosses': n_tosses,
            'n_heads': n_heads,
            'n_tails': n_tails,
            'p_heads_exp': p_heads_exp,
            'p_tails_exp': p_tails_exp,
            'cumulative_probs': cumulative_probs
        }
    
    def card_deck_probabilities(self) -> Dict[str, float]:
        """
        Calculate various probabilities for a standard 52-card deck.
        
        Returns:
        --------
        dict
            Dictionary of calculated probabilities
        """
        print("\n=== Card Deck Probabilities ===")
        
        # Define deck characteristics
        total_cards = 52
        hearts = 13
        face_cards = 12  # Jack, Queen, King of each suit
        aces = 4
        
        # Calculate probabilities
        p_heart = hearts / total_cards
        p_face = face_cards / total_cards
        p_ace = aces / total_cards
        
        # Heart face cards (Jack, Queen, King of hearts)
        heart_face_cards = 3
        p_heart_and_face = heart_face_cards / total_cards
        
        # Using inclusion-exclusion principle
        p_heart_or_face = p_heart + p_face - p_heart_and_face
        
        print(f"P(Heart) = {hearts}/{total_cards} = {p_heart:.4f}")
        print(f"P(Face Card) = {face_cards}/{total_cards} = {p_face:.4f}")
        print(f"P(Heart ∩ Face Card) = {heart_face_cards}/{total_cards} = {p_heart_and_face:.4f}")
        print(f"P(Heart ∪ Face Card) = {p_heart:.4f} + {p_face:.4f} - {p_heart_and_face:.4f} = {p_heart_or_face:.4f}")
        
        # Conditional probability: P(Heart | Red)
        red_cards = 26  # Hearts + Diamonds
        p_heart_given_red = p_heart / (red_cards / total_cards)
        print(f"P(Heart | Red) = {p_heart:.4f} / {red_cards/total_cards:.4f} = {p_heart_given_red:.4f}")
        
        return {
            'p_heart': p_heart,
            'p_face': p_face,
            'p_heart_and_face': p_heart_and_face,
            'p_heart_or_face': p_heart_or_face,
            'p_heart_given_red': p_heart_given_red
        }
    
    def medical_diagnosis_example(self) -> Dict[str, float]:
        """
        Demonstrate Bayes' theorem using medical diagnosis example.
        
        Returns:
        --------
        dict
            Dictionary of calculated probabilities
        """
        print("\n=== Medical Diagnosis Example (Bayes' Theorem) ===")
        
        # Given probabilities
        p_disease = 0.01  # 1% of population has disease
        p_positive_given_disease = 0.95  # 95% sensitivity
        p_positive_given_no_disease = 0.05  # 5% false positive rate
        
        # Calculate P(Positive) using law of total probability
        p_positive = (p_positive_given_disease * p_disease + 
                     p_positive_given_no_disease * (1 - p_disease))
        
        # Calculate P(Disease | Positive) using Bayes' theorem
        p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
        
        print(f"P(Disease) = {p_disease:.3f}")
        print(f"P(Positive | Disease) = {p_positive_given_disease:.3f}")
        print(f"P(Positive | No Disease) = {p_positive_given_no_disease:.3f}")
        print(f"P(Positive) = {p_positive:.4f}")
        print(f"P(Disease | Positive) = {p_disease_given_positive:.4f}")
        print(f"Even with a positive test, only {p_disease_given_positive:.1%} chance of having the disease!")
        
        return {
            'p_disease': p_disease,
            'p_positive_given_disease': p_positive_given_disease,
            'p_positive_given_no_disease': p_positive_given_no_disease,
            'p_positive': p_positive,
            'p_disease_given_positive': p_disease_given_positive
        }


class RandomVariableSimulator:
    """
    A class for simulating and analyzing random variables.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def bernoulli_simulation(self, p: float = 0.5, n: int = 1000) -> Dict[str, Any]:
        """
        Simulate Bernoulli random variables and demonstrate their properties.
        
        Parameters:
        -----------
        p : float
            Probability of success
        n : int
            Number of trials
            
        Returns:
        --------
        dict
            Simulation results and theoretical values
        """
        print(f"\n=== Bernoulli Distribution Simulation (p={p}) ===")
        
        # Simulate Bernoulli trials
        trials = self.rng.binomial(1, p, n)
        
        # Calculate experimental statistics
        n_successes = np.sum(trials)
        p_success_exp = n_successes / n
        
        # Theoretical values
        mean_theoretical = p
        var_theoretical = p * (1 - p)
        
        # Experimental values
        mean_exp = np.mean(trials)
        var_exp = np.var(trials)
        
        print(f"Theoretical mean: {mean_theoretical:.4f}")
        print(f"Experimental mean: {mean_exp:.4f}")
        print(f"Theoretical variance: {var_theoretical:.4f}")
        print(f"Experimental variance: {var_exp:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(trials, bins=[-0.5, 0.5, 1.5], alpha=0.7, 
                label=f'Experimental (n={n})')
        plt.axhline(y=n*(1-p), color='red', linestyle='--', 
                   label=f'Theoretical P(X=0) = {1-p}')
        plt.axhline(y=n*p, color='green', linestyle='--', 
                   label=f'Theoretical P(X=1) = {p}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Bernoulli Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        cumulative_mean = np.cumsum(trials) / np.arange(1, n + 1)
        plt.plot(range(1, n + 1), cumulative_mean, alpha=0.7, 
                label='Experimental mean')
        plt.axhline(y=p, color='red', linestyle='--', 
                   label=f'Theoretical mean = {p}')
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Mean')
        plt.title('Convergence to Theoretical Mean')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'trials': trials,
            'p_success_exp': p_success_exp,
            'mean_theoretical': mean_theoretical,
            'mean_exp': mean_exp,
            'var_theoretical': var_theoretical,
            'var_exp': var_exp
        }
    
    def binomial_simulation(self, n: int = 20, p: float = 0.3, 
                          n_experiments: int = 1000) -> Dict[str, Any]:
        """
        Simulate binomial distribution and compare with theoretical values.
        
        Parameters:
        -----------
        n : int
            Number of trials
        p : float
            Probability of success
        n_experiments : int
            Number of experiments to run
            
        Returns:
        --------
        dict
            Simulation results and theoretical values
        """
        print(f"\n=== Binomial Distribution Simulation (n={n}, p={p}) ===")
        
        # Simulate binomial experiments
        successes = self.rng.binomial(n, p, n_experiments)
        
        # Calculate experimental statistics
        mean_exp = np.mean(successes)
        var_exp = np.var(successes)
        
        # Theoretical values
        mean_theoretical = n * p
        var_theoretical = n * p * (1 - p)
        
        print(f"Theoretical mean: {mean_theoretical:.2f}")
        print(f"Experimental mean: {mean_exp:.2f}")
        print(f"Theoretical variance: {var_theoretical:.2f}")
        print(f"Experimental variance: {var_exp:.2f}")
        
        # Plot histogram with theoretical PMF
        plt.figure(figsize=(12, 6))
        
        # Experimental histogram
        plt.hist(successes, bins=range(n+2), alpha=0.7, density=True,
                label=f'Experimental (n_exp={n_experiments})')
        
        # Theoretical PMF
        x_theoretical = np.arange(0, n + 1)
        pmf_theoretical = binom.pmf(x_theoretical, n, p)
        plt.plot(x_theoretical, pmf_theoretical, 'ro-', 
                label=f'Theoretical Binomial({n}, {p})')
        
        plt.xlabel('Number of Successes')
        plt.ylabel('Probability')
        plt.title('Binomial Distribution: Experimental vs Theoretical')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'successes': successes,
            'mean_theoretical': mean_theoretical,
            'mean_exp': mean_exp,
            'var_theoretical': var_theoretical,
            'var_exp': var_exp,
            'pmf_theoretical': pmf_theoretical
        }
    
    def normal_distribution_demo(self, mu: float = 0, sigma: float = 1, 
                               n_samples: int = 10000) -> Dict[str, Any]:
        """
        Demonstrate normal distribution properties and the 68-95-99.7 rule.
        
        Parameters:
        -----------
        mu : float
            Mean of the normal distribution
        sigma : float
            Standard deviation of the normal distribution
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        dict
            Simulation results and theoretical values
        """
        print(f"\n=== Normal Distribution Demo (μ={mu}, σ={sigma}) ===")
        
        # Generate samples
        samples = self.rng.normal(mu, sigma, n_samples)
        
        # Calculate experimental statistics
        mean_exp = np.mean(samples)
        std_exp = np.std(samples)
        
        print(f"Theoretical mean: {mu:.2f}")
        print(f"Experimental mean: {mean_exp:.4f}")
        print(f"Theoretical std: {sigma:.2f}")
        print(f"Experimental std: {std_exp:.4f}")
        
        # 68-95-99.7 rule demonstration
        within_1std = np.sum(np.abs(samples - mu) <= sigma) / n_samples
        within_2std = np.sum(np.abs(samples - mu) <= 2*sigma) / n_samples
        within_3std = np.sum(np.abs(samples - mu) <= 3*sigma) / n_samples
        
        print(f"\n68-95-99.7 Rule:")
        print(f"Within 1σ: {within_1std:.3f} (theoretical: 0.683)")
        print(f"Within 2σ: {within_2std:.3f} (theoretical: 0.954)")
        print(f"Within 3σ: {within_3std:.3f} (theoretical: 0.997)")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Histogram with normal curve
        plt.subplot(1, 3, 1)
        plt.hist(samples, bins=50, density=True, alpha=0.7, 
                label='Experimental')
        
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, 'r-', linewidth=2, label='Theoretical')
        
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Normal Distribution')
        plt.legend()
        
        # Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(samples, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal)')
        
        # Cumulative distribution
        plt.subplot(1, 3, 3)
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        plt.plot(sorted_samples, empirical_cdf, 'b-', alpha=0.7, 
                label='Empirical CDF')
        
        theoretical_cdf = norm.cdf(x, mu, sigma)
        plt.plot(x, theoretical_cdf, 'r-', linewidth=2, 
                label='Theoretical CDF')
        
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'samples': samples,
            'mean_exp': mean_exp,
            'std_exp': std_exp,
            'within_1std': within_1std,
            'within_2std': within_2std,
            'within_3std': within_3std
        }


class CentralLimitTheorem:
    """
    A class for demonstrating the Central Limit Theorem through simulations.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def clt_uniform_demo(self, n_samples: int = 1000, 
                         sample_sizes: List[int] = [1, 5, 10, 30]) -> None:
        """
        Demonstrate CLT using uniform distribution.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        sample_sizes : List[int]
            Different sample sizes to demonstrate convergence
        """
        print("\n=== Central Limit Theorem: Uniform Distribution ===")
        
        # Theoretical values for uniform(0,1)
        mu_uniform = 0.5
        sigma_uniform = np.sqrt(1/12)  # sqrt(1/12) for uniform(0,1)
        
        plt.figure(figsize=(15, 10))
        
        for i, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = []
            for _ in range(n_samples):
                samples = self.rng.uniform(0, 1, n)
                sample_means.append(np.mean(samples))
            
            sample_means = np.array(sample_means)
            
            # Theoretical normal distribution for sample mean
            mu_sample = mu_uniform
            sigma_sample = sigma_uniform / np.sqrt(n)
            
            plt.subplot(2, 2, i+1)
            plt.hist(sample_means, bins=30, density=True, alpha=0.7,
                    label=f'Experimental (n={n})')
            
            # Plot theoretical normal distribution
            x = np.linspace(mu_sample - 3*sigma_sample, 
                           mu_sample + 3*sigma_sample, 100)
            y = norm.pdf(x, mu_sample, sigma_sample)
            plt.plot(x, y, 'r-', linewidth=2, 
                    label=f'Theoretical N({mu_sample:.2f}, {sigma_sample:.3f}²)')
            
            plt.xlabel('Sample Mean')
            plt.ylabel('Density')
            plt.title(f'Sample Size n = {n}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Theoretical mean of uniform(0,1): {mu_uniform}")
        print(f"Theoretical std of uniform(0,1): {sigma_uniform:.4f}")
        print("As sample size increases, the distribution of sample means")
        print("converges to a normal distribution!")
    
    def clt_exponential_demo(self, n_samples: int = 1000,
                           sample_sizes: List[int] = [1, 5, 10, 30]) -> None:
        """
        Demonstrate CLT using exponential distribution (shows slower convergence).
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        sample_sizes : List[int]
            Different sample sizes to demonstrate convergence
        """
        print("\n=== Central Limit Theorem: Exponential Distribution ===")
        
        # Theoretical values for exponential(1)
        lambda_param = 1
        mu_exp = 1/lambda_param
        sigma_exp = 1/lambda_param
        
        plt.figure(figsize=(15, 10))
        
        for i, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = []
            for _ in range(n_samples):
                samples = self.rng.exponential(1/lambda_param, n)
                sample_means.append(np.mean(samples))
            
            sample_means = np.array(sample_means)
            
            # Theoretical normal distribution for sample mean
            mu_sample = mu_exp
            sigma_sample = sigma_exp / np.sqrt(n)
            
            plt.subplot(2, 2, i+1)
            plt.hist(sample_means, bins=30, density=True, alpha=0.7,
                    label=f'Experimental (n={n})')
            
            # Plot theoretical normal distribution
            x = np.linspace(mu_sample - 3*sigma_sample, 
                           mu_sample + 3*sigma_sample, 100)
            y = norm.pdf(x, mu_sample, sigma_sample)
            plt.plot(x, y, 'r-', linewidth=2, 
                    label=f'Theoretical N({mu_sample:.2f}, {sigma_sample:.3f}²)')
            
            plt.xlabel('Sample Mean')
            plt.ylabel('Density')
            plt.title(f'Sample Size n = {n}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Theoretical mean of exponential(1): {mu_exp}")
        print(f"Theoretical std of exponential(1): {sigma_exp}")
        print("Note: Exponential distribution converges more slowly due to skewness!")


class MonteCarloSimulation:
    """
    A class for Monte Carlo simulation examples.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def estimate_pi(self, n_points: int = 100000) -> float:
        """
        Estimate π using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_points : int
            Number of random points to generate
            
        Returns:
        --------
        float
            Estimated value of π
        """
        print(f"\n=== Monte Carlo Estimation of π ===")
        print(f"Using {n_points:,} random points...")
        
        # Generate random points in [-1, 1] × [-1, 1]
        x = self.rng.uniform(-1, 1, n_points)
        y = self.rng.uniform(-1, 1, n_points)
        
        # Count points inside unit circle
        inside_circle = np.sum(x**2 + y**2 <= 1)
        
        # Estimate π
        pi_estimate = 4 * inside_circle / n_points
        
        print(f"Points inside circle: {inside_circle:,}")
        print(f"Total points: {n_points:,}")
        print(f"Estimated π: {pi_estimate:.6f}")
        print(f"Actual π: {np.pi:.6f}")
        print(f"Absolute error: {abs(pi_estimate - np.pi):.6f}")
        print(f"Relative error: {abs(pi_estimate - np.pi)/np.pi:.2%}")
        
        # Visualize
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, alpha=0.1, s=1)
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(circle)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.title('Monte Carlo π Estimation')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(1, 2, 2)
        # Show convergence
        cumulative_inside = np.cumsum(x**2 + y**2 <= 1)
        cumulative_pi = 4 * cumulative_inside / np.arange(1, n_points + 1)
        plt.plot(range(1, n_points + 1), cumulative_pi, alpha=0.7)
        plt.axhline(y=np.pi, color='red', linestyle='--', label='True π')
        plt.xlabel('Number of Points')
        plt.ylabel('Estimated π')
        plt.title('Convergence of π Estimate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return pi_estimate
    
    def integration_example(self, n_points: int = 10000) -> float:
        """
        Estimate integral using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_points : int
            Number of random points to generate
            
        Returns:
        --------
        float
            Estimated integral value
        """
        print(f"\n=== Monte Carlo Integration ===")
        print("Estimating ∫₀¹ x² dx = 1/3")
        
        # Generate random points in [0, 1]
        x = self.rng.uniform(0, 1, n_points)
        
        # Function to integrate: f(x) = x²
        y = x**2
        
        # Monte Carlo estimate
        integral_estimate = np.mean(y)
        true_value = 1/3
        
        print(f"Monte Carlo estimate: {integral_estimate:.6f}")
        print(f"True value: {true_value:.6f}")
        print(f"Absolute error: {abs(integral_estimate - true_value):.6f}")
        print(f"Relative error: {abs(integral_estimate - true_value)/true_value:.2%}")
        
        # Visualize
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        x_plot = np.linspace(0, 1, 1000)
        y_plot = x_plot**2
        plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x²')
        plt.fill_between(x_plot, y_plot, alpha=0.3, label='Area to estimate')
        plt.scatter(x, y, alpha=0.1, s=1, color='red', label='Random points')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Monte Carlo Integration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cumulative_estimate = np.cumsum(y) / np.arange(1, n_points + 1)
        plt.plot(range(1, n_points + 1), cumulative_estimate, alpha=0.7)
        plt.axhline(y=true_value, color='red', linestyle='--', label='True value')
        plt.xlabel('Number of Points')
        plt.ylabel('Estimated Integral')
        plt.title('Convergence of Integral Estimate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return integral_estimate


class BayesianInference:
    """
    A class for demonstrating Bayesian inference concepts.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def coin_tossing_bayesian(self, n_tosses: int = 10, n_heads: int = 7,
                             prior_alpha: float = 2, prior_beta: float = 2) -> Dict[str, Any]:
        """
        Demonstrate Bayesian inference for coin tossing.
        
        Parameters:
        -----------
        n_tosses : int
            Number of coin tosses
        n_heads : int
            Number of heads observed
        prior_alpha : float
            Prior Beta distribution parameter α
        prior_beta : float
            Prior Beta distribution parameter β
            
        Returns:
        --------
        dict
            Prior and posterior parameters
        """
        print(f"\n=== Bayesian Coin Tossing ===")
        print(f"Prior: Beta({prior_alpha}, {prior_beta})")
        print(f"Data: {n_heads} heads in {n_tosses} tosses")
        
        # Prior parameters
        prior_mean = prior_alpha / (prior_alpha + prior_beta)
        prior_var = (prior_alpha * prior_beta) / ((prior_alpha + prior_beta)**2 * 
                                                   (prior_alpha + prior_beta + 1))
        
        # Posterior parameters (conjugate prior)
        posterior_alpha = prior_alpha + n_heads
        posterior_beta = prior_beta + (n_tosses - n_heads)
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_var = (posterior_alpha * posterior_beta) / ((posterior_alpha + posterior_beta)**2 * 
                                                             (posterior_alpha + posterior_beta + 1))
        
        print(f"Prior mean: {prior_mean:.4f}")
        print(f"Prior std: {np.sqrt(prior_var):.4f}")
        print(f"Posterior mean: {posterior_mean:.4f}")
        print(f"Posterior std: {np.sqrt(posterior_var):.4f}")
        
        # Plot prior and posterior
        x = np.linspace(0, 1, 1000)
        prior_pdf = stats.beta.pdf(x, prior_alpha, prior_beta)
        posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, prior_pdf, 'b-', linewidth=2, label=f'Prior: Beta({prior_alpha}, {prior_beta})')
        plt.axvline(x=prior_mean, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Prior mean = {prior_mean:.3f}')
        plt.xlabel('Probability of Heads (p)')
        plt.ylabel('Density')
        plt.title('Prior Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(x, posterior_pdf, 'r-', linewidth=2, 
                label=f'Posterior: Beta({posterior_alpha}, {posterior_beta})')
        plt.axvline(x=posterior_mean, color='red', linestyle='--', alpha=0.7,
                   label=f'Posterior mean = {posterior_mean:.3f}')
        plt.axvline(x=n_heads/n_tosses, color='green', linestyle='--', alpha=0.7,
                   label=f'MLE = {n_heads/n_tosses:.3f}')
        plt.xlabel('Probability of Heads (p)')
        plt.ylabel('Density')
        plt.title('Posterior Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'prior_alpha': prior_alpha,
            'prior_beta': prior_beta,
            'prior_mean': prior_mean,
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'posterior_mean': posterior_mean
        }


def main():
    """
    Main function to demonstrate all probability concepts.
    """
    print("Probability Fundamentals - Python Implementation")
    print("=" * 50)
    
    # Initialize classes
    prob_calc = ProbabilityCalculator()
    rv_sim = RandomVariableSimulator()
    clt_demo = CentralLimitTheorem()
    mc_sim = MonteCarloSimulation()
    bayes_demo = BayesianInference()
    
    # 1. Basic probability concepts
    print("\n" + "="*50)
    print("1. BASIC PROBABILITY CONCEPTS")
    print("="*50)
    
    # Coin toss experiment
    coin_results = prob_calc.coin_toss_experiment(1000)
    
    # Card deck probabilities
    card_probs = prob_calc.card_deck_probabilities()
    
    # Medical diagnosis example
    medical_probs = prob_calc.medical_diagnosis_example()
    
    # 2. Random variables
    print("\n" + "="*50)
    print("2. RANDOM VARIABLES")
    print("="*50)
    
    # Bernoulli simulation
    bernoulli_results = rv_sim.bernoulli_simulation(p=0.3, n=1000)
    
    # Binomial simulation
    binomial_results = rv_sim.binomial_simulation(n=20, p=0.3, n_experiments=1000)
    
    # Normal distribution demo
    normal_results = rv_sim.normal_distribution_demo(mu=0, sigma=1, n_samples=10000)
    
    # 3. Central Limit Theorem
    print("\n" + "="*50)
    print("3. CENTRAL LIMIT THEOREM")
    print("="*50)
    
    # CLT with uniform distribution
    clt_demo.clt_uniform_demo(n_samples=1000, sample_sizes=[1, 5, 10, 30])
    
    # CLT with exponential distribution
    clt_demo.clt_exponential_demo(n_samples=1000, sample_sizes=[1, 5, 10, 30])
    
    # 4. Monte Carlo simulation
    print("\n" + "="*50)
    print("4. MONTE CARLO SIMULATION")
    print("="*50)
    
    # Estimate π
    pi_estimate = mc_sim.estimate_pi(n_points=100000)
    
    # Integration example
    integral_estimate = mc_sim.integration_example(n_points=10000)
    
    # 5. Bayesian inference
    print("\n" + "="*50)
    print("5. BAYESIAN INFERENCE")
    print("="*50)
    
    # Bayesian coin tossing
    bayes_results = bayes_demo.coin_tossing_bayesian(n_tosses=10, n_heads=7)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This implementation demonstrates:")
    print("✓ Basic probability calculations and properties")
    print("✓ Random variable simulation and analysis")
    print("✓ Probability distributions (Bernoulli, Binomial, Normal)")
    print("✓ Central Limit Theorem with different distributions")
    print("✓ Monte Carlo simulation for estimation and integration")
    print("✓ Bayesian inference with conjugate priors")
    print("\nAll concepts are implemented with detailed explanations,")
    print("visualizations, and practical examples!")


if __name__ == "__main__":
    main() 