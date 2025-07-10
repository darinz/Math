"""
Statistical Inference
====================

This module implements core statistical inference concepts including:
- Hypothesis testing (one-sample, two-sample, paired)
- Confidence intervals
- P-values and significance testing
- Multiple testing corrections
- Power analysis
- Effect size calculations
- Practical applications (A/B testing, medical research, quality control)

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, chi2, f
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HypothesisTesting:
    """
    A comprehensive class for hypothesis testing with detailed implementations.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)  # For reproducible results
    
    def one_sample_ttest(self, data: np.ndarray, mu0: float, 
                         alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform one-sample t-test with comprehensive output.
        
        Parameters:
        -----------
        data : np.ndarray
            Sample data
        mu0 : float
            Hypothesized population mean
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis ('two-sided', 'greater', 'less')
            
        Returns:
        --------
        dict
            Complete test results including statistics, p-value, effect size, and decision
        """
        print(f"\n=== One-Sample t-Test ===")
        print(f"H₀: μ = {mu0}")
        print(f"H₁: μ ≠ {mu0}" if alternative == 'two-sided' else f"H₁: μ {alternative} {mu0}")
        print(f"α = {alpha}")
        
        # Calculate sample statistics
        n = len(data)
        x_bar = np.mean(data)
        s = np.std(data, ddof=1)  # Sample standard deviation
        se = s / np.sqrt(n)  # Standard error
        
        # Calculate test statistic
        t_stat = (x_bar - mu0) / se
        df = n - 1  # Degrees of freedom
        
        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == 'greater':
            p_value = 1 - t.cdf(t_stat, df)
        elif alternative == 'less':
            p_value = t.cdf(t_stat, df)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        # Calculate effect size (Cohen's d)
        effect_size = (x_bar - mu0) / s
        
        # Calculate confidence interval
        if alternative == 'two-sided':
            t_critical = t.ppf(1 - alpha/2, df)
            ci_lower = x_bar - t_critical * se
            ci_upper = x_bar + t_critical * se
        elif alternative == 'greater':
            t_critical = t.ppf(1 - alpha, df)
            ci_lower = x_bar - t_critical * se
            ci_upper = np.inf
        else:  # less
            t_critical = t.ppf(1 - alpha, df)
            ci_lower = -np.inf
            ci_upper = x_bar + t_critical * se
        
        # Decision
        decision = "Reject H₀" if p_value <= alpha else "Fail to reject H₀"
        
        # Print results
        print(f"\nSample Statistics:")
        print(f"n = {n}")
        print(f"x̄ = {x_bar:.4f}")
        print(f"s = {s:.4f}")
        print(f"SE = {se:.4f}")
        
        print(f"\nTest Results:")
        print(f"t-statistic = {t_stat:.4f}")
        print(f"df = {df}")
        print(f"p-value = {p_value:.6f}")
        print(f"Effect size (Cohen's d) = {effect_size:.4f}")
        print(f"Decision: {decision}")
        
        if alternative == 'two-sided':
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"Effect size interpretation: {effect_interpretation}")
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'decision': decision,
            'sample_mean': x_bar,
            'sample_std': s,
            'sample_size': n,
            'degrees_of_freedom': df,
            'effect_interpretation': effect_interpretation
        }
    
    def two_sample_ttest(self, group1: np.ndarray, group2: np.ndarray,
                         alpha: float = 0.05, alternative: str = 'two-sided',
                         equal_var: bool = False) -> Dict[str, Any]:
        """
        Perform two-sample t-test (independent samples).
        
        Parameters:
        -----------
        group1, group2 : np.ndarray
            Sample data for two groups
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
        equal_var : bool
            Whether to assume equal variances (pooled t-test)
            
        Returns:
        --------
        dict
            Complete test results
        """
        print(f"\n=== Two-Sample t-Test ===")
        print(f"H₀: μ₁ = μ₂")
        print(f"H₁: μ₁ ≠ μ₂" if alternative == 'two-sided' else f"H₁: μ₁ {alternative} μ₂")
        print(f"Equal variances: {equal_var}")
        print(f"α = {alpha}")
        
        # Calculate sample statistics
        n1, n2 = len(group1), len(group2)
        x1_bar, x2_bar = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Calculate test statistic
        if equal_var:
            # Pooled t-test
            pooled_var = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
        
        t_stat = (x1_bar - x2_bar) / se
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == 'greater':
            p_value = 1 - t.cdf(t_stat, df)
        elif alternative == 'less':
            p_value = t.cdf(t_stat, df)
        
        # Calculate effect size (Cohen's d)
        if equal_var:
            pooled_std = np.sqrt(pooled_var)
        else:
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
        
        effect_size = (x1_bar - x2_bar) / pooled_std
        
        # Calculate confidence interval
        if alternative == 'two-sided':
            t_critical = t.ppf(1 - alpha/2, df)
            ci_lower = (x1_bar - x2_bar) - t_critical * se
            ci_upper = (x1_bar - x2_bar) + t_critical * se
        else:
            t_critical = t.ppf(1 - alpha, df)
            if alternative == 'greater':
                ci_lower = (x1_bar - x2_bar) - t_critical * se
                ci_upper = np.inf
            else:
                ci_lower = -np.inf
                ci_upper = (x1_bar - x2_bar) + t_critical * se
        
        # Decision
        decision = "Reject H₀" if p_value <= alpha else "Fail to reject H₀"
        
        # Print results
        print(f"\nGroup Statistics:")
        print(f"Group 1: n₁ = {n1}, x̄₁ = {x1_bar:.4f}, s₁ = {s1:.4f}")
        print(f"Group 2: n₂ = {n2}, x̄₂ = {x2_bar:.4f}, s₂ = {s2:.4f}")
        
        print(f"\nTest Results:")
        print(f"t-statistic = {t_stat:.4f}")
        print(f"df = {df:.2f}")
        print(f"p-value = {p_value:.6f}")
        print(f"Effect size (Cohen's d) = {effect_size:.4f}")
        print(f"Decision: {decision}")
        
        if alternative == 'two-sided':
            print(f"95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"Effect size interpretation: {effect_interpretation}")
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'decision': decision,
            'group1_mean': x1_bar,
            'group2_mean': x2_bar,
            'group1_std': s1,
            'group2_std': s2,
            'group1_size': n1,
            'group2_size': n2,
            'degrees_of_freedom': df,
            'effect_interpretation': effect_interpretation
        }
    
    def paired_ttest(self, before: np.ndarray, after: np.ndarray,
                     alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform paired t-test for before/after or matched pairs data.
        
        Parameters:
        -----------
        before, after : np.ndarray
            Paired observations
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
            
        Returns:
        --------
        dict
            Complete test results
        """
        print(f"\n=== Paired t-Test ===")
        print(f"H₀: μ_d = 0")
        print(f"H₁: μ_d ≠ 0" if alternative == 'two-sided' else f"H₁: μ_d {alternative} 0")
        print(f"α = {alpha}")
        
        # Calculate differences
        differences = after - before
        n = len(differences)
        d_bar = np.mean(differences)
        s_d = np.std(differences, ddof=1)
        se_d = s_d / np.sqrt(n)
        
        # Calculate test statistic
        t_stat = d_bar / se_d
        df = n - 1
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == 'greater':
            p_value = 1 - t.cdf(t_stat, df)
        elif alternative == 'less':
            p_value = t.cdf(t_stat, df)
        
        # Calculate effect size
        effect_size = d_bar / s_d
        
        # Calculate confidence interval
        if alternative == 'two-sided':
            t_critical = t.ppf(1 - alpha/2, df)
            ci_lower = d_bar - t_critical * se_d
            ci_upper = d_bar + t_critical * se_d
        else:
            t_critical = t.ppf(1 - alpha, df)
            if alternative == 'greater':
                ci_lower = d_bar - t_critical * se_d
                ci_upper = np.inf
            else:
                ci_lower = -np.inf
                ci_upper = d_bar + t_critical * se_d
        
        # Decision
        decision = "Reject H₀" if p_value <= alpha else "Fail to reject H₀"
        
        # Print results
        print(f"\nDifference Statistics:")
        print(f"n = {n}")
        print(f"d̄ = {d_bar:.4f}")
        print(f"s_d = {s_d:.4f}")
        print(f"SE_d = {se_d:.4f}")
        
        print(f"\nTest Results:")
        print(f"t-statistic = {t_stat:.4f}")
        print(f"df = {df}")
        print(f"p-value = {p_value:.6f}")
        print(f"Effect size = {effect_size:.4f}")
        print(f"Decision: {decision}")
        
        if alternative == 'two-sided':
            print(f"95% CI for mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'decision': decision,
            'mean_difference': d_bar,
            'std_difference': s_d,
            'sample_size': n,
            'degrees_of_freedom': df
        }
    
    def z_test_proportion(self, x: int, n: int, p0: float,
                          alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform z-test for population proportion.
        
        Parameters:
        -----------
        x : int
            Number of successes
        n : int
            Sample size
        p0 : float
            Hypothesized population proportion
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
            
        Returns:
        --------
        dict
            Complete test results
        """
        print(f"\n=== Z-Test for Proportion ===")
        print(f"H₀: p = {p0}")
        print(f"H₁: p ≠ {p0}" if alternative == 'two-sided' else f"H₁: p {alternative} {p0}")
        print(f"α = {alpha}")
        
        # Calculate sample proportion
        p_hat = x / n
        
        # Calculate test statistic
        se = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (p_hat - p0) / se
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - norm.cdf(z_stat)
        elif alternative == 'less':
            p_value = norm.cdf(z_stat)
        
        # Calculate confidence interval
        se_ci = np.sqrt(p_hat * (1 - p_hat) / n)
        if alternative == 'two-sided':
            z_critical = norm.ppf(1 - alpha/2)
            ci_lower = p_hat - z_critical * se_ci
            ci_upper = p_hat + z_critical * se_ci
        else:
            z_critical = norm.ppf(1 - alpha)
            if alternative == 'greater':
                ci_lower = p_hat - z_critical * se_ci
                ci_upper = 1.0
            else:
                ci_lower = 0.0
                ci_upper = p_hat + z_critical * se_ci
        
        # Decision
        decision = "Reject H₀" if p_value <= alpha else "Fail to reject H₀"
        
        # Print results
        print(f"\nSample Statistics:")
        print(f"n = {n}")
        print(f"x = {x}")
        print(f"p̂ = {p_hat:.4f}")
        
        print(f"\nTest Results:")
        print(f"z-statistic = {z_stat:.4f}")
        print(f"p-value = {p_value:.6f}")
        print(f"Decision: {decision}")
        
        if alternative == 'two-sided':
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'test_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'decision': decision,
            'sample_proportion': p_hat,
            'sample_size': n,
            'successes': x
        }


class ConfidenceIntervals:
    """
    A class for calculating confidence intervals for various parameters.
    """
    
    def __init__(self):
        pass
    
    def mean_ci(self, data: np.ndarray, confidence: float = 0.95,
                sigma_known: bool = False, sigma: float = None) -> Tuple[float, float]:
        """
        Calculate confidence interval for population mean.
        
        Parameters:
        -----------
        data : np.ndarray
            Sample data
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
        sigma_known : bool
            Whether population standard deviation is known
        sigma : float
            Population standard deviation (if known)
            
        Returns:
        --------
        tuple
            (lower_bound, upper_bound)
        """
        n = len(data)
        x_bar = np.mean(data)
        alpha = 1 - confidence
        
        if sigma_known and sigma is not None:
            # Z-interval
            se = sigma / np.sqrt(n)
            z_critical = norm.ppf(1 - alpha/2)
            margin_of_error = z_critical * se
        else:
            # t-interval
            s = np.std(data, ddof=1)
            se = s / np.sqrt(n)
            df = n - 1
            t_critical = t.ppf(1 - alpha/2, df)
            margin_of_error = t_critical * se
        
        ci_lower = x_bar - margin_of_error
        ci_upper = x_bar + margin_of_error
        
        print(f"\n=== Confidence Interval for Mean ===")
        print(f"Confidence level: {confidence*100:.0f}%")
        print(f"Sample mean: {x_bar:.4f}")
        print(f"Standard error: {se:.4f}")
        print(f"Margin of error: {margin_of_error:.4f}")
        print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return ci_lower, ci_upper
    
    def proportion_ci(self, x: int, n: int, confidence: float = 0.95,
                      method: str = 'normal') -> Tuple[float, float]:
        """
        Calculate confidence interval for population proportion.
        
        Parameters:
        -----------
        x : int
            Number of successes
        n : int
            Sample size
        confidence : float
            Confidence level
        method : str
            Method for CI calculation ('normal', 'wilson', 'agresti-coull')
            
        Returns:
        --------
        tuple
            (lower_bound, upper_bound)
        """
        p_hat = x / n
        alpha = 1 - confidence
        z_critical = norm.ppf(1 - alpha/2)
        
        if method == 'normal':
            # Standard normal approximation
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            margin_of_error = z_critical * se
            ci_lower = p_hat - margin_of_error
            ci_upper = p_hat + margin_of_error
        elif method == 'wilson':
            # Wilson score interval
            denominator = 1 + z_critical**2/n
            centre_adjustment = z_critical * np.sqrt(z_critical**2 - 1/n + 4*p_hat*(1-p_hat)*(2/n-1) + (4*p_hat-2)) / (2*n)
            centre_adjustment = centre_adjustment / denominator
            adjusted_centre = (p_hat + z_critical**2/(2*n)) / denominator
            adjusted_error = z_critical * np.sqrt(p_hat*(1-p_hat)/n + z_critical**2/(4*n**2)) / denominator
            ci_lower = adjusted_centre - adjusted_error
            ci_upper = adjusted_centre + adjusted_error
        else:
            raise ValueError("method must be 'normal' or 'wilson'")
        
        print(f"\n=== Confidence Interval for Proportion ===")
        print(f"Confidence level: {confidence*100:.0f}%")
        print(f"Sample proportion: {p_hat:.4f}")
        print(f"Sample size: {n}")
        print(f"Successes: {x}")
        print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return ci_lower, ci_upper
    
    def difference_means_ci(self, group1: np.ndarray, group2: np.ndarray,
                           confidence: float = 0.95, equal_var: bool = False) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference in means.
        
        Parameters:
        -----------
        group1, group2 : np.ndarray
            Sample data for two groups
        confidence : float
            Confidence level
        equal_var : bool
            Whether to assume equal variances
            
        Returns:
        --------
        tuple
            (lower_bound, upper_bound)
        """
        n1, n2 = len(group1), len(group2)
        x1_bar, x2_bar = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        alpha = 1 - confidence
        
        if equal_var:
            # Pooled standard error
            pooled_var = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's standard error
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
        
        t_critical = t.ppf(1 - alpha/2, df)
        margin_of_error = t_critical * se
        
        diff = x1_bar - x2_bar
        ci_lower = diff - margin_of_error
        ci_upper = diff + margin_of_error
        
        print(f"\n=== Confidence Interval for Difference in Means ===")
        print(f"Confidence level: {confidence*100:.0f}%")
        print(f"Sample difference: {diff:.4f}")
        print(f"Standard error: {se:.4f}")
        print(f"Margin of error: {margin_of_error:.4f}")
        print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return ci_lower, ci_upper


class PowerAnalysis:
    """
    A class for power analysis and sample size determination.
    """
    
    def __init__(self):
        pass
    
    def power_ttest(self, effect_size: float, n: int, alpha: float = 0.05,
                    alternative: str = 'two-sided') -> float:
        """
        Calculate power for t-test.
        
        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        n : int
            Sample size per group
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
            
        Returns:
        --------
        float
            Power (1 - β)
        """
        if alternative == 'two-sided':
            critical_t = t.ppf(1 - alpha/2, 2*n - 2)
        else:
            critical_t = t.ppf(1 - alpha, 2*n - 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n/2)
        
        if alternative == 'two-sided':
            power = 1 - t.cdf(critical_t, 2*n - 2, ncp) + t.cdf(-critical_t, 2*n - 2, ncp)
        else:
            power = 1 - t.cdf(critical_t, 2*n - 2, ncp)
        
        return power
    
    def sample_size_ttest(self, effect_size: float, power: float = 0.8,
                          alpha: float = 0.05, alternative: str = 'two-sided') -> int:
        """
        Calculate required sample size for t-test.
        
        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        power : float
            Desired power
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
            
        Returns:
        --------
        int
            Required sample size per group
        """
        if alternative == 'two-sided':
            z_alpha = norm.ppf(1 - alpha/2)
        else:
            z_alpha = norm.ppf(1 - alpha)
        
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size)**2
        
        return int(np.ceil(n))
    
    def power_curve(self, effect_sizes: np.ndarray, n: int, alpha: float = 0.05,
                   alternative: str = 'two-sided') -> np.ndarray:
        """
        Calculate power for range of effect sizes.
        
        Parameters:
        -----------
        effect_sizes : np.ndarray
            Range of effect sizes
        n : int
            Sample size per group
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
            
        Returns:
        --------
        np.ndarray
            Power values for each effect size
        """
        powers = np.array([self.power_ttest(d, n, alpha, alternative) for d in effect_sizes])
        return powers
    
    def plot_power_analysis(self, effect_sizes: np.ndarray, sample_sizes: List[int],
                           alpha: float = 0.05, alternative: str = 'two-sided'):
        """
        Create power analysis visualization.
        
        Parameters:
        -----------
        effect_sizes : np.ndarray
            Range of effect sizes
        sample_sizes : List[int]
            List of sample sizes to compare
        alpha : float
            Significance level
        alternative : str
            Alternative hypothesis
        """
        plt.figure(figsize=(12, 8))
        
        for n in sample_sizes:
            powers = self.power_curve(effect_sizes, n, alpha, alternative)
            plt.plot(effect_sizes, powers, marker='o', label=f'n = {n}')
        
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.ylabel('Power')
        plt.title('Power Analysis for Two-Sample t-Test')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class MultipleTesting:
    """
    A class for multiple testing corrections.
    """
    
    def __init__(self):
        pass
    
    def bonferroni_correction(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Apply Bonferroni correction.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Array of p-values
        alpha : float
            Family-wise significance level
            
        Returns:
        --------
        np.ndarray
            Boolean array indicating which hypotheses to reject
        """
        m = len(p_values)
        alpha_adjusted = alpha / m
        rejections = p_values <= alpha_adjusted
        
        print(f"\n=== Bonferroni Correction ===")
        print(f"Number of tests: {m}")
        print(f"Original α: {alpha}")
        print(f"Adjusted α: {alpha_adjusted:.6f}")
        print(f"Number of rejections: {np.sum(rejections)}")
        
        return rejections
    
    def benjamini_hochberg(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR control.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Array of p-values
        alpha : float
            False discovery rate
        
        Returns:
        --------
        np.ndarray
            Boolean array indicating which hypotheses to reject
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Find largest k such that p_{(k)} <= (k/m) * alpha
        k = 0
        for i in range(m):
            if sorted_p_values[i] <= (i + 1) / m * alpha:
                k = i + 1
        
        rejections = np.zeros(m, dtype=bool)
        if k > 0:
            rejections[sorted_indices[:k]] = True
        
        print(f"\n=== Benjamini-Hochberg FDR Control ===")
        print(f"Number of tests: {m}")
        print(f"FDR level: {alpha}")
        print(f"Number of rejections: {np.sum(rejections)}")
        print(f"Estimated FDR: {np.sum(rejections) / max(1, np.sum(rejections)):.4f}")
        
        return rejections
    
    def holm_correction(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Apply Holm's step-down procedure.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Array of p-values
        alpha : float
            Family-wise significance level
        
        Returns:
        --------
        np.ndarray
            Boolean array indicating which hypotheses to reject
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Find largest k such that p_{(i)} <= alpha / (m - i + 1) for all i <= k
        k = 0
        for i in range(m):
            if sorted_p_values[i] <= alpha / (m - i):
                k = i + 1
            else:
                break
        
        rejections = np.zeros(m, dtype=bool)
        if k > 0:
            rejections[sorted_indices[:k]] = True
        
        print(f"\n=== Holm's Step-Down Procedure ===")
        print(f"Number of tests: {m}")
        print(f"FWER level: {alpha}")
        print(f"Number of rejections: {np.sum(rejections)}")
        
        return rejections
    
    def compare_corrections(self, p_values: np.ndarray, alpha: float = 0.05):
        """
        Compare different multiple testing correction methods.
        
        Parameters:
        -----------
        p_values : np.ndarray
            Array of p-values
        alpha : float
            Significance level
        """
        print(f"\n=== Multiple Testing Correction Comparison ===")
        print(f"Original p-values: {p_values}")
        print(f"α = {alpha}")
        
        # Apply different corrections
        bonf_rejections = self.bonferroni_correction(p_values, alpha)
        bh_rejections = self.benjamini_hochberg(p_values, alpha)
        holm_rejections = self.holm_correction(p_values, alpha)
        
        # Create comparison table
        methods = ['Bonferroni', 'Benjamini-Hochberg', 'Holm']
        rejections = [np.sum(bonf_rejections), np.sum(bh_rejections), np.sum(holm_rejections)]
        
        print(f"\nComparison:")
        print(f"{'Method':<20} {'Rejections':<12} {'Power'}")
        print("-" * 40)
        for method, rej in zip(methods, rejections):
            power_rank = "High" if rej == max(rejections) else "Medium" if rej > 0 else "Low"
            print(f"{method:<20} {rej:<12} {power_rank}")


class PracticalApplications:
    """
    A class demonstrating practical applications of statistical inference.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng(42)
    
    def ab_testing_example(self):
        """
        Demonstrate A/B testing for website conversion rates.
        """
        print("\n=== A/B Testing Example: Website Conversion Rates ===")
        
        # Simulate A/B test data
        n1, n2 = 1000, 1000  # Visitors per design
        p1, p2 = 0.05, 0.065  # True conversion rates
        
        # Generate data
        conversions1 = self.rng.binomial(n1, p1)
        conversions2 = self.rng.binomial(n2, p2)
        
        p1_hat = conversions1 / n1
        p2_hat = conversions2 / n2
        
        print(f"Design A: {conversions1}/{n1} conversions ({p1_hat:.3f})")
        print(f"Design B: {conversions2}/{n2} conversions ({p2_hat:.3f})")
        
        # Hypothesis test
        pooled_p = (conversions1 + conversions2) / (n1 + n2)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        z_stat = (p2_hat - p1_hat) / se
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        
        print(f"\nHypothesis Test:")
        print(f"H₀: p₁ = p₂")
        print(f"H₁: p₁ ≠ p₂")
        print(f"z-statistic: {z_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
        
        # Confidence interval
        se_ci = np.sqrt(p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2)
        z_critical = norm.ppf(0.975)
        margin_of_error = z_critical * se_ci
        ci_lower = (p2_hat - p1_hat) - margin_of_error
        ci_upper = (p2_hat - p1_hat) + margin_of_error
        
        print(f"\n95% Confidence Interval for difference:")
        print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    def medical_research_example(self):
        """
        Demonstrate medical research example with drug effectiveness.
        """
        print("\n=== Medical Research Example: Drug Effectiveness ===")
        
        # Simulate medical trial data
        n1, n2 = 50, 50  # Patients per group
        mu1, mu2 = 140, 130  # True blood pressure means
        sigma1, sigma2 = 15, 12  # Standard deviations
        
        # Generate data
        placebo = self.rng.normal(mu1, sigma1, n1)
        drug = self.rng.normal(mu2, sigma2, n2)
        
        print(f"Placebo group: n = {n1}, x̄ = {np.mean(placebo):.1f}, s = {np.std(placebo, ddof=1):.1f}")
        print(f"Drug group: n = {n2}, x̄ = {np.mean(drug):.1f}, s = {np.std(drug, ddof=1):.1f}")
        
        # Perform t-test
        ht = HypothesisTesting()
        results = ht.two_sample_ttest(placebo, drug, alternative='greater')
        
        # Calculate effect size
        pooled_std = np.sqrt(((n1-1)*np.var(placebo, ddof=1) + (n2-1)*np.var(drug, ddof=1)) / (n1+n2-2))
        effect_size = (np.mean(placebo) - np.mean(drug)) / pooled_std
        
        print(f"\nEffect Size (Cohen's d): {effect_size:.3f}")
        
        return results
    
    def quality_control_example(self):
        """
        Demonstrate quality control example.
        """
        print("\n=== Quality Control Example: Manufacturing Process ===")
        
        # Simulate quality control data
        target = 100  # Target weight in grams
        true_mean = 102  # Actual process mean
        sigma = 5  # Process standard deviation
        n = 25  # Sample size
        
        # Generate sample
        sample = self.rng.normal(true_mean, sigma, n)
        
        print(f"Target weight: {target} g")
        print(f"Sample: n = {n}, x̄ = {np.mean(sample):.2f} g, s = {np.std(sample, ddof=1):.2f} g")
        
        # Perform t-test
        ht = HypothesisTesting()
        results = ht.one_sample_ttest(sample, target, alternative='two-sided')
        
        # Calculate confidence interval
        ci = ConfidenceIntervals()
        ci_lower, ci_upper = ci.mean_ci(sample, confidence=0.95)
        
        print(f"\nProcess Control:")
        if ci_lower <= target <= ci_upper:
            print("✓ Process mean is within acceptable range")
        else:
            print("✗ Process mean is outside acceptable range")
        
        return results


def main():
    """
    Main function to demonstrate all statistical inference concepts.
    """
    print("Statistical Inference - Python Implementation")
    print("=" * 50)
    
    # Initialize classes
    ht = HypothesisTesting()
    ci = ConfidenceIntervals()
    pa = PowerAnalysis()
    mt = MultipleTesting()
    pa_app = PracticalApplications()
    
    # 1. One-sample t-test example
    print("\n" + "="*50)
    print("1. ONE-SAMPLE T-TEST")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.normal(105, 10, 20)  # Mean=105, SD=10, n=20
    
    # Test against hypothesized mean of 100
    results = ht.one_sample_ttest(sample_data, mu0=100, alpha=0.05, alternative='two-sided')
    
    # 2. Two-sample t-test example
    print("\n" + "="*50)
    print("2. TWO-SAMPLE T-TEST")
    print("="*50)
    
    # Generate two groups
    group1 = np.random.normal(85, 12, 30)
    group2 = np.random.normal(78, 15, 25)
    
    results = ht.two_sample_ttest(group1, group2, alpha=0.05, alternative='two-sided')
    
    # 3. Paired t-test example
    print("\n" + "="*50)
    print("3. PAIRED T-TEST")
    print("="*50)
    
    # Generate before/after data
    before = np.array([180, 165, 200, 175, 190, 185, 170, 195, 180, 175])
    after = before - np.random.normal(5, 2, 10)  # Average 5 unit reduction
    
    results = ht.paired_ttest(before, after, alpha=0.05, alternative='less')
    
    # 4. Confidence intervals
    print("\n" + "="*50)
    print("4. CONFIDENCE INTERVALS")
    print("="*50)
    
    # Mean CI
    ci_lower, ci_upper = ci.mean_ci(sample_data, confidence=0.95)
    
    # Proportion CI
    x, n = 65, 100  # 65 successes out of 100 trials
    ci_lower, ci_upper = ci.proportion_ci(x, n, confidence=0.95)
    
    # Difference in means CI
    ci_lower, ci_upper = ci.difference_means_ci(group1, group2, confidence=0.95)
    
    # 5. Power analysis
    print("\n" + "="*50)
    print("5. POWER ANALYSIS")
    print("="*50)
    
    effect_sizes = np.array([0.2, 0.5, 0.8, 1.0, 1.5])
    sample_sizes = [20, 50, 100]
    
    pa.plot_power_analysis(effect_sizes, sample_sizes)
    
    # Calculate required sample size for medium effect
    required_n = pa.sample_size_ttest(effect_size=0.5, power=0.8, alpha=0.05)
    print(f"\nRequired sample size for medium effect (d=0.5) with 80% power: {required_n} per group")
    
    # 6. Multiple testing
    print("\n" + "="*50)
    print("6. MULTIPLE TESTING")
    print("="*50)
    
    # Simulate multiple p-values
    p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    
    mt.compare_corrections(p_values, alpha=0.05)
    
    # 7. Practical applications
    print("\n" + "="*50)
    print("7. PRACTICAL APPLICATIONS")
    print("="*50)
    
    # A/B testing
    ab_results = pa_app.ab_testing_example()
    
    # Medical research
    medical_results = pa_app.medical_research_example()
    
    # Quality control
    qc_results = pa_app.quality_control_example()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This implementation demonstrates:")
    print("✓ Hypothesis testing (one-sample, two-sample, paired)")
    print("✓ Confidence interval construction")
    print("✓ Power analysis and sample size determination")
    print("✓ Multiple testing corrections")
    print("✓ Practical applications (A/B testing, medical research, quality control)")
    print("✓ Effect size calculations and interpretation")
    print("\nAll concepts are implemented with detailed explanations,")
    print("visualizations, and practical examples!")


if __name__ == "__main__":
    main() 