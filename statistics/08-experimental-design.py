"""
Experimental Design Implementation

This module provides comprehensive implementations of experimental design concepts,
including randomized controlled trials, factorial designs, blocking, power analysis,
A/B testing, and practical applications.

Key Concepts Covered:
- Randomized Controlled Trials: Gold standard for causal inference
- Factorial Designs: Testing multiple factors and interactions
- Blocking and Randomization: Controlling for known sources of variation
- Sample Size Determination: Power analysis for different designs
- A/B Testing: Digital experimentation frameworks
- Practical Applications: Clinical trials, agricultural experiments, industrial optimization

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, f, chi2
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RandomizedControlledTrial:
    """
    Randomized Controlled Trial Implementation
    
    Implements various types of RCTs including simple, stratified, and cluster randomization.
    """
    
    def __init__(self):
        self.treatment_assignments = None
        self.outcomes = None
        self.strata = None
        
    def simple_randomization(self, n_treatment, n_control, seed=None):
        """
        Simple randomization for RCT.
        
        Parameters:
        -----------
        n_treatment : int
            Number of treatment subjects
        n_control : int
            Number of control subjects
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        assignments : array
            Treatment assignments (1 for treatment, 0 for control)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_total = n_treatment + n_control
        assignments = np.zeros(n_total)
        
        # Randomly assign treatment
        treatment_indices = np.random.choice(n_total, n_treatment, replace=False)
        assignments[treatment_indices] = 1
        
        self.treatment_assignments = assignments
        return assignments
    
    def stratified_randomization(self, strata, n_treatment_per_stratum, n_control_per_stratum, seed=None):
        """
        Stratified randomization for RCT.
        
        Parameters:
        -----------
        strata : array-like
            Stratum assignments for each subject
        n_treatment_per_stratum : dict
            Number of treatment subjects per stratum
        n_control_per_stratum : dict
            Number of control subjects per stratum
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        assignments : array
            Treatment assignments
        """
        if seed is not None:
            np.random.seed(seed)
        
        unique_strata = np.unique(strata)
        assignments = np.zeros(len(strata))
        
        for stratum in unique_strata:
            stratum_mask = strata == stratum
            stratum_indices = np.where(stratum_mask)[0]
            
            n_treatment = n_treatment_per_stratum.get(stratum, 0)
            n_control = n_control_per_stratum.get(stratum, 0)
            
            # Randomize within stratum
            stratum_assignments = np.zeros(len(stratum_indices))
            treatment_indices = np.random.choice(len(stratum_indices), n_treatment, replace=False)
            stratum_assignments[treatment_indices] = 1
            
            assignments[stratum_indices] = stratum_assignments
        
        self.treatment_assignments = assignments
        self.strata = strata
        return assignments
    
    def cluster_randomization(self, cluster_ids, n_treatment_clusters, n_control_clusters, seed=None):
        """
        Cluster randomization for RCT.
        
        Parameters:
        -----------
        cluster_ids : array-like
            Cluster assignments for each subject
        n_treatment_clusters : int
            Number of treatment clusters
        n_control_clusters : int
            Number of control clusters
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        assignments : array
            Treatment assignments
        """
        if seed is not None:
            np.random.seed(seed)
        
        unique_clusters = np.unique(cluster_ids)
        n_total_clusters = len(unique_clusters)
        
        # Randomly assign clusters to treatment
        treatment_clusters = np.random.choice(unique_clusters, n_treatment_clusters, replace=False)
        
        assignments = np.zeros(len(cluster_ids))
        for cluster in treatment_clusters:
            cluster_mask = cluster_ids == cluster
            assignments[cluster_mask] = 1
        
        self.treatment_assignments = assignments
        return assignments
    
    def analyze_rct(self, outcomes, alpha=0.05):
        """
        Analyze RCT results using difference-in-means estimator.
        
        Parameters:
        -----------
        outcomes : array-like
            Outcome measurements
        alpha : float
            Significance level
            
        Returns:
        --------
        results : dict
            Analysis results including ATE, standard error, confidence interval
        """
        outcomes = np.array(outcomes)
        treatment_mask = self.treatment_assignments == 1
        control_mask = self.treatment_assignments == 0
        
        # Calculate means and variances
        treatment_mean = np.mean(outcomes[treatment_mask])
        control_mean = np.mean(outcomes[control_mask])
        treatment_var = np.var(outcomes[treatment_mask], ddof=1)
        control_var = np.var(outcomes[control_mask], ddof=1)
        
        # Sample sizes
        n_treatment = np.sum(treatment_mask)
        n_control = np.sum(control_mask)
        
        # ATE estimate
        ate = treatment_mean - control_mean
        
        # Standard error
        se = np.sqrt(treatment_var/n_treatment + control_var/n_control)
        
        # Degrees of freedom for t-test
        df = n_treatment + n_control - 2
        
        # Confidence interval
        t_critical = t.ppf(1 - alpha/2, df)
        ci_lower = ate - t_critical * se
        ci_upper = ate + t_critical * se
        
        # t-statistic and p-value
        t_stat = ate / se
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        
        results = {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_stat': t_stat,
            'p_value': p_value,
            'treatment_mean': treatment_mean,
            'control_mean': control_mean,
            'n_treatment': n_treatment,
            'n_control': n_control
        }
        
        return results
    
    def stratified_analysis(self, outcomes):
        """
        Analyze stratified RCT results.
        
        Parameters:
        -----------
        outcomes : array-like
            Outcome measurements
            
        Returns:
        --------
        results : dict
            Stratified analysis results
        """
        if self.strata is None:
            raise ValueError("No strata information available")
        
        unique_strata = np.unique(self.strata)
        stratum_effects = []
        stratum_weights = []
        
        for stratum in unique_strata:
            stratum_mask = self.strata == stratum
            stratum_outcomes = outcomes[stratum_mask]
            stratum_assignments = self.treatment_assignments[stratum_mask]
            
            # Analyze within stratum
            treatment_mask = stratum_assignments == 1
            control_mask = stratum_assignments == 0
            
            if np.sum(treatment_mask) > 0 and np.sum(control_mask) > 0:
                treatment_mean = np.mean(stratum_outcomes[treatment_mask])
                control_mean = np.mean(stratum_outcomes[control_mask])
                stratum_effect = treatment_mean - control_mean
                
                stratum_effects.append(stratum_effect)
                stratum_weights.append(np.sum(stratum_mask))
        
        # Weighted average of stratum effects
        weights = np.array(stratum_weights) / np.sum(stratum_weights)
        stratified_ate = np.sum(weights * np.array(stratum_effects))
        
        return {
            'stratified_ate': stratified_ate,
            'stratum_effects': stratum_effects,
            'stratum_weights': weights
        }


class FactorialDesign:
    """
    Factorial Design Implementation
    
    Implements factorial designs and their analysis.
    """
    
    def __init__(self):
        self.factors = None
        self.levels = None
        self.design_matrix = None
        
    def create_2x2_design(self, n_replicates=1):
        """
        Create 2x2 factorial design.
        
        Parameters:
        -----------
        n_replicates : int
            Number of replicates per treatment combination
            
        Returns:
        --------
        design_matrix : DataFrame
            Design matrix with factor levels
        """
        # Create all combinations
        factor_a = np.repeat([0, 1], 2 * n_replicates)
        factor_b = np.tile(np.repeat([0, 1], n_replicates), 2)
        
        design_matrix = pd.DataFrame({
            'Factor_A': factor_a,
            'Factor_B': factor_b,
            'Treatment': [f'A{i}B{j}' for i, j in zip(factor_a, factor_b)]
        })
        
        self.design_matrix = design_matrix
        self.factors = ['Factor_A', 'Factor_B']
        self.levels = [2, 2]
        
        return design_matrix
    
    def create_2k_design(self, k, n_replicates=1):
        """
        Create 2^k factorial design.
        
        Parameters:
        -----------
        k : int
            Number of factors
        n_replicates : int
            Number of replicates per treatment combination
            
        Returns:
        --------
        design_matrix : DataFrame
            Design matrix with factor levels
        """
        # Generate all combinations
        combinations = []
        for i in range(2**k):
            combination = [int(bit) for bit in format(i, f'0{k}b')]
            combinations.extend([combination] * n_replicates)
        
        design_matrix = pd.DataFrame(combinations, 
                                   columns=[f'Factor_{chr(65+i)}' for i in range(k)])
        
        # Create treatment labels
        treatment_labels = []
        for combo in combinations:
            label = ''.join([f'{chr(65+i)}{level}' for i, level in enumerate(combo)])
            treatment_labels.append(label)
        
        design_matrix['Treatment'] = treatment_labels
        
        self.design_matrix = design_matrix
        self.factors = [f'Factor_{chr(65+i)}' for i in range(k)]
        self.levels = [2] * k
        
        return design_matrix
    
    def analyze_factorial(self, outcomes):
        """
        Analyze factorial design using ANOVA.
        
        Parameters:
        -----------
        outcomes : array-like
            Outcome measurements
            
        Returns:
        --------
        results : dict
            ANOVA results including main effects and interactions
        """
        if self.design_matrix is None:
            raise ValueError("No design matrix available")
        
        # Add outcomes to design matrix
        design_with_outcomes = self.design_matrix.copy()
        design_with_outcomes['Outcome'] = outcomes
        
        # Perform ANOVA
        if len(self.factors) == 2:
            # 2x2 factorial design
            from scipy.stats import f_oneway
            
            # Main effects
            factor_a_levels = design_with_outcomes['Factor_A'].unique()
            factor_b_levels = design_with_outcomes['Factor_B'].unique()
            
            # Factor A main effect
            factor_a_groups = [design_with_outcomes[design_with_outcomes['Factor_A'] == level]['Outcome'].values 
                              for level in factor_a_levels]
            f_stat_a, p_value_a = f_oneway(*factor_a_groups)
            
            # Factor B main effect
            factor_b_groups = [design_with_outcomes[design_with_outcomes['Factor_B'] == level]['Outcome'].values 
                              for level in factor_b_levels]
            f_stat_b, p_value_b = f_oneway(*factor_b_groups)
            
            # Interaction effect (simplified)
            interaction_groups = []
            for a_level in factor_a_levels:
                for b_level in factor_b_levels:
                    group = design_with_outcomes[(design_with_outcomes['Factor_A'] == a_level) & 
                                               (design_with_outcomes['Factor_B'] == b_level)]['Outcome'].values
                    interaction_groups.append(group)
            
            f_stat_interaction, p_value_interaction = f_oneway(*interaction_groups)
            
            results = {
                'factor_a': {'f_stat': f_stat_a, 'p_value': p_value_a},
                'factor_b': {'f_stat': f_stat_b, 'p_value': p_value_b},
                'interaction': {'f_stat': f_stat_interaction, 'p_value': p_value_interaction}
            }
        
        return results
    
    def calculate_main_effects(self, outcomes):
        """
        Calculate main effects for factorial design.
        
        Parameters:
        -----------
        outcomes : array-like
            Outcome measurements
            
        Returns:
        --------
        main_effects : dict
            Main effects for each factor
        """
        if self.design_matrix is None:
            raise ValueError("No design matrix available")
        
        design_with_outcomes = self.design_matrix.copy()
        design_with_outcomes['Outcome'] = outcomes
        
        main_effects = {}
        
        for factor in self.factors:
            factor_levels = design_with_outcomes[factor].unique()
            level_means = []
            
            for level in factor_levels:
                level_mean = design_with_outcomes[design_with_outcomes[factor] == level]['Outcome'].mean()
                level_means.append(level_mean)
            
            # Main effect is difference between levels
            main_effect = level_means[1] - level_means[0]
            main_effects[factor] = main_effect
        
        return main_effects


class PowerAnalysis:
    """
    Power Analysis Implementation
    
    Implements power analysis for various experimental designs.
    """
    
    def __init__(self):
        pass
    
    def power_t_test(self, effect_size, n1, n2, alpha=0.05):
        """
        Calculate power for two-sample t-test.
        
        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        n1 : int
            Sample size for group 1
        n2 : int
            Sample size for group 2
        alpha : float
            Significance level
            
        Returns:
        --------
        power : float
            Statistical power
        """
        # Calculate degrees of freedom
        df = n1 + n2 - 2
        
        # Calculate non-centrality parameter
        ncp = effect_size / np.sqrt(1/n1 + 1/n2)
        
        # Calculate critical value
        t_critical = t.ppf(1 - alpha/2, df)
        
        # Calculate power
        power = 1 - t.cdf(t_critical, df, ncp) + t.cdf(-t_critical, df, ncp)
        
        return power
    
    def sample_size_t_test(self, effect_size, power=0.8, alpha=0.05, ratio=1):
        """
        Calculate required sample size for two-sample t-test.
        
        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        power : float
            Desired power
        alpha : float
            Significance level
        ratio : float
            Ratio of sample sizes (n2/n1)
            
        Returns:
        --------
        n1 : int
            Sample size for group 1
        n2 : int
            Sample size for group 2
        """
        # Calculate required sample size
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n1 = int(np.ceil((z_alpha + z_beta)**2 * (1 + ratio) / (effect_size**2 * ratio)))
        n2 = int(n1 * ratio)
        
        return n1, n2
    
    def power_proportion_test(self, p1, p2, n1, n2, alpha=0.05):
        """
        Calculate power for proportion test.
        
        Parameters:
        -----------
        p1 : float
            Proportion for group 1
        p2 : float
            Proportion for group 2
        n1 : int
            Sample size for group 1
        n2 : int
            Sample size for group 2
        alpha : float
            Significance level
            
        Returns:
        --------
        power : float
            Statistical power
        """
        # Calculate pooled proportion
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Calculate standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        # Calculate effect size
        effect_size = (p2 - p1) / se
        
        # Calculate power
        z_alpha = norm.ppf(1 - alpha/2)
        power = 1 - norm.cdf(z_alpha - effect_size) + norm.cdf(-z_alpha - effect_size)
        
        return power
    
    def sample_size_proportion_test(self, p1, p2, power=0.8, alpha=0.05, ratio=1):
        """
        Calculate required sample size for proportion test.
        
        Parameters:
        -----------
        p1 : float
            Proportion for group 1
        p2 : float
            Proportion for group 2
        power : float
            Desired power
        alpha : float
            Significance level
        ratio : float
            Ratio of sample sizes (n2/n1)
            
        Returns:
        --------
        n1 : int
            Sample size for group 1
        n2 : int
            Sample size for group 2
        """
        # Calculate pooled proportion
        pooled_p = (p1 + ratio * p2) / (1 + ratio)
        
        # Calculate required sample size
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        numerator = (z_alpha + z_beta)**2 * (pooled_p * (1 - pooled_p) * (1 + ratio))
        denominator = (p2 - p1)**2 * ratio
        
        n1 = int(np.ceil(numerator / denominator))
        n2 = int(n1 * ratio)
        
        return n1, n2


class ABTesting:
    """
    A/B Testing Implementation
    
    Implements A/B testing frameworks including sequential testing and multi-armed bandits.
    """
    
    def __init__(self):
        self.variant_a_data = None
        self.variant_b_data = None
        
    def design_ab_test(self, baseline_conversion, expected_improvement, power=0.8, alpha=0.05):
        """
        Design A/B test with required sample size.
        
        Parameters:
        -----------
        baseline_conversion : float
            Baseline conversion rate
        expected_improvement : float
            Expected improvement in conversion rate
        power : float
            Desired power
        alpha : float
            Significance level
            
        Returns:
        --------
        sample_size : int
            Required sample size per variant
        """
        new_conversion = baseline_conversion + expected_improvement
        
        n1, n2 = self.sample_size_proportion_test(
            baseline_conversion, new_conversion, power, alpha
        )
        
        return n1
    
    def analyze_ab_test(self, variant_a_data, variant_b_data, alpha=0.05):
        """
        Analyze A/B test results.
        
        Parameters:
        -----------
        variant_a_data : array-like
            Data for variant A
        variant_b_data : array-like
            Data for variant B
        alpha : float
            Significance level
            
        Returns:
        --------
        results : dict
            Analysis results
        """
        # Calculate proportions
        p_a = np.mean(variant_a_data)
        p_b = np.mean(variant_b_data)
        
        # Sample sizes
        n_a = len(variant_a_data)
        n_b = len(variant_b_data)
        
        # Pooled proportion
        pooled_p = (p_a * n_a + p_b * n_b) / (n_a + n_b)
        
        # Standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
        
        # Test statistic
        z_stat = (p_b - p_a) / se
        
        # p-value
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        
        # Confidence interval
        z_critical = norm.ppf(1 - alpha/2)
        ci_lower = (p_b - p_a) - z_critical * se
        ci_upper = (p_b - p_a) + z_critical * se
        
        # Effect size
        effect_size = (p_b - p_a) / np.sqrt(pooled_p * (1 - pooled_p))
        
        results = {
            'p_a': p_a,
            'p_b': p_b,
            'difference': p_b - p_a,
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_size': effect_size,
            'significant': p_value < alpha
        }
        
        return results
    
    def sequential_probability_ratio_test(self, variant_a_data, variant_b_data, alpha=0.05, beta=0.1):
        """
        Implement Sequential Probability Ratio Test (SPRT).
        
        Parameters:
        -----------
        variant_a_data : array-like
            Data for variant A
        variant_b_data : array-like
            Data for variant B
        alpha : float
            Type I error rate
        beta : float
            Type II error rate
            
        Returns:
        --------
        results : dict
            SPRT results
        """
        # Calculate boundaries
        a_boundary = np.log((1 - beta) / alpha)
        b_boundary = np.log(beta / (1 - alpha))
        
        # Calculate test statistic
        p_a = np.mean(variant_a_data)
        p_b = np.mean(variant_b_data)
        
        # Log-likelihood ratio
        n_a = len(variant_a_data)
        n_b = len(variant_b_data)
        
        successes_a = np.sum(variant_a_data)
        successes_b = np.sum(variant_b_data)
        
        # Test statistic
        test_stat = (successes_b * np.log(p_b/p_a) + 
                    (n_b - successes_b) * np.log((1-p_b)/(1-p_a)) -
                    successes_a * np.log(p_a/p_a) - 
                    (n_a - successes_a) * np.log((1-p_a)/(1-p_a)))
        
        # Decision
        if test_stat >= a_boundary:
            decision = 'reject_null'
        elif test_stat <= b_boundary:
            decision = 'accept_null'
        else:
            decision = 'continue'
        
        results = {
            'test_statistic': test_stat,
            'a_boundary': a_boundary,
            'b_boundary': b_boundary,
            'decision': decision
        }
        
        return results
    
    def epsilon_greedy_bandit(self, n_arms, epsilon=0.1, n_trials=1000):
        """
        Implement ε-greedy multi-armed bandit algorithm.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms
        epsilon : float
            Exploration rate
        n_trials : int
            Number of trials
            
        Returns:
        --------
        results : dict
            Bandit results
        """
        # Initialize
        true_means = np.random.normal(0, 1, n_arms)
        estimated_means = np.zeros(n_arms)
        arm_counts = np.zeros(n_arms)
        cumulative_reward = 0
        rewards = []
        arm_choices = []
        
        for trial in range(n_trials):
            # Choose arm
            if np.random.random() < epsilon:
                # Explore: choose random arm
                arm = np.random.randint(0, n_arms)
            else:
                # Exploit: choose best estimated arm
                arm = np.argmax(estimated_means)
            
            # Get reward
            reward = np.random.normal(true_means[arm], 1)
            
            # Update estimates
            arm_counts[arm] += 1
            estimated_means[arm] = ((estimated_means[arm] * (arm_counts[arm] - 1) + reward) / 
                                   arm_counts[arm])
            
            cumulative_reward += reward
            rewards.append(reward)
            arm_choices.append(arm)
        
        results = {
            'true_means': true_means,
            'estimated_means': estimated_means,
            'arm_counts': arm_counts,
            'cumulative_reward': cumulative_reward,
            'rewards': rewards,
            'arm_choices': arm_choices
        }
        
        return results


class BlockingDesign:
    """
    Blocking Design Implementation
    
    Implements randomized block designs and Latin square designs.
    """
    
    def __init__(self):
        self.blocks = None
        self.treatments = None
        
    def randomized_block_design(self, n_blocks, n_treatments, n_replicates=1):
        """
        Create randomized block design.
        
        Parameters:
        -----------
        n_blocks : int
            Number of blocks
        n_treatments : int
            Number of treatments
        n_replicates : int
            Number of replicates per treatment in each block
            
        Returns:
        --------
        design_matrix : DataFrame
            Design matrix
        """
        blocks = []
        treatments = []
        replicates = []
        
        for block in range(n_blocks):
            for treatment in range(n_treatments):
                for replicate in range(n_replicates):
                    blocks.append(block)
                    treatments.append(treatment)
                    replicates.append(replicate)
        
        design_matrix = pd.DataFrame({
            'Block': blocks,
            'Treatment': treatments,
            'Replicate': replicates
        })
        
        self.blocks = blocks
        self.treatments = treatments
        
        return design_matrix
    
    def latin_square_design(self, n):
        """
        Create n x n Latin square design.
        
        Parameters:
        -----------
        n : int
            Size of Latin square
            
        Returns:
        --------
        design_matrix : DataFrame
            Latin square design matrix
        """
        # Create Latin square
        square = np.zeros((n, n), dtype=int)
        
        # Fill first row
        square[0] = np.arange(n)
        
        # Fill remaining rows with cyclic shifts
        for i in range(1, n):
            square[i] = (square[i-1] + 1) % n
        
        # Create design matrix
        rows = []
        cols = []
        treatments = []
        
        for i in range(n):
            for j in range(n):
                rows.append(i)
                cols.append(j)
                treatments.append(square[i, j])
        
        design_matrix = pd.DataFrame({
            'Row': rows,
            'Column': cols,
            'Treatment': treatments
        })
        
        return design_matrix
    
    def analyze_blocked_design(self, outcomes):
        """
        Analyze randomized block design.
        
        Parameters:
        -----------
        outcomes : array-like
            Outcome measurements
            
        Returns:
        --------
        results : dict
            Analysis results
        """
        if self.blocks is None or self.treatments is None:
            raise ValueError("No blocking design available")
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'Block': self.blocks,
            'Treatment': self.treatments,
            'Outcome': outcomes
        })
        
        # Calculate means
        grand_mean = np.mean(outcomes)
        block_means = analysis_df.groupby('Block')['Outcome'].mean()
        treatment_means = analysis_df.groupby('Treatment')['Outcome'].mean()
        
        # Calculate effects
        block_effects = block_means - grand_mean
        treatment_effects = treatment_means - grand_mean
        
        # ANOVA (simplified)
        from scipy.stats import f_oneway
        
        # Treatment effect
        treatment_groups = [analysis_df[analysis_df['Treatment'] == t]['Outcome'].values 
                           for t in analysis_df['Treatment'].unique()]
        f_stat_treatment, p_value_treatment = f_oneway(*treatment_groups)
        
        # Block effect
        block_groups = [analysis_df[analysis_df['Block'] == b]['Outcome'].values 
                       for b in analysis_df['Block'].unique()]
        f_stat_block, p_value_block = f_oneway(*block_groups)
        
        results = {
            'grand_mean': grand_mean,
            'block_effects': block_effects.to_dict(),
            'treatment_effects': treatment_effects.to_dict(),
            'treatment_f_stat': f_stat_treatment,
            'treatment_p_value': p_value_treatment,
            'block_f_stat': f_stat_block,
            'block_p_value': p_value_block
        }
        
        return results


def create_sample_experimental_data():
    """
    Create sample data for experimental design demonstrations.
    """
    np.random.seed(42)
    
    # RCT data
    n_treatment = 50
    n_control = 50
    
    # Treatment effect
    treatment_effect = 5.0
    treatment_outcomes = np.random.normal(85, 8, n_treatment) + treatment_effect
    control_outcomes = np.random.normal(85, 8, n_control)
    
    # Factorial design data
    n_replicates = 4
    factor_a_effects = [0, 3]  # No effect, positive effect
    factor_b_effects = [0, 2]  # No effect, positive effect
    interaction_effects = [[0, 1], [1, 0]]  # Interaction effects
    
    factorial_outcomes = []
    for a_level in [0, 1]:
        for b_level in [0, 1]:
            for rep in range(n_replicates):
                outcome = (80 + factor_a_effects[a_level] + factor_b_effects[b_level] + 
                          interaction_effects[a_level][b_level] + np.random.normal(0, 2))
                factorial_outcomes.append(outcome)
    
    # A/B test data
    n_ab = 1000
    variant_a_data = np.random.binomial(1, 0.15, n_ab)  # 15% conversion
    variant_b_data = np.random.binomial(1, 0.18, n_ab)  # 18% conversion
    
    # Blocked design data
    n_blocks = 5
    n_treatments = 3
    block_effects = np.random.normal(0, 2, n_blocks)
    treatment_effects = [0, 3, 6]  # Treatment effects
    
    blocked_outcomes = []
    for block in range(n_blocks):
        for treatment in range(n_treatments):
            outcome = (80 + block_effects[block] + treatment_effects[treatment] + 
                      np.random.normal(0, 2))
            blocked_outcomes.append(outcome)
    
    return {
        'treatment_outcomes': treatment_outcomes,
        'control_outcomes': control_outcomes,
        'factorial_outcomes': factorial_outcomes,
        'variant_a_data': variant_a_data,
        'variant_b_data': variant_b_data,
        'blocked_outcomes': blocked_outcomes
    }


def demonstrate_rct():
    """
    Demonstrate Randomized Controlled Trial design and analysis.
    """
    print("=" * 60)
    print("RANDOMIZED CONTROLLED TRIAL DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_experimental_data()
    
    # Simple randomization
    rct = RandomizedControlledTrial()
    assignments = rct.simple_randomization(50, 50, seed=42)
    
    print("1. Simple Randomization:")
    print(f"   Treatment assignments: {np.sum(assignments)} treatment, {len(assignments) - np.sum(assignments)} control")
    
    # Analyze RCT
    all_outcomes = np.concatenate([data['treatment_outcomes'], data['control_outcomes']])
    results = rct.analyze_rct(all_outcomes)
    
    print(f"\n2. RCT Analysis Results:")
    print(f"   ATE: {results['ate']:.3f}")
    print(f"   Standard Error: {results['se']:.3f}")
    print(f"   95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    print(f"   t-statistic: {results['t_stat']:.3f}")
    print(f"   p-value: {results['p_value']:.4f}")
    print(f"   Significant: {results['p_value'] < 0.05}")
    
    # Stratified randomization
    print(f"\n3. Stratified Randomization:")
    strata = np.random.choice(['Young', 'Middle', 'Old'], 100)
    n_treatment_per_stratum = {'Young': 15, 'Middle': 20, 'Old': 15}
    n_control_per_stratum = {'Young': 15, 'Middle': 20, 'Old': 15}
    
    stratified_assignments = rct.stratified_randomization(
        strata, n_treatment_per_stratum, n_control_per_stratum, seed=42
    )
    
    print(f"   Stratified assignments created")
    
    # Stratified analysis
    stratified_results = rct.stratified_analysis(all_outcomes)
    print(f"   Stratified ATE: {stratified_results['stratified_ate']:.3f}")


def demonstrate_factorial_design():
    """
    Demonstrate factorial design and analysis.
    """
    print("\n" + "=" * 60)
    print("FACTORIAL DESIGN DEMONSTRATION")
    print("=" * 60)
    
    # Create factorial design
    factorial = FactorialDesign()
    design_matrix = factorial.create_2x2_design(n_replicates=4)
    
    print("1. 2x2 Factorial Design:")
    print(design_matrix.head(8))
    
    # Analyze factorial design
    data = create_sample_experimental_data()
    results = factorial.analyze_factorial(data['factorial_outcomes'])
    
    print(f"\n2. Factorial Analysis Results:")
    print(f"   Factor A - F-stat: {results['factor_a']['f_stat']:.3f}, p-value: {results['factor_a']['p_value']:.4f}")
    print(f"   Factor B - F-stat: {results['factor_b']['f_stat']:.3f}, p-value: {results['factor_b']['p_value']:.4f}")
    print(f"   Interaction - F-stat: {results['interaction']['f_stat']:.3f}, p-value: {results['interaction']['p_value']:.4f}")
    
    # Calculate main effects
    main_effects = factorial.calculate_main_effects(data['factorial_outcomes'])
    print(f"\n3. Main Effects:")
    for factor, effect in main_effects.items():
        print(f"   {factor}: {effect:.3f}")
    
    # Create 2^3 design
    print(f"\n4. 2^3 Factorial Design:")
    design_3 = factorial.create_2k_design(3, n_replicates=2)
    print(design_3.head())


def demonstrate_power_analysis():
    """
    Demonstrate power analysis for different designs.
    """
    print("\n" + "=" * 60)
    print("POWER ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    power_analysis = PowerAnalysis()
    
    # Power analysis for t-test
    print("1. Power Analysis for t-test:")
    effect_size = 0.5  # Medium effect
    power = power_analysis.power_t_test(effect_size, 30, 30, alpha=0.05)
    print(f"   Effect size: {effect_size}")
    print(f"   Sample sizes: 30, 30")
    print(f"   Power: {power:.3f}")
    
    # Sample size calculation
    n1, n2 = power_analysis.sample_size_t_test(effect_size, power=0.8, alpha=0.05)
    print(f"\n2. Sample Size Calculation:")
    print(f"   Required sample sizes: {n1}, {n2}")
    
    # Power analysis for proportion test
    print(f"\n3. Power Analysis for Proportion Test:")
    p1, p2 = 0.15, 0.18  # Conversion rates
    power_prop = power_analysis.power_proportion_test(p1, p2, 1000, 1000, alpha=0.05)
    print(f"   Baseline conversion: {p1}")
    print(f"   New conversion: {p2}")
    print(f"   Sample sizes: 1000, 1000")
    print(f"   Power: {power_prop:.3f}")
    
    # Sample size for proportion test
    n1_prop, n2_prop = power_analysis.sample_size_proportion_test(p1, p2, power=0.8, alpha=0.05)
    print(f"\n4. Sample Size for Proportion Test:")
    print(f"   Required sample sizes: {n1_prop}, {n2_prop}")


def demonstrate_ab_testing():
    """
    Demonstrate A/B testing frameworks.
    """
    print("\n" + "=" * 60)
    print("A/B TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_experimental_data()
    
    # Design A/B test
    ab_test = ABTesting()
    sample_size = ab_test.design_ab_test(baseline_conversion=0.15, expected_improvement=0.03, power=0.8)
    
    print("1. A/B Test Design:")
    print(f"   Baseline conversion: 15%")
    print(f"   Expected improvement: 3%")
    print(f"   Required sample size per variant: {sample_size}")
    
    # Analyze A/B test
    results = ab_test.analyze_ab_test(data['variant_a_data'], data['variant_b_data'])
    
    print(f"\n2. A/B Test Analysis:")
    print(f"   Variant A conversion: {results['p_a']:.3f}")
    print(f"   Variant B conversion: {results['p_b']:.3f}")
    print(f"   Difference: {results['difference']:.3f}")
    print(f"   z-statistic: {results['z_stat']:.3f}")
    print(f"   p-value: {results['p_value']:.4f}")
    print(f"   95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    print(f"   Significant: {results['significant']}")
    
    # Sequential testing
    sprt_results = ab_test.sequential_probability_ratio_test(
        data['variant_a_data'][:500], data['variant_b_data'][:500]
    )
    
    print(f"\n3. Sequential Testing (SPRT):")
    print(f"   Test statistic: {sprt_results['test_statistic']:.3f}")
    print(f"   Decision: {sprt_results['decision']}")
    
    # Multi-armed bandit
    bandit_results = ab_test.epsilon_greedy_bandit(n_arms=3, epsilon=0.1, n_trials=1000)
    
    print(f"\n4. Multi-Armed Bandit (ε-Greedy):")
    print(f"   True means: {bandit_results['true_means']}")
    print(f"   Estimated means: {bandit_results['estimated_means']}")
    print(f"   Arm counts: {bandit_results['arm_counts']}")
    print(f"   Cumulative reward: {bandit_results['cumulative_reward']:.3f}")


def demonstrate_blocking():
    """
    Demonstrate blocking designs.
    """
    print("\n" + "=" * 60)
    print("BLOCKING DESIGN DEMONSTRATION")
    print("=" * 60)
    
    # Create randomized block design
    blocking = BlockingDesign()
    block_design = blocking.randomized_block_design(n_blocks=5, n_treatments=3, n_replicates=1)
    
    print("1. Randomized Block Design:")
    print(block_design.head(10))
    
    # Analyze blocked design
    data = create_sample_experimental_data()
    block_results = blocking.analyze_blocked_design(data['blocked_outcomes'])
    
    print(f"\n2. Blocked Design Analysis:")
    print(f"   Grand mean: {block_results['grand_mean']:.3f}")
    print(f"   Treatment F-stat: {block_results['treatment_f_stat']:.3f}")
    print(f"   Treatment p-value: {block_results['treatment_p_value']:.4f}")
    print(f"   Block F-stat: {block_results['block_f_stat']:.3f}")
    print(f"   Block p-value: {block_results['block_p_value']:.4f}")
    
    # Latin square design
    latin_square = blocking.latin_square_design(3)
    
    print(f"\n3. 3x3 Latin Square Design:")
    print(latin_square)


def demonstrate_practical_applications():
    """
    Demonstrate practical applications of experimental design.
    """
    print("\n" + "=" * 60)
    print("PRACTICAL APPLICATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Clinical trial design
    print("1. Clinical Trial Design:")
    print("   Phase I: Safety and dose finding (20-80 patients)")
    print("   Phase II: Efficacy and safety (100-300 patients)")
    print("   Phase III: Confirmatory efficacy (300-3000 patients)")
    
    # Agricultural experiment
    print(f"\n2. Agricultural Experiment:")
    print("   Split-plot design for irrigation methods and crop varieties")
    print("   Whole plots: Irrigation methods (drip, sprinkler)")
    print("   Subplots: Varieties (A, B, C, D)")
    print("   Blocks: Fields")
    
    # Industrial experiment
    print(f"\n3. Industrial Experiment:")
    print("   Response surface methodology for chemical process")
    print("   Factors: Temperature, pressure, catalyst concentration")
    print("   Response: Yield percentage")
    print("   Design: Central composite design")
    
    # Social science experiment
    print(f"\n4. Social Science Experiment:")
    print("   Behavioral economics study")
    print("   Treatment: Automatic enrollment vs. opt-in")
    print("   Control: Standard enrollment process")
    print("   Outcome: Participation rate")
    
    # Digital optimization
    print(f"\n5. Digital Optimization:")
    print("   A/B testing for website optimization")
    print("   Variants: Current design vs. new design")
    print("   Metrics: Conversion rate, click-through rate")
    print("   Platform: Online experimentation")


def main():
    """
    Main function to run all demonstrations.
    """
    print("EXPERIMENTAL DESIGN IMPLEMENTATION")
    print("Comprehensive demonstration of experimental design concepts and applications")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrate_rct()
    demonstrate_factorial_design()
    demonstrate_power_analysis()
    demonstrate_ab_testing()
    demonstrate_blocking()
    demonstrate_practical_applications()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTAL DESIGN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("✓ Randomized controlled trials with simple and stratified randomization")
    print("✓ Factorial designs (2x2 and 2^k) with ANOVA analysis")
    print("✓ Power analysis for t-tests and proportion tests")
    print("✓ A/B testing with sequential testing and multi-armed bandits")
    print("✓ Blocking designs including randomized blocks and Latin squares")
    print("✓ Sample size determination for various experimental designs")
    print("✓ Practical applications in clinical, agricultural, and industrial settings")
    print("✓ Causal inference through proper experimental design")
    print("✓ Multiple testing considerations")
    print("✓ Real-world experimental scenarios")


if __name__ == "__main__":
    main() 