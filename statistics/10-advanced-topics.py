"""
Advanced Topics in Statistics: Python Implementations

This module implements advanced statistical methods including:
- Non-parametric methods (Mann-Whitney U, Kruskal-Wallis, permutation tests)
- Survival analysis (Kaplan-Meier, Cox proportional hazards)
- Mixed models (random intercept/slope)
- Causal inference (propensity score matching, IV, DoWhy)
- Robust statistics (median, MAD, robust regression)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, ttest_ind, median_abs_deviation
from sklearn.utils import resample
from lifelines import KaplanMeierFitter, CoxPHFitter
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, LinearRegression
from sklearn.model_selection import train_test_split

# 1. Non-Parametric Methods

def demonstrate_nonparametric_methods():
    print("\n=== Non-Parametric Methods ===")
    # Mann-Whitney U test
    group_a = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    group_b = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    u_stat, p_u = mannwhitneyu(group_a, group_b, alternative='two-sided')
    print(f"Mann-Whitney U: U={u_stat}, p={p_u:.4f}")

    # Kruskal-Wallis test
    group_c = np.array([3, 4, 5, 6, 7, 8])
    h_stat, p_kw = kruskal(group_a, group_b, group_c)
    print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f}")

    # Permutation test
    def stat(x, y):
        return np.mean(x) - np.mean(y)
    observed = stat(group_a, group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    n_perms = 1000
    perm_stats = []
    for _ in range(n_perms):
        np.random.shuffle(combined)
        perm_stats.append(stat(combined[:n_a], combined[n_a:]))
    p_perm = np.mean(np.abs(perm_stats) >= np.abs(observed))
    print(f"Permutation test: observed diff={observed:.2f}, p={p_perm:.4f}")

# 2. Survival Analysis

def demonstrate_survival_analysis():
    print("\n=== Survival Analysis ===")
    # Synthetic survival data
    durations = np.array([6, 8, 12, 15, 18])
    event_observed = np.array([1, 0, 1, 1, 0])
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed)
    print("Kaplan-Meier survival probabilities:")
    print(kmf.survival_function_)
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.show()

    # Cox proportional hazards model
    df = pd.DataFrame({
        'time': [6, 8, 12, 15, 18],
        'event': [1, 0, 1, 1, 0],
        'treatment': [0, 1, 0, 1, 0],
        'age': [60, 65, 70, 72, 68]
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    print("\nCox model summary:")
    print(cph.summary)
    cph.plot()
    plt.title("Cox Model Coefficient Plot")
    plt.show()

# 3. Mixed Models

def demonstrate_mixed_models():
    print("\n=== Mixed Models ===")
    # Simulate data: students within classrooms
    np.random.seed(42)
    n_classrooms = 10
    n_students = 20
    classroom_ids = np.repeat(np.arange(n_classrooms), n_students)
    student_effects = np.random.normal(0, 5, n_classrooms)
    scores = 75 + student_effects[classroom_ids] + np.random.normal(0, 10, n_classrooms * n_students)
    df = pd.DataFrame({'score': scores, 'classroom': classroom_ids})
    X = np.ones((len(scores), 1))  # intercept only
    model = MixedLM(df['score'], X, groups=df['classroom'])
    result = model.fit()
    print(result.summary())

# 4. Causal Inference

def demonstrate_causal_inference():
    print("\n=== Causal Inference ===")
    # Simulate data for propensity score matching
    np.random.seed(0)
    n = 200
    X = np.random.normal(0, 1, (n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    T = np.random.binomial(1, propensity)
    Y = 2 * T + X[:, 0] + np.random.normal(0, 1, n)
    df = pd.DataFrame({'T': T, 'Y': Y, 'X0': X[:, 0], 'X1': X[:, 1]})
    # Estimate propensity scores
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(df[['X0', 'X1']], df['T'])
    df['propensity'] = lr.predict_proba(df[['X0', 'X1']])[:, 1]
    # Match treated and control (nearest neighbor)
    treated = df[df['T'] == 1]
    control = df[df['T'] == 0]
    matched = treated.copy()
    matched['Y_control'] = control.loc[(control['propensity'].values[:, None] - treated['propensity'].values).argmin(axis=0), 'Y'].values
    ate = (matched['Y'] - matched['Y_control']).mean()
    print(f"Estimated ATE (propensity score matching): {ate:.3f}")

    # Instrumental Variables (2SLS)
    Z = np.random.binomial(1, 0.5, n)
    T_iv = 0.7 * Z + 0.3 * np.random.binomial(1, 0.5, n)
    Y_iv = 1.5 * T_iv + np.random.normal(0, 1, n)
    df_iv = pd.DataFrame({'Y': Y_iv, 'T': T_iv, 'Z': Z})
    # First stage
    first_stage = sm.OLS(df_iv['T'], sm.add_constant(df_iv['Z'])).fit()
    df_iv['T_hat'] = first_stage.fittedvalues
    # Second stage
    second_stage = sm.OLS(df_iv['Y'], sm.add_constant(df_iv['T_hat'])).fit()
    print(f"IV estimate (2SLS): {second_stage.params['T_hat']:.3f}")

    # DoWhy example (if installed)
    try:
        from dowhy import CausalModel
        data = df.copy()
        data['outcome'] = data['Y']
        data['treatment'] = data['T']
        causal_graph = """
digraph {
    X0 -> T;
    X1 -> T;
    T -> outcome;
    X0 -> outcome;
    X1 -> outcome;
}
"""
        model = CausalModel(
            data=data,
            treatment='treatment',
            outcome='outcome',
            graph=causal_graph
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
        print(f"DoWhy ATE estimate: {estimate.value}")
    except ImportError:
        print("DoWhy not installed; skipping DoWhy example.")

# 5. Robust Statistics

def demonstrate_robust_statistics():
    print("\n=== Robust Statistics ===")
    # Median and MAD
    data = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 1000])
    print(f"Mean: {np.mean(data):.1f}, Median: {np.median(data):.1f}, MAD: {median_abs_deviation(data):.1f}")

    # Robust regression (Huber)
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.normal(0, 1, 100)
    y[::10] += 20  # add outliers
    X_ = sm.add_constant(X)
    huber = RLM(y, X_, M=HuberT()).fit()
    ols = sm.OLS(y, X_).fit()
    print(f"OLS slope: {ols.params[1]:.2f}, Huber slope: {huber.params[1]:.2f}")

    # LAD regression
    lad = sm.QuantReg(y, X_).fit(q=0.5)
    print(f"LAD slope: {lad.params[1]:.2f}")

    # RANSAC and Theil-Sen
    X_rs = X.reshape(-1, 1)
    ransac = RANSACRegressor().fit(X_rs, y)
    ts = TheilSenRegressor().fit(X_rs, y)
    print(f"RANSAC slope: {ransac.estimator_.coef_[0]:.2f}, Theil-Sen slope: {ts.coef_[0]:.2f}")

    # Plot regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X, ols.predict(X_), label='OLS', lw=2)
    plt.plot(X, huber.predict(X_), label='Huber', lw=2)
    plt.plot(X, lad.predict(X_), label='LAD', lw=2)
    plt.plot(X, ransac.predict(X_rs), label='RANSAC', lw=2)
    plt.plot(X, ts.predict(X_rs), label='Theil-Sen', lw=2)
    plt.legend()
    plt.title('Robust Regression Methods')
    plt.show()


def main():
    print("ADVANCED TOPICS IN STATISTICS: DEMONSTRATION")
    demonstrate_nonparametric_methods()
    demonstrate_survival_analysis()
    demonstrate_mixed_models()
    demonstrate_causal_inference()
    demonstrate_robust_statistics()
    print("\nAll advanced topics demonstrated.")

if __name__ == "__main__":
    main() 