# Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

Regression analysis is a fundamental statistical technique for modeling relationships between variables. It's essential for prediction, understanding causal relationships, and making data-driven decisions in AI/ML.

## Table of Contents
- [Simple Linear Regression](#simple-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)
- [Model Diagnostics](#model-diagnostics)
- [Variable Selection](#variable-selection)
- [Polynomial Regression](#polynomial-regression)
- [Logistic Regression](#logistic-regression)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Simple Linear Regression

Simple linear regression models the relationship between a dependent variable (Y) and a single independent variable (X) using a linear function.

### Mathematical Foundation

**Model Specification:**
$$Y_i = \beta_0 + \beta_1 X_i + \epsilon_i, \quad i = 1, 2, \ldots, n$$

Where:
- $Y_i$ is the dependent variable for observation i
- $X_i$ is the independent variable for observation i
- $\beta_0$ is the intercept (y-intercept)
- $\beta_1$ is the slope coefficient
- $\epsilon_i$ is the error term (residual)

**Assumptions:**
1. **Linearity**: The relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Error variance is constant
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Not applicable for simple regression

**Least Squares Estimation:**
The goal is to minimize the sum of squared residuals:
$$\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2$$

**Normal Equations:**
$$\frac{\partial}{\partial \beta_0} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2 = 0$$
$$\frac{\partial}{\partial \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2 = 0$$

**Solution:**
$$\hat{\beta_1} = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$
$$\hat{\beta_0} = \bar{Y} - \hat{\beta_1} \bar{X}$$

**Derivation of Slope Coefficient:**
Starting with the normal equation for β₁:
$$\sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i) X_i = 0$$
$$\sum_{i=1}^{n} Y_i X_i - \beta_0 \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0$$

Substituting $\beta_0 = \bar{Y} - \beta_1 \bar{X}$:
$$\sum_{i=1}^{n} Y_i X_i - (\bar{Y} - \beta_1 \bar{X}) \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0$$
$$\sum_{i=1}^{n} Y_i X_i - \bar{Y} \sum_{i=1}^{n} X_i + \beta_1 \bar{X} \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0$$
$$\sum_{i=1}^{n} Y_i X_i - n \bar{Y} \bar{X} = \beta_1 (\sum_{i=1}^{n} X_i^2 - n \bar{X}^2)$$
$$\beta_1 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2}$$

```python
def simple_linear_regression(X, y):
    """
    Perform simple linear regression using least squares
    
    Mathematical implementation:
    β₁ = Cov(X,Y) / Var(X)
    β₀ = Ȳ - β₁X̄
    
    Parameters:
    X: array-like, independent variable
    y: array-like, dependent variable
    
    Returns:
    dict: regression results
    """
    X = np.array(X)
    y = np.array(y)
    
    n = len(X)
    
    # Calculate means
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate slope (β₁)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    if denominator == 0:
        raise ValueError("X has zero variance")
    
    beta_1 = numerator / denominator
    
    # Calculate intercept (β₀)
    beta_0 = y_mean - beta_1 * X_mean
    
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * X
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard errors
    mse = ss_res / (n - 2)  # Mean squared error
    se_beta_1 = np.sqrt(mse / np.sum((X - X_mean) ** 2))
    se_beta_0 = np.sqrt(mse * (1/n + X_mean**2 / np.sum((X - X_mean) ** 2)))
    
    return {
        'intercept': beta_0,
        'slope': beta_1,
        'r_squared': r_squared,
        'residuals': residuals,
        'y_pred': y_pred,
        'se_intercept': se_beta_0,
        'se_slope': se_beta_1,
        'mse': mse,
        'n': n
    }

def calculate_correlation(X, y):
    """
    Calculate correlation coefficient
    
    Mathematical implementation:
    r = Cov(X,Y) / (σ_X × σ_Y)
    
    Parameters:
    X: array-like, independent variable
    y: array-like, dependent variable
    
    Returns:
    float: correlation coefficient
    """
    X = np.array(X)
    y = np.array(y)
    
    # Calculate means
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate correlation
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2) * np.sum((y - y_mean) ** 2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

# Generate sample data
n_samples = 100
X = np.random.uniform(0, 10, n_samples)
true_slope = 2.5
true_intercept = 1.0
noise_std = 1.5

# True relationship: Y = 1.0 + 2.5*X + ε
y_true = true_intercept + true_slope * X
y = y_true + np.random.normal(0, noise_std, n_samples)

# Perform regression
results = simple_linear_regression(X, y)
correlation = calculate_correlation(X, y)

print("Simple Linear Regression Results:")
print(f"True intercept: {true_intercept:.2f}")
print(f"Estimated intercept: {results['intercept']:.4f}")
print(f"True slope: {true_slope:.2f}")
print(f"Estimated slope: {results['slope']:.4f}")
print(f"R-squared: {results['r_squared']:.4f}")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"Standard error (intercept): {results['se_intercept']:.4f}")
print(f"Standard error (slope): {results['se_slope']:.4f}")

# Verify mathematical relationship: R² = r² for simple linear regression
print(f"R² = r²: {abs(results['r_squared'] - correlation**2) < 1e-10}")

# Visualize the regression
plt.figure(figsize=(15, 10))

# Plot 1: Scatter plot with regression line
plt.subplot(2, 3, 1)
plt.scatter(X, y, alpha=0.6, color='blue', label='Data points')
plt.plot(X, y_true, 'r--', linewidth=2, label='True relationship')
plt.plot(X, results['y_pred'], 'g-', linewidth=2, label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals vs X
plt.subplot(2, 3, 2)
plt.scatter(X, results['residuals'], alpha=0.6, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals vs X')
plt.grid(True, alpha=0.3)

# Plot 3: Residuals vs Predicted
plt.subplot(2, 3, 3)
plt.scatter(results['y_pred'], results['residuals'], alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True, alpha=0.3)

# Plot 4: Q-Q plot of residuals
plt.subplot(2, 3, 4)
from scipy.stats import probplot
probplot(results['residuals'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True, alpha=0.3)

# Plot 5: Histogram of residuals
plt.subplot(2, 3, 5)
plt.hist(results['residuals'], bins=15, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True, alpha=0.3)

# Plot 6: Leverage plot (studentized residuals)
plt.subplot(2, 3, 6)
# Calculate leverage
X_with_const = np.column_stack([np.ones(len(X)), X])
H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
leverage = np.diag(H)

# Calculate studentized residuals
mse = results['mse']
studentized_residuals = results['residuals'] / np.sqrt(mse * (1 - leverage))

plt.scatter(leverage, studentized_residuals, alpha=0.6, color='brown')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')
plt.title('Leverage Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate the mathematical relationship between correlation and slope
print(f"\nMathematical Relationship:")
print(f"Correlation coefficient (r): {correlation:.4f}")
print(f"Slope coefficient (β₁): {results['slope']:.4f}")
print(f"Standard deviation of X: {np.std(X):.4f}")
print(f"Standard deviation of Y: {np.std(y):.4f}")
print(f"r × (σ_Y / σ_X) = {correlation * np.std(y) / np.std(X):.4f}")
print(f"β₁ = r × (σ_Y / σ_X): {abs(results['slope'] - correlation * np.std(y) / np.std(X)) < 1e-10}")
```

### Statistical Inference in Regression

**Hypothesis Testing for Slope:**
- **Null Hypothesis**: H₀: β₁ = 0 (no linear relationship)
- **Alternative Hypothesis**: H₁: β₁ ≠ 0 (linear relationship exists)

**Test Statistic:**
$$t = \frac{\hat{\beta_1} - 0}{\text{SE}(\hat{\beta_1})} \sim t_{n-2}$$

**Confidence Interval for Slope:**
$$\hat{\beta_1} \pm t_{\alpha/2, n-2} \times \text{SE}(\hat{\beta_1})$$

**Prediction Interval:**
For a new observation X₀:
$$\hat{Y}_0 \pm t_{\alpha/2, n-2} \times \sqrt{\text{MSE} \left(1 + \frac{1}{n} + \frac{(X_0 - \bar{X})^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}\right)}$$

```python
def regression_inference(X, y, alpha=0.05):
    """
    Perform statistical inference for regression parameters
    
    Mathematical implementation:
    t = β₁ / SE(β₁)
    CI = β₁ ± t_{α/2, n-2} × SE(β₁)
    
    Parameters:
    X: array-like, independent variable
    y: array-like, dependent variable
    alpha: float, significance level
    
    Returns:
    dict: inference results
    """
    results = simple_linear_regression(X, y)
    
    # Degrees of freedom
    df = results['n'] - 2
    
    # T-statistic for slope
    t_stat_slope = results['slope'] / results['se_slope']
    
    # P-value for slope (two-tailed)
    from scipy.stats import t
    p_value_slope = 2 * (1 - t.cdf(abs(t_stat_slope), df))
    
    # Critical value
    t_critical = t.ppf(1 - alpha/2, df)
    
    # Confidence intervals
    ci_slope_lower = results['slope'] - t_critical * results['se_slope']
    ci_slope_upper = results['slope'] + t_critical * results['se_slope']
    
    ci_intercept_lower = results['intercept'] - t_critical * results['se_intercept']
    ci_intercept_upper = results['intercept'] + t_critical * results['se_intercept']
    
    return {
        't_statistic_slope': t_stat_slope,
        'p_value_slope': p_value_slope,
        'ci_slope': (ci_slope_lower, ci_slope_upper),
        'ci_intercept': (ci_intercept_lower, ci_intercept_upper),
        't_critical': t_critical,
        'degrees_of_freedom': df,
        **results
    }

def prediction_interval(X, y, X_new, alpha=0.05):
    """
    Calculate prediction interval for new observations
    
    Mathematical implementation:
    PI = Ŷ₀ ± t_{α/2, n-2} × √(MSE × (1 + 1/n + (X₀-X̄)²/SSX))
    
    Parameters:
    X: array-like, independent variable
    y: array-like, dependent variable
    X_new: array-like, new X values
    alpha: float, significance level
    
    Returns:
    tuple: (predictions, lower_bounds, upper_bounds)
    """
    results = simple_linear_regression(X, y)
    
    # Degrees of freedom
    df = results['n'] - 2
    
    # Critical value
    from scipy.stats import t
    t_critical = t.ppf(1 - alpha/2, df)
    
    # Calculate predictions
    y_pred_new = results['intercept'] + results['slope'] * X_new
    
    # Calculate prediction intervals
    X_mean = np.mean(X)
    ssx = np.sum((X - X_mean) ** 2)
    
    # Standard error of prediction
    se_pred = np.sqrt(results['mse'] * (1 + 1/results['n'] + (X_new - X_mean)**2 / ssx))
    
    # Prediction intervals
    lower_bounds = y_pred_new - t_critical * se_pred
    upper_bounds = y_pred_new + t_critical * se_pred
    
    return y_pred_new, lower_bounds, upper_bounds

# Perform inference
inference_results = regression_inference(X, y, alpha=0.05)

print("Statistical Inference Results:")
print(f"T-statistic for slope: {inference_results['t_statistic_slope']:.4f}")
print(f"P-value for slope: {inference_results['p_value_slope']:.4f}")
print(f"95% CI for slope: ({inference_results['ci_slope'][0]:.4f}, {inference_results['ci_slope'][1]:.4f})")
print(f"95% CI for intercept: ({inference_results['ci_intercept'][0]:.4f}, {inference_results['ci_intercept'][1]:.4f})")
print(f"Degrees of freedom: {inference_results['degrees_of_freedom']}")

# Test if slope is significantly different from zero
alpha = 0.05
if inference_results['p_value_slope'] < alpha:
    print(f"Reject H₀: Slope is significantly different from zero (p < {alpha})")
else:
    print(f"Fail to reject H₀: No evidence that slope differs from zero (p ≥ {alpha})")

# Calculate prediction intervals
X_new = np.linspace(0, 10, 50)
y_pred_new, lower_bounds, upper_bounds = prediction_interval(X, y, X_new, alpha=0.05)

# Visualize inference results
plt.figure(figsize=(15, 5))

# Plot 1: Regression with confidence and prediction intervals
plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.6, color='blue', label='Data points')
plt.plot(X_new, y_pred_new, 'g-', linewidth=2, label='Regression line')
plt.fill_between(X_new, lower_bounds, upper_bounds, alpha=0.3, color='red', label='95% Prediction interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regression with Prediction Intervals')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: T-distribution with critical region
plt.subplot(1, 3, 2)
from scipy.stats import t
t_range = np.linspace(-4, 4, 1000)
t_pdf = t.pdf(t_range, inference_results['degrees_of_freedom'])

plt.plot(t_range, t_pdf, 'b-', linewidth=2, label=f't-distribution (df={inference_results["degrees_of_freedom"]})')
plt.axvline(inference_results['t_statistic_slope'], color='red', linestyle='--', 
            label=f't = {inference_results["t_statistic_slope"]:.3f}')
plt.axvline(inference_results['t_critical'], color='orange', linestyle=':', 
            label=f'Critical value = {inference_results["t_critical"]:.3f}')
plt.axvline(-inference_results['t_critical'], color='orange', linestyle=':', 
            label=f'Critical value = -{inference_results["t_critical"]:.3f}')
plt.fill_between(t_range, t_pdf, where=(t_range > inference_results['t_critical']) | (t_range < -inference_results['t_critical']), 
                 alpha=0.3, color='red', label='Rejection region')
plt.xlabel('t')
plt.ylabel('Probability Density')
plt.title('T-Distribution')
plt.legend()

# Plot 3: Confidence intervals
plt.subplot(1, 3, 3)
parameters = ['Intercept', 'Slope']
estimates = [inference_results['intercept'], inference_results['slope']]
ci_lower = [inference_results['ci_intercept'][0], inference_results['ci_slope'][0]]
ci_upper = [inference_results['ci_intercept'][1], inference_results['ci_slope'][1]]

x_pos = np.arange(len(parameters))
plt.errorbar(x_pos, estimates, yerr=[estimates[i] - ci_lower[i] for i in range(len(parameters))], 
             fmt='o', capsize=5, capthick=2, linewidth=2, label='95% CI')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='H₀: β = 0')
plt.xticks(x_pos, parameters)
plt.ylabel('Parameter Estimate')
plt.title('Parameter Estimates with 95% CI')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate the relationship between R² and correlation
print(f"\nRelationship between R² and correlation:")
print(f"R²: {inference_results['r_squared']:.4f}")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"Correlation squared: {correlation**2:.4f}")
print(f"R² = r²: {abs(inference_results['r_squared'] - correlation**2) < 1e-10}")
```

### Model Diagnostics

**Residual Analysis:**
1. **Normality**: Residuals should be normally distributed
2. **Independence**: Residuals should be independent
3. **Homoscedasticity**: Residual variance should be constant
4. **Linearity**: Relationship should be linear

**Influence Diagnostics:**
- **Leverage**: Measures how far an observation is from the center of X
- **Cook's Distance**: Measures the influence of each observation
- **DFFITS**: Measures the influence on fitted values

```python
def regression_diagnostics(X, y):
    """
    Perform comprehensive regression diagnostics
    
    Parameters:
    X: array-like, independent variable
    y: array-like, dependent variable
    
    Returns:
    dict: diagnostic results
    """
    results = simple_linear_regression(X, y)
    
    # Calculate leverage
    X_with_const = np.column_stack([np.ones(len(X)), X])
    H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
    leverage = np.diag(H)
    
    # Calculate studentized residuals
    mse = results['mse']
    studentized_residuals = results['residuals'] / np.sqrt(mse * (1 - leverage))
    
    # Calculate Cook's distance
    p = 2  # Number of parameters
    cook_distance = (studentized_residuals**2 / p) * (leverage / (1 - leverage))
    
    # Calculate DFFITS
    dffits = studentized_residuals * np.sqrt(leverage / (1 - leverage))
    
    # Test for normality (Shapiro-Wilk)
    from scipy.stats import shapiro
    shapiro_stat, shapiro_p = shapiro(results['residuals'])
    
    # Test for homoscedasticity (Breusch-Pagan)
    # Using a simplified version
    squared_residuals = results['residuals']**2
    X_with_const_resid = np.column_stack([np.ones(len(X)), X])
    try:
        from scipy.stats import f
        # Calculate R² for squared residuals
        res_results = simple_linear_regression(X, squared_residuals)
        bp_stat = res_results['r_squared'] * len(X)
        bp_p = 1 - chi2.cdf(bp_stat, 1)  # 1 degree of freedom
    except:
        bp_stat, bp_p = np.nan, np.nan
    
    return {
        'leverage': leverage,
        'studentized_residuals': studentized_residuals,
        'cook_distance': cook_distance,
        'dffits': dffits,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        **results
    }

# Perform diagnostics
diagnostics = regression_diagnostics(X, y)

print("Regression Diagnostics:")
print(f"Shapiro-Wilk test for normality:")
print(f"  Statistic: {diagnostics['shapiro_stat']:.4f}")
print(f"  P-value: {diagnostics['shapiro_p']:.4f}")
print(f"  Normality assumption: {'Rejected' if diagnostics['shapiro_p'] < 0.05 else 'Not rejected'}")

print(f"\nBreusch-Pagan test for homoscedasticity:")
print(f"  Statistic: {diagnostics['bp_stat']:.4f}")
print(f"  P-value: {diagnostics['bp_p']:.4f}")
print(f"  Homoscedasticity assumption: {'Rejected' if diagnostics['bp_p'] < 0.05 else 'Not rejected'}")

# Identify influential observations
high_leverage = diagnostics['leverage'] > 2 * (2 + 1) / len(X)  # 2(p+1)/n
high_cook = diagnostics['cook_distance'] > 4 / len(X)  # 4/n
high_dffits = abs(diagnostics['dffits']) > 2 * np.sqrt(2 / len(X))  # 2√(2/n)

print(f"\nInfluential Observations:")
print(f"High leverage: {np.sum(high_leverage)} observations")
print(f"High Cook's distance: {np.sum(high_cook)} observations")
print(f"High DFFITS: {np.sum(high_dffits)} observations")

# Visualize diagnostics
plt.figure(figsize=(15, 10))

# Plot 1: Residuals vs Fitted
plt.subplot(2, 3, 1)
plt.scatter(diagnostics['y_pred'], diagnostics['residuals'], alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True, alpha=0.3)

# Plot 2: Q-Q plot
plt.subplot(2, 3, 2)
probplot(diagnostics['residuals'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True, alpha=0.3)

# Plot 3: Leverage plot
plt.subplot(2, 3, 3)
plt.scatter(diagnostics['leverage'], diagnostics['studentized_residuals'], alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.axhline(y=2, color='orange', linestyle=':', alpha=0.7)
plt.axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')
plt.title('Leverage Plot')
plt.grid(True, alpha=0.3)

# Plot 4: Cook's distance
plt.subplot(2, 3, 4)
plt.plot(diagnostics['cook_distance'], 'o-', alpha=0.7, color='purple')
plt.axhline(y=4/len(X), color='red', linestyle='--', label='4/n threshold')
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: DFFITS
plt.subplot(2, 3, 5)
plt.plot(diagnostics['dffits'], 'o-', alpha=0.7, color='brown')
threshold = 2 * np.sqrt(2 / len(X))
plt.axhline(y=threshold, color='red', linestyle='--', label=f'±{threshold:.3f} threshold')
plt.axhline(y=-threshold, color='red', linestyle='--')
plt.xlabel('Observation')
plt.ylabel('DFFITS')
plt.title('DFFITS')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Scale-location plot
plt.subplot(2, 3, 6)
sqrt_abs_residuals = np.sqrt(np.abs(diagnostics['residuals']))
plt.scatter(diagnostics['y_pred'], sqrt_abs_residuals, alpha=0.6, color='orange')
plt.xlabel('Fitted Values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary of diagnostic findings
print(f"\nDiagnostic Summary:")
print(f"✓ Linearity: Check residuals vs fitted plot")
print(f"✓ Normality: {'✓' if diagnostics['shapiro_p'] >= 0.05 else '✗'} (Shapiro-Wilk p = {diagnostics['shapiro_p']:.4f})")
print(f"✓ Homoscedasticity: {'✓' if diagnostics['bp_p'] >= 0.05 else '✗'} (Breusch-Pagan p = {diagnostics['bp_p']:.4f})")
print(f"✓ Independence: Check for patterns in residuals vs fitted")
print(f"✓ No influential observations: {'✓' if np.sum(high_cook) == 0 else '✗'} ({np.sum(high_cook)} influential)")
```

## Multiple Linear Regression

### Multiple Variables

```python
def generate_multiple_regression_data(n=200):
    """Generate data for multiple regression"""
    # Generate features
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    
    # True coefficients
    true_coeffs = [2.0, -1.5, 0.8, 1.2]  # intercept, x1, x2, x3
    
    # Generate target with noise
    noise = np.random.normal(0, 0.5, n)
    y = true_coeffs[0] + true_coeffs[1] * x1 + true_coeffs[2] * x2 + true_coeffs[3] * x3 + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    return df, true_coeffs

df, true_coeffs = generate_multiple_regression_data()

# Multiple regression with scikit-learn
X = df[['x1', 'x2', 'x3']]
y = df['y']

model_multi = LinearRegression()
model_multi.fit(X, y)
y_pred_multi = model_multi.predict(X)

print("Multiple Linear Regression Results")
print(f"Intercept: {model_multi.intercept_:.3f}")
for i, (feature, coef) in enumerate(zip(['x1', 'x2', 'x3'], model_multi.coef_)):
    print(f"{feature} coefficient: {coef:.3f} (true: {true_coeffs[i+1]:.3f})")

print(f"R-squared: {r2_score(y, y_pred_multi):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_multi)):.3f}")

# Statsmodels multiple regression
X_sm = sm.add_constant(X)
sm_model_multi = sm.OLS(y, X_sm).fit()

print("\nStatsmodels Multiple Regression")
print(sm_model_multi.summary())

# Visualize multiple regression
plt.figure(figsize=(15, 5))

# Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y, y_pred_multi, alpha=0.7, color='skyblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Actual vs Predicted')

# Residuals
plt.subplot(1, 3, 2)
residuals_multi = y - y_pred_multi
plt.scatter(y_pred_multi, residuals_multi, alpha=0.7, color='lightgreen')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Coefficient comparison
plt.subplot(1, 3, 3)
features = ['x1', 'x2', 'x3']
estimated_coeffs = model_multi.coef_
true_coeffs_features = true_coeffs[1:]

x_pos = np.arange(len(features))
width = 0.35

plt.bar(x_pos - width/2, estimated_coeffs, width, label='Estimated', alpha=0.7)
plt.bar(x_pos + width/2, true_coeffs_features, width, label='True', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Coefficient Comparison')
plt.xticks(x_pos, features)
plt.legend()

plt.tight_layout()
plt.show()
```

## Model Diagnostics

### Assumption Checking

```python
def regression_diagnostics(model, X, y, y_pred):
    """Perform comprehensive regression diagnostics"""
    residuals = y - y_pred
    
    # 1. Linearity
    linearity_test = sm.stats.diagnostic.linear_harvey_collier(model)
    
    # 2. Normality of residuals
    normality_test = stats.shapiro(residuals)
    
    # 3. Homoscedasticity
    homoscedasticity_test = sm.stats.diagnostic.het_breuschpagan(residuals, X)
    
    # 4. Independence (Durbin-Watson)
    dw_stat = sm.stats.durbin_watson(residuals)
    
    # 5. Multicollinearity (VIF)
    vif_data = []
    for i in range(X.shape[1]):
        vif = sm.stats.outliers_influence.variance_inflation_factor(X.values, i)
        vif_data.append(vif)
    
    return {
        'linearity_pvalue': linearity_test[1],
        'normality_pvalue': normality_test[1],
        'homoscedasticity_pvalue': homoscedasticity_test[1],
        'durbin_watson': dw_stat,
        'vif': vif_data
    }

# Perform diagnostics
X_diag = sm.add_constant(X)
diagnostics = regression_diagnostics(sm_model_multi, X_diag, y, y_pred_multi)

print("Regression Diagnostics")
print(f"Linearity test p-value: {diagnostics['linearity_pvalue']:.3f}")
print(f"Normality test p-value: {diagnostics['normality_pvalue']:.3f}")
print(f"Homoscedasticity test p-value: {diagnostics['homoscedasticity_pvalue']:.3f}")
print(f"Durbin-Watson statistic: {diagnostics['durbin_watson']:.3f}")

print("\nVIF Values:")
for i, feature in enumerate(['x1', 'x2', 'x3']):
    print(f"{feature}: {diagnostics['vif'][i+1]:.3f}")

# Visualize diagnostics
plt.figure(figsize=(15, 10))

# Residuals vs Fitted
plt.subplot(2, 3, 1)
plt.scatter(y_pred_multi, residuals_multi, alpha=0.7, color='skyblue')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# Q-Q plot
plt.subplot(2, 3, 2)
stats.probplot(residuals_multi, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

# Scale-Location plot
plt.subplot(2, 3, 3)
residuals_abs = np.abs(residuals_multi)
plt.scatter(y_pred_multi, residuals_abs, alpha=0.7, color='lightgreen')
plt.xlabel('Fitted Values')
plt.ylabel('|Residuals|')
plt.title('Scale-Location Plot')

# Residuals vs Leverage
plt.subplot(2, 3, 4)
influence = sm_model_multi.get_influence()
leverage = influence.hat_matrix_diag
plt.scatter(leverage, residuals_multi, alpha=0.7, color='orange')
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.title('Residuals vs Leverage')

# Cook's Distance
plt.subplot(2, 3, 5)
cooks_distance = influence.cooks_distance[0]
plt.bar(range(len(cooks_distance)), cooks_distance, alpha=0.7, color='purple')
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")

# Histogram of residuals
plt.subplot(2, 3, 6)
plt.hist(residuals_multi, bins=20, alpha=0.7, color='pink', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

plt.tight_layout()
plt.show()
```

### Outlier Detection

```python
def detect_outliers(model, X, y, y_pred):
    """Detect outliers using various methods"""
    residuals = y - y_pred
    
    # 1. Standardized residuals
    std_residuals = residuals / np.std(residuals)
    outliers_std = np.abs(std_residuals) > 3
    
    # 2. Studentized residuals
    influence = model.get_influence()
    studentized_residuals = influence.resid_studentized_external
    outliers_studentized = np.abs(studentized_residuals) > 3
    
    # 3. Leverage
    leverage = influence.hat_matrix_diag
    leverage_threshold = 2 * (X.shape[1] + 1) / X.shape[0]
    outliers_leverage = leverage > leverage_threshold
    
    # 4. Cook's Distance
    cooks_distance = influence.cooks_distance[0]
    cooks_threshold = 4 / X.shape[0]
    outliers_cooks = cooks_distance > cooks_threshold
    
    return {
        'std_residuals': std_residuals,
        'studentized_residuals': studentized_residuals,
        'leverage': leverage,
        'cooks_distance': cooks_distance,
        'outliers_std': outliers_std,
        'outliers_studentized': outliers_studentized,
        'outliers_leverage': outliers_leverage,
        'outliers_cooks': outliers_cooks
    }

outlier_results = detect_outliers(sm_model_multi, X_diag, y, y_pred_multi)

print("Outlier Detection Results")
print(f"Outliers (standardized residuals): {np.sum(outlier_results['outliers_std'])}")
print(f"Outliers (studentized residuals): {np.sum(outlier_results['outliers_studentized'])}")
print(f"High leverage points: {np.sum(outlier_results['outliers_leverage'])}")
print(f"High Cook's distance: {np.sum(outlier_results['outliers_cooks'])}")

# Visualize outliers
plt.figure(figsize=(15, 5))

# Standardized residuals
plt.subplot(1, 3, 1)
plt.scatter(range(len(outlier_results['std_residuals'])), outlier_results['std_residuals'], 
           alpha=0.7, color='skyblue')
plt.axhline(3, color='red', linestyle='--', alpha=0.7)
plt.axhline(-3, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Observation')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals')

# Leverage
plt.subplot(1, 3, 2)
plt.scatter(range(len(outlier_results['leverage'])), outlier_results['leverage'], 
           alpha=0.7, color='lightgreen')
leverage_threshold = 2 * (X_diag.shape[1] + 1) / X_diag.shape[0]
plt.axhline(leverage_threshold, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Observation')
plt.ylabel('Leverage')
plt.title('Leverage')

# Cook's Distance
plt.subplot(1, 3, 3)
plt.bar(range(len(outlier_results['cooks_distance'])), outlier_results['cooks_distance'], 
        alpha=0.7, color='orange')
cooks_threshold = 4 / X_diag.shape[0]
plt.axhline(cooks_threshold, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")

plt.tight_layout()
plt.show()
```

## Variable Selection

### Stepwise Selection

```python
def stepwise_selection(X, y, direction='forward'):
    """Perform stepwise variable selection"""
    features = list(X.columns)
    selected_features = []
    remaining_features = features.copy()
    
    if direction == 'forward':
        while remaining_features:
            best_feature = None
            best_score = -np.inf
            
            for feature in remaining_features:
                current_features = selected_features + [feature]
                X_current = X[current_features]
                X_current = sm.add_constant(X_current)
                
                model = sm.OLS(y, X_current).fit()
                score = model.aic  # Lower AIC is better
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                print(f"Added {best_feature}, AIC: {best_score:.3f}")
            else:
                break
    
    return selected_features

# Generate more complex data for variable selection
def generate_complex_data(n=300):
    """Generate data with some irrelevant features"""
    np.random.seed(42)
    
    # Relevant features
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    
    # Irrelevant features
    x4 = np.random.normal(0, 1, n)
    x5 = np.random.normal(0, 1, n)
    x6 = np.random.normal(0, 1, n)
    
    # Target
    y = 2.0 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'y': y
    })
    
    return df

df_complex = generate_complex_data()
X_complex = df_complex[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y_complex = df_complex['y']

print("Stepwise Variable Selection")
selected_features = stepwise_selection(X_complex, y_complex, direction='forward')
print(f"Selected features: {selected_features}")

# Compare models
X_full = sm.add_constant(X_complex)
X_selected = sm.add_constant(X_complex[selected_features])

model_full = sm.OLS(y_complex, X_full).fit()
model_selected = sm.OLS(y_complex, X_selected).fit()

print(f"\nModel Comparison:")
print(f"Full model AIC: {model_full.aic:.3f}")
print(f"Selected model AIC: {model_selected.aic:.3f}")
print(f"Full model R²: {model_full.rsquared:.3f}")
print(f"Selected model R²: {model_selected.rsquared:.3f}")
```

### Regularization

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

def regularization_comparison(X, y):
    """Compare different regularization methods"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None
        }
    
    return results, models

reg_results, reg_models = regularization_comparison(X_complex, y_complex)

print("Regularization Comparison")
for name, result in reg_results.items():
    print(f"{name}: R² = {result['r2']:.3f}, RMSE = {result['rmse']:.3f}")

# Visualize coefficient shrinkage
plt.figure(figsize=(12, 4))

# Coefficient comparison
plt.subplot(1, 2, 1)
features = X_complex.columns
x_pos = np.arange(len(features))
width = 0.2

for i, (name, result) in enumerate(reg_results.items()):
    if result['coefficients'] is not None:
        plt.bar(x_pos + i*width, result['coefficients'], width, label=name, alpha=0.7)

plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Coefficient Comparison')
plt.xticks(x_pos + width*1.5, features, rotation=45)
plt.legend()

# Performance comparison
plt.subplot(1, 2, 2)
models = list(reg_results.keys())
r2_scores = [reg_results[name]['r2'] for name in models]
rmse_scores = [reg_results[name]['rmse'] for name in models]

x_pos = np.arange(len(models))
plt.bar(x_pos - 0.2, r2_scores, 0.4, label='R²', alpha=0.7)
plt.bar(x_pos + 0.2, rmse_scores, 0.4, label='RMSE', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance')
plt.xticks(x_pos, models)
plt.legend()

plt.tight_layout()
plt.show()
```

## Polynomial Regression

```python
def polynomial_regression_example():
    """Demonstrate polynomial regression"""
    # Generate non-linear data
    x = np.random.uniform(-3, 3, 100)
    y = 2 + 3*x - 0.5*x**2 + 0.1*x**3 + np.random.normal(0, 0.5, 100)
    
    # Fit polynomial models of different degrees
    degrees = [1, 2, 3, 4, 5]
    models = {}
    scores = {}
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        models[degree] = model
        scores[degree] = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
    
    return x, y, models, scores, degrees

x_poly, y_poly, poly_models, poly_scores, degrees = polynomial_regression_example()

print("Polynomial Regression Results")
for degree in degrees:
    print(f"Degree {degree}: R² = {poly_scores[degree]['r2']:.3f}, RMSE = {poly_scores[degree]['rmse']:.3f}")

# Visualize polynomial fits
plt.figure(figsize=(15, 5))

# Data and fits
plt.subplot(1, 3, 1)
plt.scatter(x_poly, y_poly, alpha=0.7, color='skyblue', label='Data')

x_plot = np.linspace(-3, 3, 100)
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_plot = poly.fit_transform(x_plot.reshape(-1, 1))
    y_plot = poly_models[degree].predict(X_plot)
    plt.plot(x_plot, y_plot, linewidth=2, label=f'Degree {degree}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Fits')
plt.legend()

# R-squared vs degree
plt.subplot(1, 3, 2)
r2_values = [poly_scores[d]['r2'] for d in degrees]
plt.plot(degrees, r2_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('R²')
plt.title('R² vs Polynomial Degree')
plt.grid(True, alpha=0.3)

# RMSE vs degree
plt.subplot(1, 3, 3)
rmse_values = [poly_scores[d]['rmse'] for d in degrees]
plt.plot(degrees, rmse_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('RMSE vs Polynomial Degree')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Logistic Regression

```python
def logistic_regression_example():
    """Demonstrate logistic regression"""
    # Generate binary classification data
    np.random.seed(42)
    n = 200
    
    # Features
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # True coefficients
    true_coeffs = [0.5, 1.2, -0.8]  # intercept, x1, x2
    
    # Generate probabilities
    logits = true_coeffs[0] + true_coeffs[1] * x1 + true_coeffs[2] * x2
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Generate binary outcomes
    y = np.random.binomial(1, probabilities)
    
    # Create DataFrame
    df_logistic = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })
    
    return df_logistic, true_coeffs

df_logistic, true_coeffs = logistic_regression_example()

# Fit logistic regression
X_logistic = df_logistic[['x1', 'x2']]
y_logistic = df_logistic['y']

# Scikit-learn
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_logistic, y_logistic)
y_pred_logistic = model_logistic.predict(X_logistic)
y_prob_logistic = model_logistic.predict_proba(X_logistic)[:, 1]

# Statsmodels
X_logistic_sm = sm.add_constant(X_logistic)
sm_model_logistic = sm.Logit(y_logistic, X_logistic_sm).fit()

print("Logistic Regression Results")
print(f"Scikit-learn coefficients: {model_logistic.intercept_[0]:.3f}, {model_logistic.coef_[0]}")
print(f"True coefficients: {true_coeffs}")
print(f"Accuracy: {np.mean(y_pred_logistic == y_logistic):.3f}")

print("\nStatsmodels Results:")
print(sm_model_logistic.summary())

# Visualize logistic regression
plt.figure(figsize=(15, 5))

# Data with decision boundary
plt.subplot(1, 3, 1)
scatter = plt.scatter(X_logistic['x1'], X_logistic['x2'], c=y_logistic, cmap='RdYlBu', alpha=0.7)

# Decision boundary
x1_min, x1_max = X_logistic['x1'].min(), X_logistic['x1'].max()
x2_min, x2_max = X_logistic['x2'].min(), X_logistic['x2'].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
Z = model_logistic.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='red', linewidths=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression Decision Boundary')
plt.colorbar(scatter)

# Probability surface
plt.subplot(1, 3, 2)
Z_prob = model_logistic.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1]
Z_prob = Z_prob.reshape(xx1.shape)
contour = plt.contourf(xx1, xx2, Z_prob, levels=20, cmap='RdYlBu')
plt.colorbar(contour)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Probability Surface')

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_logistic, y_prob_logistic)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()
```

## Practical Applications

### Real Estate Price Prediction

```python
def real_estate_example():
    """Simulate real estate price prediction"""
    np.random.seed(42)
    n = 500
    
    # Generate features
    square_feet = np.random.normal(2000, 500, n)
    bedrooms = np.random.poisson(3, n)
    bathrooms = np.random.poisson(2, n)
    age = np.random.exponential(10, n)
    distance_to_city = np.random.exponential(5, n)
    
    # Generate price with realistic relationships
    base_price = 200000
    price = (base_price + 
             100 * square_feet + 
             15000 * bedrooms + 
             25000 * bathrooms - 
             2000 * age - 
             5000 * distance_to_city + 
             np.random.normal(0, 20000, n))
    
    df_real_estate = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'distance_to_city': distance_to_city,
        'price': price
    })
    
    return df_real_estate

df_real_estate = real_estate_example()

# Fit model
X_real_estate = df_real_estate.drop('price', axis=1)
y_real_estate = df_real_estate['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_real_estate, y_real_estate, 
                                                    test_size=0.3, random_state=42)

# Fit multiple models
models_real_estate = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

results_real_estate = {}

for name, model in models_real_estate.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results_real_estate[name] = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': np.mean(np.abs(y_test - y_pred))
    }

print("Real Estate Price Prediction Results")
for name, result in results_real_estate.items():
    print(f"{name}: R² = {result['r2']:.3f}, RMSE = ${result['rmse']:.0f}, MAE = ${result['mae']:.0f}")

# Feature importance
model_linear = models_real_estate['Linear']
feature_importance = pd.DataFrame({
    'feature': X_real_estate.columns,
    'coefficient': model_linear.coef_
})
feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)

print(f"\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['coefficient']:.2f}")

# Visualize results
plt.figure(figsize=(15, 5))

# Actual vs Predicted
plt.subplot(1, 3, 1)
y_pred_linear = models_real_estate['Linear'].predict(X_test)
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='skyblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# Feature importance
plt.subplot(1, 3, 2)
plt.barh(feature_importance['feature'], feature_importance['coefficient'], alpha=0.7)
plt.xlabel('Coefficient')
plt.title('Feature Importance')

# Residuals
plt.subplot(1, 3, 3)
residuals_real_estate = y_test - y_pred_linear
plt.scatter(y_pred_linear, residuals_real_estate, alpha=0.7, color='lightgreen')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()
```

## Practice Problems

1. **Model Comparison**: Create a function that compares multiple regression models and provides comprehensive diagnostics.

2. **Feature Engineering**: Implement automated feature engineering techniques (polynomial features, interactions, etc.).

3. **Cross-Validation**: Build a robust cross-validation framework for regression models.

4. **Model Interpretation**: Create functions to interpret regression coefficients and their significance.

## Further Reading

- "Applied Linear Regression Models" by Kutner, Nachtsheim, and Neter
- "Introduction to Linear Regression Analysis" by Montgomery, Peck, and Vining
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Regression Analysis by Example" by Chatterjee and Hadi

## Key Takeaways

- **Linear regression** models linear relationships between variables
- **Multiple regression** extends to multiple predictors
- **Model diagnostics** are crucial for validating assumptions
- **Variable selection** helps build parsimonious models
- **Regularization** prevents overfitting and improves generalization
- **Polynomial regression** captures non-linear relationships
- **Logistic regression** handles binary classification problems
- **Cross-validation** provides reliable model performance estimates

In the next chapter, we'll explore time series analysis, including trend analysis, seasonality, and forecasting techniques. 