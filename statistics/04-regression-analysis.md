# Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

Regression analysis is a fundamental statistical technique for modeling relationships between variables. This chapter covers linear regression, multiple regression, model diagnostics, and their applications in AI/ML.

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

### Basic Concepts

```python
def generate_linear_data(n=100):
    """Generate synthetic linear data"""
    x = np.random.uniform(0, 10, n)
    true_slope = 2.5
    true_intercept = 1.0
    noise = np.random.normal(0, 1, n)
    y = true_slope * x + true_intercept + noise
    
    return x, y, true_slope, true_intercept

x, y, true_slope, true_intercept = generate_linear_data(100)

# Manual calculation of regression coefficients
def manual_linear_regression(x, y):
    """Calculate linear regression coefficients manually"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared

slope, intercept, r_squared = manual_linear_regression(x, y)

print("Simple Linear Regression Results")
print(f"True slope: {true_slope:.3f}, Estimated slope: {slope:.3f}")
print(f"True intercept: {true_intercept:.3f}, Estimated intercept: {intercept:.3f}")
print(f"R-squared: {r_squared:.3f}")

# Visualize the regression
plt.figure(figsize=(12, 4))

# Scatter plot with regression line
plt.subplot(1, 3, 1)
plt.scatter(x, y, alpha=0.7, color='skyblue')
x_line = np.linspace(0, 10, 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.plot(x_line, true_slope * x_line + true_intercept, 'g--', linewidth=2, label=f'True: y = {true_slope:.2f}x + {true_intercept:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()

# Residuals
plt.subplot(1, 3, 2)
y_pred = slope * x + intercept
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.7, color='lightgreen')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Y')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Q-Q plot of residuals
plt.subplot(1, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()
```

### Using Scikit-learn

```python
def sklearn_linear_regression(x, y):
    """Perform linear regression using scikit-learn"""
    # Reshape x for sklearn
    X = x.reshape(-1, 1)
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    return model, y_pred, r2, mse, rmse

model, y_pred, r2, mse, rmse = sklearn_linear_regression(x, y)

print("Scikit-learn Linear Regression")
print(f"Slope: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R-squared: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, x.reshape(-1, 1), y, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Using Statsmodels

```python
def statsmodels_linear_regression(x, y):
    """Perform linear regression using statsmodels"""
    # Add constant for intercept
    X = sm.add_constant(x)
    
    # Create and fit model
    model = sm.OLS(y, X).fit()
    
    return model

sm_model = statsmodels_linear_regression(x, y)

print("Statsmodels Linear Regression")
print(sm_model.summary())

# Extract key statistics
print(f"\nKey Statistics:")
print(f"R-squared: {sm_model.rsquared:.3f}")
print(f"Adjusted R-squared: {sm_model.rsquared_adj:.3f}")
print(f"F-statistic: {sm_model.fvalue:.3f}")
print(f"P-value (F-test): {sm_model.f_pvalue:.3e}")
print(f"AIC: {sm_model.aic:.3f}")
print(f"BIC: {sm_model.bic:.3f}")
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