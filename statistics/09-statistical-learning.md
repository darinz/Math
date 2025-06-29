# Statistical Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

# Chapter 9: Statistical Learning

## Overview

Statistical learning encompasses the methods and techniques used to build predictive models from data. This chapter covers cross-validation, model selection, regularization, ensemble methods, and model evaluation - all essential skills for machine learning and data science.

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement various cross-validation techniques
- Understand the bias-variance tradeoff
- Apply regularization methods to prevent overfitting
- Use ensemble methods for improved predictions
- Evaluate model performance using appropriate metrics
- Select optimal models using information criteria

## Prerequisites

- Understanding of regression and classification concepts
- Familiarity with scikit-learn
- Basic knowledge of probability and statistics

## Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, 
    LeaveOneOut, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

---

## 1. Cross-Validation

Cross-validation is a technique for assessing how well a model will generalize to new, unseen data. It helps prevent overfitting and provides a more reliable estimate of model performance.

### 1.1 Holdout Validation

The simplest form of validation splits data into training and test sets.

```python
# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = 2*X[:, 0] + 1.5*X[:, 1] - 0.5*X[:, 2] + np.random.normal(0, 0.1, 1000)

# Holdout validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
```

### 1.2 K-Fold Cross-Validation

K-fold cross-validation divides data into K folds and trains K models, each using K-1 folds for training and 1 fold for validation.

```python
# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(LinearRegression(), X, y, cv=kfold, scoring='r2')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize CV scores
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), cv_scores, 'bo-')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('K-Fold Cross-Validation Scores')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(cv_scores)
plt.ylabel('R² Score')
plt.title('Distribution of CV Scores')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 1.3 Leave-One-Out Cross-Validation

LOOCV uses n-1 samples for training and 1 sample for validation, repeated n times.

```python
# Leave-One-Out CV (computationally expensive for large datasets)
loocv = LeaveOneOut()
loocv_scores = cross_val_score(LinearRegression(), X[:100], y[:100], cv=loocv, scoring='r2')

print(f"LOOCV mean score: {loocv_scores.mean():.4f}")
print(f"LOOCV std score: {loocv_scores.std():.4f}")
```

### 1.4 Stratified Cross-Validation

For classification problems, stratified CV maintains the proportion of samples for each class.

```python
# Generate classification data
np.random.seed(42)
X_clf = np.random.randn(1000, 5)
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

# Stratified K-fold for classification
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_scores = cross_val_score(
    LogisticRegression(), X_clf, y_clf, 
    cv=stratified_kfold, scoring='accuracy'
)

print(f"Stratified CV accuracy: {clf_scores.mean():.4f} (+/- {clf_scores.std() * 2:.4f})")
```

---

## 2. Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between model complexity and generalization error.

### 2.1 Understanding Bias and Variance

```python
def generate_polynomial_data(n_samples=100, noise=0.1, degree=1):
    """Generate polynomial data with noise"""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y_true = 2 * X + 1  # True linear relationship
    y_noisy = y_true + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y_noisy, y_true

def fit_polynomial(X, y, degree):
    """Fit polynomial of given degree"""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    model = Pipeline([
        ('poly', poly),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    return model

# Generate data
X, y, y_true = generate_polynomial_data(n_samples=50, noise=0.5)

# Fit models with different complexities
degrees = [1, 3, 10, 15]
models = []
predictions = []

for degree in degrees:
    model = fit_polynomial(X, y, degree)
    models.append(model)
    pred = model.predict(X)
    predictions.append(pred)

# Visualize bias-variance tradeoff
plt.figure(figsize=(15, 10))

for i, (degree, pred) in enumerate(zip(degrees, predictions)):
    plt.subplot(2, 2, i+1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'g-', linewidth=2, label='True relationship')
    plt.plot(X, pred, 'r-', linewidth=2, label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate bias and variance
def calculate_bias_variance(X, y, y_true, model, n_iterations=100):
    """Calculate bias and variance of model predictions"""
    predictions = []
    
    for _ in range(n_iterations):
        # Add noise to data
        y_noisy = y + np.random.normal(0, 0.1, len(y))
        model.fit(X, y_noisy)
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    
    # Bias: difference between mean prediction and true values
    bias = np.mean((mean_pred - y_true) ** 2)
    
    # Variance: variability of predictions
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias, variance

# Calculate for different polynomial degrees
bias_scores = []
variance_scores = []

for degree in degrees:
    model = fit_polynomial(X, y, degree)
    bias, variance = calculate_bias_variance(X, y, y_true, model)
    bias_scores.append(bias)
    variance_scores.append(variance)

# Plot bias-variance tradeoff
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(degrees, bias_scores, 'bo-', label='Bias²')
plt.plot(degrees, variance_scores, 'ro-', label='Variance')
plt.xlabel('Polynomial Degree')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
total_error = np.array(bias_scores) + np.array(variance_scores)
plt.plot(degrees, total_error, 'go-', label='Total Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('Total Error')
plt.title('Total Error (Bias² + Variance)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 3. Model Selection

Model selection involves choosing the best model from a set of candidates based on performance metrics and complexity.

### 3.1 Information Criteria

Information criteria balance model fit with complexity.

```python
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def calculate_aic_bic(X, y, model):
    """Calculate AIC and BIC for a model"""
    model.fit(X, y)
    y_pred = model.predict(X)
    n = len(y)
    k = X.shape[1] + 1  # +1 for intercept
    
    # Calculate RSS
    rss = np.sum((y - y_pred) ** 2)
    
    # AIC = n * log(RSS/n) + 2k
    aic = n * np.log(rss/n) + 2*k
    
    # BIC = n * log(RSS/n) + k*log(n)
    bic = n * np.log(rss/n) + k*np.log(n)
    
    return aic, bic

# Compare different polynomial models
aic_scores = []
bic_scores = []
mse_scores = []

for degree in range(1, 11):
    model = fit_polynomial(X, y, degree)
    aic, bic = calculate_aic_bic(X, y, model)
    mse = mean_squared_error(y, model.predict(X))
    
    aic_scores.append(aic)
    bic_scores.append(bic)
    mse_scores.append(mse)

# Plot information criteria
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, 11), mse_scores, 'bo-')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training MSE')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, 11), aic_scores, 'ro-')
plt.xlabel('Polynomial Degree')
plt.ylabel('AIC')
plt.title('Akaike Information Criterion')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, 11), bic_scores, 'go-')
plt.xlabel('Polynomial Degree')
plt.ylabel('BIC')
plt.title('Bayesian Information Criterion')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Best model by MSE: Degree {np.argmin(mse_scores) + 1}")
print(f"Best model by AIC: Degree {np.argmin(aic_scores) + 1}")
print(f"Best model by BIC: Degree {np.argmin(bic_scores) + 1}")
```

### 3.2 Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Visualize grid search results
results = grid_search.cv_results_
C_values = [0.1, 1, 10, 100]

plt.figure(figsize=(12, 5))

# Plot for RBF kernel
rbf_scores = []
for C in C_values:
    mask = (results['param_C'] == C) & (results['param_kernel'] == 'rbf') & (results['param_gamma'] == 'scale')
    if np.any(mask):
        rbf_scores.append(-results['mean_test_score'][mask][0])
    else:
        rbf_scores.append(np.nan)

plt.subplot(1, 2, 1)
plt.plot(C_values, rbf_scores, 'bo-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('MSE')
plt.title('SVR RBF Kernel Performance')
plt.grid(True)

# Plot for linear kernel
linear_scores = []
for C in C_values:
    mask = (results['param_C'] == C) & (results['param_kernel'] == 'linear')
    if np.any(mask):
        linear_scores.append(-results['mean_test_score'][mask][0])
    else:
        linear_scores.append(np.nan)

plt.subplot(1, 2, 2)
plt.plot(C_values, linear_scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('MSE')
plt.title('SVR Linear Kernel Performance')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 4. Regularization

Regularization techniques help prevent overfitting by adding constraints to the model parameters.

### 4.1 Ridge Regression (L2 Regularization)

```python
# Generate data with multicollinearity
np.random.seed(42)
X_ridge = np.random.randn(100, 10)
# Create multicollinearity
X_ridge[:, 2] = X_ridge[:, 0] + 0.1 * np.random.randn(100)
X_ridge[:, 3] = X_ridge[:, 1] + 0.1 * np.random.randn(100)

y_ridge = 2*X_ridge[:, 0] + 1.5*X_ridge[:, 1] + np.random.normal(0, 0.1, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_ridge, y_ridge, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare Linear Regression vs Ridge
alpha_values = [0, 0.1, 1, 10, 100]
train_scores = []
test_scores = []
coefficients = []

for alpha in alpha_values:
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)
    
    model.fit(X_train_scaled, y_train)
    train_scores.append(model.score(X_train_scaled, y_train))
    test_scores.append(model.score(X_test_scaled, y_test))
    coefficients.append(model.coef_)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(alpha_values, train_scores, 'bo-', label='Training')
plt.plot(alpha_values, test_scores, 'ro-', label='Test')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² Score')
plt.title('Ridge Regression Performance')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
coefficients = np.array(coefficients)
for i in range(coefficients.shape[1]):
    plt.plot(alpha_values, coefficients[:, i], 'o-', label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Shrinkage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.subplot(1, 3, 3)
# Show coefficient magnitudes
coef_magnitudes = np.abs(coefficients)
plt.imshow(coef_magnitudes.T, aspect='auto', cmap='viridis')
plt.colorbar(label='|Coefficient|')
plt.xlabel('Alpha Index')
plt.ylabel('Feature')
plt.title('Coefficient Magnitudes Heatmap')
plt.xticks(range(len(alpha_values)), [f'{a}' for a in alpha_values])

plt.tight_layout()
plt.show()
```

### 4.2 Lasso Regression (L1 Regularization)

```python
# Lasso regression
lasso_train_scores = []
lasso_test_scores = []
lasso_coefficients = []

for alpha in alpha_values:
    if alpha == 0:
        model = LinearRegression()
    else:
        model = Lasso(alpha=alpha)
    
    model.fit(X_train_scaled, y_train)
    lasso_train_scores.append(model.score(X_train_scaled, y_train))
    lasso_test_scores.append(model.score(X_test_scaled, y_test))
    lasso_coefficients.append(model.coef_)

# Visualize Lasso results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(alpha_values, lasso_train_scores, 'bo-', label='Training')
plt.plot(alpha_values, lasso_test_scores, 'ro-', label='Test')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² Score')
plt.title('Lasso Regression Performance')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
lasso_coefficients = np.array(lasso_coefficients)
for i in range(lasso_coefficients.shape[1]):
    plt.plot(alpha_values, lasso_coefficients[:, i], 'o-', label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Path')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.subplot(1, 3, 3)
# Count non-zero coefficients
non_zero_counts = np.sum(lasso_coefficients != 0, axis=1)
plt.plot(alpha_values, non_zero_counts, 'go-')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Number of Non-zero Coefficients')
plt.title('Feature Selection by Lasso')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare Ridge vs Lasso
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alpha_values, test_scores, 'bo-', label='Ridge')
plt.plot(alpha_values, lasso_test_scores, 'ro-', label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Test R² Score')
plt.title('Ridge vs Lasso Performance')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
ridge_coef_var = np.var(coefficients, axis=1)
lasso_coef_var = np.var(lasso_coefficients, axis=1)
plt.plot(alpha_values, ridge_coef_var, 'bo-', label='Ridge')
plt.plot(alpha_values, lasso_coef_var, 'ro-', label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Variance')
plt.title('Coefficient Stability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. Ensemble Methods

Ensemble methods combine multiple models to improve prediction accuracy and robustness.

### 5.1 Bagging (Bootstrap Aggregating)

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Generate data
np.random.seed(42)
X_ensemble = np.random.randn(200, 5)
y_ensemble = 2*X_ensemble[:, 0] + 1.5*X_ensemble[:, 1] + np.random.normal(0, 0.5, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X_ensemble, y_ensemble, test_size=0.3, random_state=42
)

# Compare single tree vs bagging
single_tree = DecisionTreeRegressor(random_state=42)
bagging = BaggingRegressor(
    DecisionTreeRegressor(random_state=42),
    n_estimators=100,
    random_state=42
)

# Train models
single_tree.fit(X_train, y_train)
bagging.fit(X_train, y_train)

# Evaluate
single_tree_score = single_tree.score(X_test, y_test)
bagging_score = bagging.score(X_test, y_test)

print(f"Single Tree R²: {single_tree_score:.4f}")
print(f"Bagging R²: {bagging_score:.4f}")

# Visualize predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, single_tree.predict(X_test), alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Single Tree Predictions')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, bagging.predict(X_test), alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Bagging Predictions')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 5.2 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print(f"Random Forest R²: {rf_score:.4f}")

# Feature importance
feature_importance = rf.feature_importances_
feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare ensemble methods
ensemble_scores = {
    'Single Tree': single_tree_score,
    'Bagging': bagging_score,
    'Random Forest': rf_score
}

plt.figure(figsize=(8, 6))
methods = list(ensemble_scores.keys())
scores = list(ensemble_scores.values())
colors = ['red', 'blue', 'green']

plt.bar(methods, scores, color=colors)
plt.ylabel('R² Score')
plt.title('Comparison of Ensemble Methods')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
```

### 5.3 Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)

print(f"Gradient Boosting R²: {gb_score:.4f}")

# Learning curves
train_scores = []
test_scores = []

for i in range(1, 101, 10):
    gb_partial = GradientBoostingRegressor(
        n_estimators=i,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_partial.fit(X_train, y_train)
    train_scores.append(gb_partial.score(X_train, y_train))
    test_scores.append(gb_partial.score(X_test, y_test))

# Plot learning curves
plt.figure(figsize=(10, 6))
n_estimators = range(1, 101, 10)
plt.plot(n_estimators, train_scores, 'bo-', label='Training')
plt.plot(n_estimators, test_scores, 'ro-', label='Test')
plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.title('Gradient Boosting Learning Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare all ensemble methods
all_scores = {
    'Single Tree': single_tree_score,
    'Bagging': bagging_score,
    'Random Forest': rf_score,
    'Gradient Boosting': gb_score
}

plt.figure(figsize=(10, 6))
methods = list(all_scores.keys())
scores = list(all_scores.values())
colors = ['red', 'blue', 'green', 'orange']

plt.bar(methods, scores, color=colors)
plt.ylabel('R² Score')
plt.title('Comparison of All Ensemble Methods')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 6. Model Evaluation

Proper model evaluation is crucial for understanding model performance and making informed decisions.

### 6.1 Classification Metrics

```python
# Generate classification data
np.random.seed(42)
X_clf = np.random.randn(1000, 10)
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_clf, y_train_clf)
    y_pred = clf.predict(X_test_clf)
    y_pred_proba = clf.predict_proba(X_test_clf)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test_clf, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Compare accuracy
plt.figure(figsize=(10, 6))
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
colors = ['red', 'blue', 'green']

plt.bar(names, accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test_clf, result['probabilities'])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

# Precision-Recall Curves
plt.subplot(1, 2, 2)
for name, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test_clf, result['probabilities'])
    plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix for best classifier
best_classifier = max(results.keys(), key=lambda x: results[x]['accuracy'])
y_pred_best = results[best_classifier]['predictions']

cm = confusion_matrix(y_test_clf, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_classifier}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Detailed classification report
print(f"\nClassification Report - {best_classifier}")
print(classification_report(y_test_clf, y_pred_best))
```

### 6.2 Regression Metrics

```python
# Generate regression data
np.random.seed(42)
X_reg = np.random.randn(1000, 10)
y_reg = 2*X_reg[:, 0] + 1.5*X_reg[:, 1] - 0.5*X_reg[:, 2] + np.random.normal(0, 0.5, 1000)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train multiple regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

reg_results = {}

for name, reg in regressors.items():
    reg.fit(X_train_reg, y_train_reg)
    y_pred = reg.predict(X_test_reg)
    
    reg_results[name] = {
        'r2': reg.score(X_test_reg, y_test_reg),
        'mse': mean_squared_error(y_test_reg, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred)),
        'mae': np.mean(np.abs(y_test_reg - y_pred)),
        'predictions': y_pred
    }

# Compare metrics
metrics = ['r2', 'mse', 'rmse', 'mae']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, metric in enumerate(metrics):
    row, col = i // 2, i % 2
    names = list(reg_results.keys())
    values = [reg_results[name][metric] for name in names]
    
    axes[row, col].bar(names, values)
    axes[row, col].set_title(f'{metric.upper()} Comparison')
    axes[row, col].set_ylabel(metric.upper())
    axes[row, col].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for j, v in enumerate(values):
        axes[row, col].text(j, v + max(values)*0.01, f'{v:.4f}', 
                           ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Residual plots
plt.figure(figsize=(15, 10))
for i, (name, result) in enumerate(reg_results.items()):
    plt.subplot(2, 2, i+1)
    residuals = y_test_reg - result['predictions']
    plt.scatter(result['predictions'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {name}')
    plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. Practical Applications

### 7.1 Model Selection for House Price Prediction

```python
# Load sample data (simulated house prices)
np.random.seed(42)
n_samples = 1000

# Generate features
square_feet = np.random.uniform(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
distance_to_city = np.random.uniform(0, 30, n_samples)

# Generate target (house prices)
base_price = 200000
price = (base_price + 
         100 * square_feet + 
         15000 * bedrooms + 
         20000 * bathrooms - 
         2000 * age - 
         5000 * distance_to_city + 
         np.random.normal(0, 20000, n_samples))

# Create DataFrame
house_data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'distance_to_city': distance_to_city,
    'price': price
})

X_house = house_data.drop('price', axis=1)
y_house = house_data['price']

# Split data
X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(
    X_house, y_house, test_size=0.3, random_state=42
)

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Cross-validation comparison
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_house, y_train_house, 
                               cv=5, scoring='neg_mean_squared_error')
    cv_results[name] = {
        'mean_mse': -cv_scores.mean(),
        'std_mse': cv_scores.std(),
        'mean_rmse': np.sqrt(-cv_scores.mean())
    }

# Display results
print("Cross-Validation Results:")
print("-" * 50)
for name, result in cv_results.items():
    print(f"{name:20} RMSE: {result['mean_rmse']:.2f} ± {result['std_mse']:.2f}")

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
names = list(cv_results.keys())
rmses = [cv_results[name]['mean_rmse'] for name in names]
plt.bar(names, rmses)
plt.ylabel('RMSE')
plt.title('Cross-Validation RMSE Comparison')
plt.xticks(rotation=45)
for i, v in enumerate(rmses):
    plt.text(i, v + max(rmses)*0.01, f'{v:.0f}', ha='center', va='bottom')

# Test set performance
plt.subplot(1, 2, 2)
test_results = {}
for name, model in models.items():
    model.fit(X_train_house, y_train_house)
    y_pred = model.predict(X_test_house)
    test_results[name] = np.sqrt(mean_squared_error(y_test_house, y_pred))

test_rmses = list(test_results.values())
plt.bar(names, test_rmses)
plt.ylabel('RMSE')
plt.title('Test Set RMSE Comparison')
plt.xticks(rotation=45)
for i, v in enumerate(test_rmses):
    plt.text(i, v + max(test_rmses)*0.01, f'{v:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Feature importance for best model
best_model_name = min(test_results.keys(), key=lambda x: test_results[x])
best_model = models[best_model_name]
best_model.fit(X_train_house, y_train_house)

if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
else:
    importance = np.abs(best_model.coef_)

plt.figure(figsize=(10, 6))
feature_names = X_house.columns
plt.bar(feature_names, importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title(f'Feature Importance - {best_model_name}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 7.2 Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter distributions for Random Forest
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Randomized search
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_house, y_train_house)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {np.sqrt(-random_search.best_score_):.2f}")

# Compare with default parameters
default_rf = RandomForestRegressor(random_state=42)
default_rf.fit(X_train_house, y_train_house)
default_score = np.sqrt(mean_squared_error(y_test_house, default_rf.predict(X_test_house)))

tuned_rf = random_search.best_estimator_
tuned_score = np.sqrt(mean_squared_error(y_test_house, tuned_rf.predict(X_test_house)))

print(f"\nTest RMSE - Default: {default_score:.2f}")
print(f"Test RMSE - Tuned: {tuned_score:.2f}")
print(f"Improvement: {((default_score - tuned_score) / default_score * 100):.1f}%")

# Visualize parameter importance
param_importance = random_search.cv_results_
n_estimators_scores = []
max_depth_scores = []

for i in range(len(param_importance['param_n_estimators'])):
    n_estimators_scores.append((param_importance['param_n_estimators'][i], 
                               -param_importance['mean_test_score'][i]))
    max_depth_scores.append((param_importance['param_max_depth'][i], 
                           -param_importance['mean_test_score'][i]))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
n_est_values, n_est_scores = zip(*n_estimators_scores)
plt.scatter(n_est_values, n_est_scores, alpha=0.6)
plt.xlabel('n_estimators')
plt.ylabel('MSE')
plt.title('n_estimators vs MSE')
plt.grid(True)

plt.subplot(1, 2, 2)
depth_values, depth_scores = zip(*max_depth_scores)
plt.scatter(depth_values, depth_scores, alpha=0.6)
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.title('max_depth vs MSE')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 8. Practice Problems

### Problem 1: Cross-Validation Comparison
Compare the performance of different cross-validation strategies (K-fold, stratified K-fold, leave-one-out) on a classification dataset.

### Problem 2: Regularization Analysis
Generate data with multicollinearity and compare the performance of Linear Regression, Ridge Regression, and Lasso Regression.

### Problem 3: Ensemble Method Comparison
Implement and compare bagging, random forest, and gradient boosting on a regression problem.

### Problem 4: Model Selection
Use information criteria (AIC, BIC) and cross-validation to select the optimal polynomial degree for a regression problem.

### Problem 5: Hyperparameter Tuning
Use grid search and random search to tune hyperparameters for a machine learning model and compare the results.

---

## 9. Further Reading

### Books
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Papers
- "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection" by Kohavi
- "Random Forests" by Breiman
- "Greedy Function Approximation: A Gradient Boosting Machine" by Friedman

### Online Resources
- Scikit-learn documentation on model selection
- Cross-validation tutorials
- Ensemble methods guides

---

## 10. Key Takeaways

1. **Cross-validation** provides reliable estimates of model performance and helps prevent overfitting.

2. **The bias-variance tradeoff** is fundamental to understanding model complexity and generalization.

3. **Regularization** techniques (Ridge, Lasso) help prevent overfitting by constraining model parameters.

4. **Ensemble methods** (bagging, boosting, random forest) often provide better predictions than individual models.

5. **Model evaluation** requires multiple metrics and careful interpretation of results.

6. **Hyperparameter tuning** can significantly improve model performance but requires computational resources.

7. **Information criteria** (AIC, BIC) provide principled ways to balance model fit and complexity.

8. **Feature selection** and importance analysis help understand model behavior and improve interpretability.

---

**Next Chapter**: [Advanced Topics](10-advanced-topics.md) - Non-parametric methods, survival analysis, and specialized statistical techniques. 