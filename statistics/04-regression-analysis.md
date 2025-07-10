# Regression Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## Introduction

Regression analysis is a fundamental statistical technique for modeling relationships between variables. It's essential for prediction, understanding causal relationships, and making data-driven decisions in AI/ML.

### Why Regression Analysis Matters

Regression analysis is the cornerstone of predictive modeling and causal inference. It helps us:

1. **Understand Relationships**: Quantify how variables influence each other
2. **Make Predictions**: Forecast outcomes based on known variables
3. **Test Hypotheses**: Evaluate causal relationships between variables
4. **Control for Confounders**: Account for multiple factors simultaneously
5. **Optimize Processes**: Find optimal values for input variables

### Types of Regression

1. **Simple Linear Regression**: One predictor variable
2. **Multiple Linear Regression**: Multiple predictor variables
3. **Polynomial Regression**: Non-linear relationships
4. **Logistic Regression**: Binary outcome variables
5. **Ridge/Lasso Regression**: Regularized regression
6. **Non-linear Regression**: Complex functional forms

### The Regression Process

1. **Data Collection**: Gather relevant variables
2. **Model Specification**: Choose appropriate functional form
3. **Parameter Estimation**: Fit the model to data
4. **Model Validation**: Check assumptions and diagnostics
5. **Interpretation**: Understand coefficients and predictions
6. **Prediction**: Use model for new observations

## Table of Contents
- [Simple Linear Regression](#simple-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)
- [Model Diagnostics](#model-diagnostics)
- [Variable Selection](#variable-selection)
- [Polynomial Regression](#polynomial-regression)
- [Logistic Regression](#logistic-regression)
- [Practical Applications](#practical-applications)

## Setup

The examples in this chapter use Python libraries for regression analysis. We'll work with both theoretical concepts and practical implementations to build intuition and computational skills.

## Simple Linear Regression

Simple linear regression models the relationship between a dependent variable (Y) and a single independent variable (X) using a linear function.

### Understanding Linear Relationships

A linear relationship means that as one variable changes, the other variable changes at a constant rate. Think of it as a straight line relationship.

#### Intuitive Example: House Prices

Consider house prices vs. square footage:
- **X**: Square footage (independent variable)
- **Y**: House price (dependent variable)
- **Relationship**: As square footage increases, price increases at a constant rate
- **Model**: Price = $`\beta_0`$ + $`\beta_1`$ × Square footage

#### Visual Understanding

The goal is to find the "best" straight line through the data points:
- **Intercept ($`\beta_0`$)**: Where the line crosses the Y-axis
- **Slope ($`\beta_1`$)**: How much Y changes for each unit change in X
- **Residuals**: Vertical distances from points to the line

### Mathematical Foundation

**Model Specification:**
```math
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i, \quad i = 1, 2, \ldots, n
```

Where:
- $`Y_i`$ is the dependent variable for observation i
- $`X_i`$ is the independent variable for observation i
- $`\beta_0`$ is the intercept (y-intercept)
- $`\beta_1`$ is the slope coefficient
- $`\epsilon_i`$ is the error term (residual)

#### Intuitive Understanding

The model says: "The value of Y is a linear function of X, plus some random error." The error term accounts for:
- Measurement error
- Omitted variables
- Random variation
- Model misspecification

#### Example: Salary vs. Experience

For salary prediction:
- $`Y_i`$ = Salary of person i
- $`X_i`$ = Years of experience of person i
- $`\beta_0`$ = Starting salary (with 0 experience)
- $`\beta_1`$ = Salary increase per year of experience
- $`\epsilon_i`$ = Individual variation (skills, education, etc.)

**Assumptions:**
1. **Linearity**: The relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Error variance is constant
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Not applicable for simple regression

#### Understanding Assumptions

**Linearity**: The true relationship is a straight line
- **Violation**: Curved relationship (use polynomial regression)

**Independence**: Each observation is independent
- **Violation**: Time series data, clustered data

**Homoscedasticity**: Error variance is the same for all X values
- **Violation**: Errors get larger/smaller with X (heteroscedasticity)

**Normality**: Errors follow normal distribution
- **Violation**: Skewed or heavy-tailed errors

### Least Squares Estimation

The goal is to find the line that minimizes the sum of squared vertical distances from points to the line.

#### Mathematical Objective

**Least Squares Estimation:**
The goal is to minimize the sum of squared residuals:

```math
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
```

#### Intuitive Understanding

We want to find the line that makes the vertical distances (residuals) as small as possible. We square the distances to:
- Penalize large errors more heavily
- Avoid positive/negative cancellation
- Make the optimization problem mathematically tractable

#### Visual Interpretation

Imagine moving a line through the data points:
- **Good fit**: Small vertical distances from points to line
- **Poor fit**: Large vertical distances from points to line
- **Best fit**: Line that minimizes sum of squared distances

### Normal Equations

To find the optimal parameters, we set the derivatives equal to zero.

**Normal Equations:**
```math
\frac{\partial}{\partial \beta_0} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2 = 0
```
```math
\frac{\partial}{\partial \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2 = 0
```

#### Solution

**Solution:**
```math
\hat{\beta_1} = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
```
```math
\hat{\beta_0} = \bar{Y} - \hat{\beta_1} \bar{X}
```

#### Intuitive Understanding

**Slope ($`\hat{\beta_1}`$):**
- Numerator: How X and Y vary together (covariance)
- Denominator: How X varies (variance of X)
- Ratio: How much Y changes per unit change in X

**Intercept ($`\hat{\beta_0}`$):**
- Ensures the line passes through the point ($`\bar{X}, \bar{Y}`$)
- Y-intercept when X = 0

#### Example: Simple Calculation

Given data:
- $`\bar{X} = 5`$, $`\bar{Y} = 20`$
- $`\sum(X_i - \bar{X})(Y_i - \bar{Y}) = 30`$
- $`\sum(X_i - \bar{X})^2 = 10`$

Then:
- $`\hat{\beta_1} = \frac{30}{10} = 3`$
- $`\hat{\beta_0} = 20 - 3 \times 5 = 5`$
- Model: $`\hat{Y} = 5 + 3X`$

### Derivation of Slope Coefficient

**Derivation of Slope Coefficient:**
Starting with the normal equation for $`\beta_1`$:

```math
\sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i) X_i = 0
```
```math
\sum_{i=1}^{n} Y_i X_i - \beta_0 \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0
```

Substituting $`\beta_0 = \bar{Y} - \beta_1 \bar{X}`$:

```math
\sum_{i=1}^{n} Y_i X_i - (\bar{Y} - \beta_1 \bar{X}) \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0
```
```math
\sum_{i=1}^{n} Y_i X_i - \bar{Y} \sum_{i=1}^{n} X_i + \beta_1 \bar{X} \sum_{i=1}^{n} X_i - \beta_1 \sum_{i=1}^{n} X_i^2 = 0
```
```math
\sum_{i=1}^{n} Y_i X_i - n \bar{Y} \bar{X} = \beta_1 (\sum_{i=1}^{n} X_i^2 - n \bar{X}^2)
```
```math
\beta_1 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
```

#### Understanding the Derivation

This derivation shows that:
1. We start with the normal equation (derivative = 0)
2. We substitute the expression for $`\beta_0`$
3. We rearrange terms to isolate $`\beta_1`$
4. The result is the covariance divided by variance

### Model Evaluation

#### Coefficient of Determination (R²)

R² measures the proportion of variance in Y explained by X:

```math
R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}
```

Where:
- **SSR** = Sum of Squares Regression
- **SSE** = Sum of Squares Error
- **SST** = Sum of Squares Total

#### Intuitive Understanding

- **R² = 0**: Model explains none of the variance
- **R² = 1**: Model explains all of the variance
- **R² = 0.75**: Model explains 75% of the variance

#### Example: R² Interpretation

For house prices vs. square footage:
- **R² = 0.82**: Square footage explains 82% of house price variation
- **Remaining 18%**: Due to location, age, condition, etc.

#### Adjusted R²

For multiple regression, adjusted R² penalizes for number of predictors:

```math
R^2_{adj} = 1 - \frac{\text{SSE}/(n-p-1)}{\text{SST}/(n-1)}
```

Where p is the number of predictors.

### Statistical Inference in Regression

Statistical inference helps us determine if the relationship is statistically significant and estimate uncertainty.

#### Hypothesis Testing for Slope

**Hypothesis Testing for Slope:**
- **Null Hypothesis**: $`H_0: \beta_1 = 0`$ (no linear relationship)
- **Alternative Hypothesis**: $`H_1: \beta_1 \neq 0`$ (linear relationship exists)

#### Test Statistic

**Test Statistic:**
```math
t = \frac{\hat{\beta_1} - 0}{\text{SE}(\hat{\beta_1})} \sim t_{n-2}
```

#### Intuitive Understanding

The t-test asks: "Is the slope significantly different from zero?"
- **Large t-value**: Strong evidence against null hypothesis
- **Small t-value**: Weak evidence against null hypothesis
- **Degrees of freedom**: n-2 (we estimate 2 parameters)

#### Example: Salary vs. Experience

For salary regression:
- $`\hat{\beta_1} = 5000`$ (salary increase per year)
- $`\text{SE}(\hat{\beta_1}) = 800`$
- $`t = \frac{5000}{800} = 6.25`$
- p-value ≈ 0.0001
- Conclusion: Strong evidence that experience affects salary

#### Confidence Interval for Slope

**Confidence Interval for Slope:**
```math
\hat{\beta_1} \pm t_{\alpha/2, n-2} \times \text{SE}(\hat{\beta_1})
```

#### Example: Confidence Interval

For the salary example:
- 95% CI: $`5000 \pm 1.96 \times 800 = [3432, 6568]`$
- We are 95% confident that each year of experience increases salary by $3,432 to $6,568

#### Prediction Interval

For predicting a new observation:

**Prediction Interval:**
For a new observation $`X_0`$:

```math
\hat{Y}_0 \pm t_{\alpha/2, n-2} \times \sqrt{\text{MSE} \left(1 + \frac{1}{n} + \frac{(X_0 - \bar{X})^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}\right)}
```

#### Understanding Prediction Intervals

Prediction intervals account for:
1. **Model uncertainty**: Uncertainty in parameter estimates
2. **Individual variation**: Random variation in new observations
3. **Distance from center**: Predictions farther from $`\bar{X}`$ are less certain

#### Example: House Price Prediction

For a 2000 sq ft house:
- **Point prediction**: $`\hat{Y} = 150,000 + 75 \times 2000 = 300,000`$
- **95% PI**: $`[280,000, 320,000]`$
- We are 95% confident the house will sell for $280K-$320K

### Model Diagnostics

Model diagnostics help us verify that our assumptions are met and identify potential problems.

#### Residual Analysis

**Residual Analysis:**
1. **Normality**: Residuals should be normally distributed
2. **Independence**: Residuals should be independent
3. **Homoscedasticity**: Residual variance should be constant
4. **Linearity**: Relationship should be linear

#### Understanding Residuals

**Residuals** are the differences between observed and predicted values:
```math
e_i = Y_i - \hat{Y}_i
```

#### Diagnostic Plots

1. **Residuals vs. Fitted Values**: Check homoscedasticity and linearity
2. **Normal Q-Q Plot**: Check normality
3. **Residuals vs. Predictor**: Check for patterns
4. **Leverage Plot**: Identify influential points

#### Example: Diagnostic Interpretation

**Good Model:**
- Residuals randomly scattered around zero
- No patterns in residual plots
- Normal Q-Q plot follows straight line

**Problematic Model:**
- Residuals show patterns (curved, funnel-shaped)
- Outliers in Q-Q plot
- Heteroscedasticity (variance changes with X)

#### Influence Diagnostics

**Influence Diagnostics:**
- **Leverage**: Measures how far an observation is from the center of X
- **Cook's Distance**: Measures the influence of each observation
- **DFFITS**: Measures the influence on fitted values

#### Understanding Influence

**High Leverage Points**: Observations with unusual X values
- Can have large impact on regression line
- May not be outliers if they follow the pattern

**Influential Points**: Observations that change the regression line significantly
- High leverage + high residual
- Can distort the relationship

#### Example: Influence Analysis

For house price data:
- **High leverage**: Very large or small houses
- **Influential**: Large house with unusually low price
- **Action**: Investigate unusual observations

## Multiple Linear Regression

Multiple linear regression extends simple regression to include multiple predictor variables.

### Understanding Multiple Regression

Multiple regression allows us to:
1. **Control for confounders**: Account for multiple factors simultaneously
2. **Improve predictions**: Use more information
3. **Test specific hypotheses**: Isolate effects of individual variables
4. **Reduce bias**: Avoid omitted variable bias

#### Example: House Price Prediction

Instead of just square footage, we might include:
- **X₁**: Square footage
- **X₂**: Number of bedrooms
- **X₃**: Age of house
- **X₄**: Distance to city center
- **Y**: House price

### Mathematical Foundation

**Model Specification:**
```math
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_p X_{ip} + \epsilon_i
```

Where:
- $`Y_i`$ is the dependent variable
- $`X_{ij}`$ is the j-th predictor for observation i
- $`\beta_j`$ is the coefficient for predictor j
- $`\epsilon_i`$ is the error term

#### Matrix Notation

In matrix form:
```math
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
```

Where:
- $`\mathbf{Y}`$ is the n×1 vector of responses
- $`\mathbf{X}`$ is the n×(p+1) design matrix
- $`\boldsymbol{\beta}`$ is the (p+1)×1 coefficient vector
- $`\boldsymbol{\epsilon}`$ is the n×1 error vector

#### Least Squares Solution

The least squares solution is:
```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}
```

#### Intuitive Understanding

Each coefficient $`\beta_j`$ represents:
- The change in Y for a one-unit increase in Xⱼ
- **Holding all other variables constant**
- This is the key difference from simple regression

#### Example: Salary Prediction

Model: Salary = $`\beta_0`$ + $`\beta_1`$ × Experience + $`\beta_2`$ × Education + $`\beta_3`$ × Gender

- $`\beta_1 = 3000`$: Each year of experience adds $3,000 to salary
- $`\beta_2 = 5000`$: Each year of education adds $5,000 to salary
- **Interpretation**: Holding experience and gender constant, each year of education increases salary by $5,000

### Assumptions

**Multiple Regression Assumptions:**
1. **Linearity**: Linear relationship with each predictor
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant error variance
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Predictors are not highly correlated
6. **No endogeneity**: Predictors are not correlated with errors

#### Multicollinearity

**Multicollinearity** occurs when predictors are highly correlated:
- Makes coefficient estimates unstable
- Increases standard errors
- Makes interpretation difficult

#### Detection Methods

1. **Correlation Matrix**: Check pairwise correlations
2. **Variance Inflation Factor (VIF)**: $`VIF_j = \frac{1}{1-R_j^2}`$
3. **Condition Number**: Ratio of largest to smallest eigenvalue

#### Example: Multicollinearity

For house price model:
- **Problem**: Square footage and number of bedrooms are correlated
- **Solution**: Use square footage per bedroom instead
- **Alternative**: Ridge regression to handle multicollinearity

### Coefficient Interpretation

#### Partial Effects

Each coefficient represents the **partial effect** of that variable:
- Effect of Xⱼ on Y, holding all other variables constant
- Also called the "ceteris paribus" effect

#### Example: Education Effect

In salary model:
- **Simple regression**: Education coefficient = $8,000
- **Multiple regression**: Education coefficient = $5,000
- **Difference**: Simple regression was confounded by experience
- **Interpretation**: Education effect is $5,000 when controlling for experience

#### Standardized Coefficients

Standardized coefficients allow comparison of effect sizes:

```math
\beta_j^* = \beta_j \times \frac{s_{X_j}}{s_Y}
```

Where $`s_{X_j}`$ and $`s_Y`$ are standard deviations.

#### Example: Standardized Comparison

For house price model:
- **Square footage**: $`\beta_1^* = 0.6`$ (large effect)
- **Age**: $`\beta_2^* = -0.3`$ (medium effect)
- **Bedrooms**: $`\beta_3^* = 0.1`$ (small effect)

### Model Selection

#### Stepwise Selection

**Forward Selection:**
1. Start with no predictors
2. Add predictor with highest F-statistic
3. Continue until no significant improvement

**Backward Elimination:**
1. Start with all predictors
2. Remove predictor with lowest t-statistic
3. Continue until all remaining predictors are significant

#### Information Criteria

**Akaike Information Criterion (AIC):**
```math
AIC = 2p - 2\ln(L)
```

**Bayesian Information Criterion (BIC):**
```math
BIC = \ln(n)p - 2\ln(L)
```

Where p is the number of parameters and L is the likelihood.

#### Example: Model Comparison

Comparing three models:
- **Model 1**: R² = 0.75, AIC = 120
- **Model 2**: R² = 0.78, AIC = 125
- **Model 3**: R² = 0.80, AIC = 130

**Choice**: Model 1 (lowest AIC despite lower R²)

## Model Diagnostics

### Assumption Checking

#### Linearity

**Testing Linearity:**
1. **Residual plots**: Check for patterns
2. **Component-plus-residual plots**: Check each predictor
3. **Polynomial terms**: Test for non-linear relationships

#### Independence

**Testing Independence:**
1. **Durbin-Watson test**: For time series data
2. **Residual plots**: Check for patterns
3. **Domain knowledge**: Understand data collection

#### Homoscedasticity

**Testing Homoscedasticity:**
1. **Residual plots**: Check for funnel patterns
2. **Breusch-Pagan test**: Formal statistical test
3. **White's test**: Robust to non-normality

#### Normality

**Testing Normality:**
1. **Q-Q plots**: Check for straight line
2. **Shapiro-Wilk test**: Formal test
3. **Histogram**: Visual check

### Outlier Detection

#### Types of Outliers

**Leverage Points**: Unusual X values
- High leverage but may not be influential
- Can be detected using hat matrix

**Influential Points**: Change regression line significantly
- High leverage + high residual
- Detected using Cook's distance

**Outliers**: Unusual Y values
- Large residuals
- May or may not be influential

#### Detection Methods

**Cook's Distance:**
```math
D_i = \frac{e_i^2}{p \times \text{MSE}} \times \frac{h_{ii}}{(1-h_{ii})^2}
```

Where $`h_{ii}`$ is the leverage and $`e_i`$ is the residual.

**DFFITS:**
```math
DFFITS_i = \frac{\hat{Y}_i - \hat{Y}_{i(i)}}{\sqrt{\text{MSE}_{(i)} h_{ii}}}
```

Where $`\hat{Y}_{i(i)}`$ is the prediction without observation i.

#### Example: Outlier Analysis

For house price data:
- **Observation 15**: High leverage (very large house)
- **Observation 23**: High residual (unusually low price)
- **Observation 7**: High influence (large house, low price)
- **Action**: Investigate observation 7

## Variable Selection

### Stepwise Selection

#### Forward Selection

**Algorithm:**
1. Start with intercept only
2. Add variable with highest F-statistic
3. Continue until no significant improvement
4. Use F-to-enter criterion (e.g., p < 0.05)

#### Backward Elimination

**Algorithm:**
1. Start with all variables
2. Remove variable with lowest t-statistic
3. Continue until all remaining variables are significant
4. Use F-to-remove criterion (e.g., p > 0.10)

#### Stepwise Regression

**Algorithm:**
1. Start with forward selection
2. After each addition, check if any variables can be removed
3. Continue until no variables can be added or removed

#### Example: Variable Selection

For house price model:
- **Step 1**: Add square footage (R² = 0.75)
- **Step 2**: Add age (R² = 0.82)
- **Step 3**: Add bedrooms (R² = 0.84)
- **Step 4**: Remove age (not significant when controlling for others)
- **Final model**: Square footage + bedrooms

### Regularization

#### Ridge Regression

**Ridge Regression** adds L2 penalty:

```math
\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (Y_i - \mathbf{X}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{p} \beta_j^2
```

#### Lasso Regression

**Lasso Regression** adds L1 penalty:

```math
\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (Y_i - \mathbf{X}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{p} |\beta_j|
```

#### Elastic Net

**Elastic Net** combines both penalties:

```math
\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (Y_i - \mathbf{X}_i^T\boldsymbol{\beta})^2 + \lambda \left(\alpha \sum_{j=1}^{p} |\beta_j| + (1-\alpha) \sum_{j=1}^{p} \beta_j^2\right)
```

#### Understanding Regularization

**Benefits:**
- **Ridge**: Handles multicollinearity, shrinks coefficients
- **Lasso**: Performs variable selection, sets some coefficients to zero
- **Elastic Net**: Combines benefits of both

**Tuning Parameter λ:**
- **Large λ**: More regularization, simpler model
- **Small λ**: Less regularization, more complex model

#### Example: Regularization Comparison

For house price model:
- **OLS**: All 10 variables, R² = 0.85
- **Ridge**: All 10 variables, R² = 0.84, smaller coefficients
- **Lasso**: 6 variables, R² = 0.83, some coefficients = 0
- **Elastic Net**: 7 variables, R² = 0.84, balanced approach

## Polynomial Regression

Polynomial regression captures non-linear relationships by including polynomial terms.

### Mathematical Foundation

**Polynomial Model:**
```math
Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \cdots + \beta_p X_i^p + \epsilon_i
```

#### Example: Quadratic Regression

**Quadratic Model:**
```math
Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \epsilon_i
```

#### Understanding Polynomial Terms

**Linear term ($`\beta_1 X`$)**: Rate of change
**Quadratic term ($`\beta_2 X^2`$)**: Curvature
**Cubic term ($`\beta_3 X^3`$)**: Inflection points

#### Example: Temperature vs. Time

For daily temperature:
- **Linear**: Temperature increases linearly with time
- **Quadratic**: Temperature increases, then decreases (daily cycle)
- **Cubic**: More complex seasonal patterns

### Model Selection

#### Choosing Polynomial Degree

**Methods:**
1. **Visual inspection**: Plot data and fitted curves
2. **F-test**: Test significance of highest order term
3. **Cross-validation**: Choose degree that minimizes prediction error
4. **Information criteria**: AIC, BIC

#### Example: Degree Selection

For house price vs. square footage:
- **Linear**: R² = 0.75, AIC = 120
- **Quadratic**: R² = 0.82, AIC = 115
- **Cubic**: R² = 0.83, AIC = 118
- **Choice**: Quadratic (lowest AIC)

### Overfitting Prevention

#### Cross-Validation

**K-fold Cross-Validation:**
1. Split data into K folds
2. Train on K-1 folds, test on remaining fold
3. Repeat K times
4. Average prediction error

#### Example: CV Results

For polynomial regression:
- **Degree 1**: CV error = 0.25
- **Degree 2**: CV error = 0.18
- **Degree 3**: CV error = 0.22
- **Choice**: Degree 2 (lowest CV error)

## Logistic Regression

Logistic regression models binary outcomes using the logistic function.

### Mathematical Foundation

**Logistic Model:**
```math
P(Y_i = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip})}}
```

#### Understanding the Logistic Function

**Logistic Function:**
```math
f(x) = \frac{1}{1 + e^{-x}}
```

**Properties:**
- Output between 0 and 1
- S-shaped curve
- Symmetric around x = 0

#### Example: Credit Approval

**Model**: P(Approved) = $`\frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{Income} + \beta_2 \text{Credit Score})}}`$

**Interpretation:**
- $`\beta_1 > 0`$: Higher income increases approval probability
- $`\beta_2 > 0`$: Higher credit score increases approval probability

### Odds and Log-Odds

#### Odds Ratio

**Odds**: $`\frac{P(Y=1)}{P(Y=0)}`$

**Log-Odds**: $`\ln\left(\frac{P(Y=1)}{P(Y=0)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p`$

#### Interpretation

**Odds Ratio**: $`e^{\beta_j}`$ represents the multiplicative change in odds for a one-unit increase in Xⱼ.

#### Example: Medical Study

For treatment effect:
- **Odds ratio**: $`e^{\beta_1} = 2.5`$
- **Interpretation**: Treatment increases odds of recovery by 2.5 times
- **Alternative**: Treatment increases odds by 150%

### Model Evaluation

#### Classification Metrics

**Accuracy**: Proportion of correct predictions
**Sensitivity**: True positive rate
**Specificity**: True negative rate
**AUC-ROC**: Area under receiver operating characteristic curve

#### Example: Model Performance

For credit approval model:
- **Accuracy**: 85%
- **Sensitivity**: 80% (approve 80% of good applicants)
- **Specificity**: 90% (reject 90% of bad applicants)
- **AUC**: 0.88 (good discrimination)

## Practical Applications

### Real Estate Price Prediction

#### Data Description

**Variables:**
- **Price**: Dependent variable
- **Square footage**: Primary predictor
- **Bedrooms**: Number of bedrooms
- **Age**: House age in years
- **Location**: Distance to city center

#### Model Building

**Step 1: Data Exploration**
- Check for outliers and missing values
- Examine correlations between predictors
- Visualize relationships

**Step 2: Model Specification**
- Start with linear terms
- Add polynomial terms if needed
- Consider interactions

**Step 3: Model Fitting**
- Fit multiple models
- Compare using AIC, BIC, cross-validation
- Check assumptions

**Step 4: Model Validation**
- Residual analysis
- Outlier detection
- Prediction accuracy

#### Example Results

**Final Model:**
```math
\text{Price} = 50,000 + 75 \times \text{SqFt} + 10,000 \times \text{Bedrooms} - 1,000 \times \text{Age}
```

**Interpretation:**
- Each square foot adds $75 to price
- Each bedroom adds $10,000 to price
- Each year of age reduces price by $1,000

### Medical Research

#### Clinical Trial Analysis

**Study Design:**
- **Treatment**: New drug vs. placebo
- **Outcome**: Binary (recovery/no recovery)
- **Covariates**: Age, gender, baseline severity

**Model:**
```math
P(\text{Recovery}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{Treatment} + \beta_2 \text{Age} + \beta_3 \text{Gender})}}
```

**Results:**
- **Treatment effect**: $`e^{\beta_1} = 2.3`$ (odds ratio)
- **Age effect**: $`e^{\beta_2} = 0.95`$ (older patients less likely to recover)
- **Gender effect**: $`e^{\beta_3} = 1.1`$ (slight advantage for females)

### Business Analytics

#### Customer Churn Prediction

**Model:**
```math
P(\text{Churn}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{Usage} + \beta_2 \text{Support Calls} + \beta_3 \text{Contract Length})}}
```

**Results:**
- **Usage effect**: Higher usage reduces churn
- **Support calls**: More calls increase churn
- **Contract length**: Longer contracts reduce churn

**Business Impact:**
- Identify high-risk customers
- Target retention efforts
- Optimize pricing strategies

## Practice Problems

### Problem 1: Model Comparison

**Objective**: Create comprehensive functions for model comparison and selection.

**Tasks**:
1. Implement stepwise selection algorithms
2. Calculate and compare information criteria (AIC, BIC)
3. Perform cross-validation for model selection
4. Create standardized model comparison reports
5. Include regularization methods (Ridge, Lasso, Elastic Net)

**Example Implementation**:
```python
def model_comparison(models, data, cv_folds=5):
    """
    Compare multiple regression models using various criteria.
    
    Returns: comparison table with R², AIC, BIC, CV error
    """
    # Implementation here
```

### Problem 2: Feature Engineering

**Objective**: Implement automated feature engineering techniques.

**Tasks**:
1. Create polynomial features automatically
2. Generate interaction terms
3. Implement feature scaling and normalization
4. Add feature selection methods
5. Create feature importance analysis

### Problem 3: Cross-Validation Framework

**Objective**: Build robust cross-validation for regression models.

**Tasks**:
1. Implement k-fold cross-validation
2. Add leave-one-out cross-validation
3. Create time series cross-validation
4. Include nested cross-validation for hyperparameter tuning
5. Add cross-validation visualization

### Problem 4: Model Interpretation

**Objective**: Create functions to interpret regression coefficients and their significance.

**Tasks**:
1. Calculate and interpret standardized coefficients
2. Create coefficient confidence intervals
3. Implement partial dependence plots
4. Add variable importance measures
5. Create model explanation tools

### Problem 5: Real-World Data Analysis

**Objective**: Apply regression analysis to real datasets.

**Tasks**:
1. Choose a dataset (housing, medical, business)
2. Perform exploratory data analysis
3. Build and compare multiple models
4. Conduct thorough model diagnostics
5. Write comprehensive analysis report

## Further Reading

### Books
- **"Applied Linear Regression Models"** by Kutner, Nachtsheim, and Neter
- **"Introduction to Linear Regression Analysis"** by Montgomery, Peck, and Vining
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
- **"Regression Analysis by Example"** by Chatterjee and Hadi
- **"Generalized Linear Models"** by McCullagh and Nelder

### Online Resources
- **StatQuest**: YouTube channel with clear regression explanations
- **Khan Academy**: Linear regression and statistics courses
- **Coursera**: Machine Learning course by Andrew Ng
- **edX**: Statistical Learning course

### Advanced Topics
- **Generalized Linear Models**: Extensions beyond linear regression
- **Mixed Effects Models**: For hierarchical data structures
- **Non-linear Regression**: Complex functional forms
- **Robust Regression**: Methods resistant to outliers
- **Bayesian Regression**: Probabilistic approach to regression

## Key Takeaways

### Fundamental Concepts
- **Linear regression** models linear relationships between variables
- **Multiple regression** extends to multiple predictors with partial effects
- **Model diagnostics** are crucial for validating assumptions
- **Variable selection** helps build parsimonious models
- **Regularization** prevents overfitting and improves generalization
- **Polynomial regression** captures non-linear relationships
- **Logistic regression** handles binary classification problems
- **Cross-validation** provides reliable model performance estimates

### Mathematical Tools
- **Least squares estimation** minimizes sum of squared residuals
- **Normal equations** provide analytical solutions for parameters
- **Statistical inference** tests significance of relationships
- **Model diagnostics** verify assumptions and identify problems
- **Regularization methods** handle multicollinearity and overfitting

### Applications
- **Predictive modeling** forecasts outcomes based on predictors
- **Causal inference** isolates effects of specific variables
- **Business analytics** supports data-driven decision making
- **Medical research** evaluates treatment effectiveness
- **Quality control** monitors and optimizes processes

### Best Practices
- **Always check assumptions** before interpreting results
- **Use cross-validation** for reliable performance estimates
- **Consider multiple models** and compare systematically
- **Interpret coefficients** in context of other variables
- **Validate predictions** on new data

### Next Steps
In the following chapters, we'll build on regression foundations to explore:
- **Time Series Analysis**: Modeling temporal dependencies and trends
- **Analysis of Variance**: Comparing means across multiple groups
- **Nonparametric Methods**: When assumptions are violated
- **Advanced Topics**: Specialized methods for complex data structures

Remember that regression analysis is not just about fitting lines to data—it's about understanding relationships, making predictions, and drawing valid conclusions from data. The methods and concepts covered in this chapter provide the foundation for sophisticated data analysis and evidence-based decision making. 