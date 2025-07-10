"""
Regression Analysis Implementation

This module provides comprehensive implementations of regression analysis concepts,
including simple linear regression, multiple regression, model diagnostics, 
variable selection, polynomial regression, logistic regression, and practical applications.

Key Concepts Covered:
- Simple Linear Regression: Basic linear modeling with one predictor
- Multiple Linear Regression: Modeling with multiple predictors
- Model Diagnostics: Checking assumptions and identifying problems
- Variable Selection: Stepwise selection and regularization methods
- Polynomial Regression: Capturing non-linear relationships
- Logistic Regression: Binary classification modeling
- Practical Applications: Real-world examples and case studies

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleLinearRegression:
    """
    Simple Linear Regression Implementation
    
    Implements the mathematical foundation of simple linear regression:
    Y = β₀ + β₁X + ε
    
    Features:
    - Manual parameter estimation using normal equations
    - Statistical inference (hypothesis testing, confidence intervals)
    - Model diagnostics and assumption checking
    - Prediction intervals
    - Comprehensive model evaluation
    """
    
    def __init__(self):
        self.beta_0 = None  # Intercept
        self.beta_1 = None  # Slope
        self.r_squared = None
        self.residuals = None
        self.fitted_values = None
        self.n = None
        self.df = None  # Degrees of freedom (n-2)
        
    def fit(self, X, y):
        """
        Fit simple linear regression model using normal equations.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Independent variable
        y : array-like, shape (n_samples,)
            Dependent variable
            
        Returns:
        --------
        self : object
            Fitted model
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        self.n = len(X)
        self.df = self.n - 2
        
        # Calculate means
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope using normal equation
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        self.beta_1 = numerator / denominator
        
        # Calculate intercept
        self.beta_0 = y_mean - self.beta_1 * X_mean
        
        # Calculate fitted values and residuals
        self.fitted_values = self.beta_0 + self.beta_1 * X
        self.residuals = y - self.fitted_values
        
        # Calculate R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Independent variable values
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        return self.beta_0 + self.beta_1 * np.array(X)
    
    def get_standard_errors(self):
        """
        Calculate standard errors for coefficients.
        
        Returns:
        --------
        se_beta_0 : float
            Standard error of intercept
        se_beta_1 : float
            Standard error of slope
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        # Calculate MSE
        mse = np.sum(self.residuals ** 2) / self.df
        
        # Calculate standard error of slope
        X_mean = np.mean(X)
        ss_x = np.sum((X - X_mean) ** 2)
        se_beta_1 = np.sqrt(mse / ss_x)
        
        # Calculate standard error of intercept
        se_beta_0 = np.sqrt(mse * (1/self.n + X_mean**2 / ss_x))
        
        return se_beta_0, se_beta_1
    
    def hypothesis_test(self, alpha=0.05):
        """
        Perform hypothesis test for slope coefficient.
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Significance level
            
        Returns:
        --------
        t_stat : float
            t-statistic
        p_value : float
            p-value
        significant : bool
            Whether the relationship is significant
        """
        se_beta_0, se_beta_1 = self.get_standard_errors()
        
        # Test statistic for slope
        t_stat = self.beta_1 / se_beta_1
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.df))
        
        significant = p_value < alpha
        
        return t_stat, p_value, significant
    
    def confidence_interval(self, alpha=0.05):
        """
        Calculate confidence intervals for coefficients.
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Significance level
            
        Returns:
        --------
        ci_beta_0 : tuple
            Confidence interval for intercept
        ci_beta_1 : tuple
            Confidence interval for slope
        """
        se_beta_0, se_beta_1 = self.get_standard_errors()
        
        # Critical value
        t_crit = stats.t.ppf(1 - alpha/2, self.df)
        
        # Confidence intervals
        ci_beta_0 = (self.beta_0 - t_crit * se_beta_0, 
                     self.beta_0 + t_crit * se_beta_0)
        ci_beta_1 = (self.beta_1 - t_crit * se_beta_1, 
                     self.beta_1 + t_crit * se_beta_1)
        
        return ci_beta_0, ci_beta_1
    
    def prediction_interval(self, X_new, alpha=0.05):
        """
        Calculate prediction intervals for new observations.
        
        Parameters:
        -----------
        X_new : array-like
            New X values for prediction
        alpha : float, default=0.05
            Significance level
            
        Returns:
        --------
        predictions : array
            Point predictions
        lower_bound : array
            Lower bound of prediction intervals
        upper_bound : array
            Upper bound of prediction intervals
        """
        X_new = np.array(X_new)
        predictions = self.predict(X_new)
        
        # Calculate MSE
        mse = np.sum(self.residuals ** 2) / self.df
        
        # Critical value
        t_crit = stats.t.ppf(1 - alpha/2, self.df)
        
        # Calculate prediction interval width
        X_train = np.array(X).flatten()
        X_mean = np.mean(X_train)
        ss_x = np.sum((X_train - X_mean) ** 2)
        
        # Prediction interval standard error
        pi_se = np.sqrt(mse * (1 + 1/self.n + (X_new - X_mean)**2 / ss_x))
        
        # Prediction intervals
        lower_bound = predictions - t_crit * pi_se
        upper_bound = predictions + t_crit * pi_se
        
        return predictions, lower_bound, upper_bound
    
    def diagnostic_plots(self, X, y):
        """
        Create diagnostic plots for model validation.
        
        Parameters:
        -----------
        X : array-like
            Independent variable
        y : array-like
            Dependent variable
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Fitted Values
        axes[0, 0].scatter(self.fitted_values, self.residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        
        # 2. Normal Q-Q Plot
        stats.probplot(self.residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Residuals vs Predictor
        axes[1, 0].scatter(X, self.residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predictor')
        
        # 4. Leverage Plot
        X_array = np.array(X).flatten()
        X_mean = np.mean(X_array)
        leverage = (X_array - X_mean)**2 / np.sum((X_array - X_mean)**2)
        
        axes[1, 1].scatter(leverage, self.residuals, alpha=0.6)
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Leverage Plot')
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """
        Print comprehensive model summary.
        """
        print("=" * 50)
        print("SIMPLE LINEAR REGRESSION SUMMARY")
        print("=" * 50)
        print(f"Model: Y = {self.beta_0:.4f} + {self.beta_1:.4f} * X")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Sample size: {self.n}")
        print(f"Degrees of freedom: {self.df}")
        
        # Statistical inference
        t_stat, p_value, significant = self.hypothesis_test()
        ci_beta_0, ci_beta_1 = self.confidence_interval()
        
        print("\nSTATISTICAL INFERENCE:")
        print(f"t-statistic for slope: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant relationship: {significant}")
        print(f"95% CI for intercept: ({ci_beta_0[0]:.4f}, {ci_beta_0[1]:.4f})")
        print(f"95% CI for slope: ({ci_beta_1[0]:.4f}, {ci_beta_1[1]:.4f})")


class MultipleLinearRegression:
    """
    Multiple Linear Regression Implementation
    
    Extends simple regression to multiple predictors:
    Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
    
    Features:
    - Matrix-based parameter estimation
    - Multicollinearity detection
    - Variable selection methods
    - Model diagnostics
    - Coefficient interpretation
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.adj_r_squared = None
        self.residuals = None
        self.fitted_values = None
        self.n = None
        self.p = None
        
    def fit(self, X, y):
        """
        Fit multiple linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Independent variables
        y : array-like, shape (n_samples,)
            Dependent variable
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.n = X.shape[0]
        self.p = X.shape[1]
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(self.n), X])
        
        # Solve normal equations: β = (X'X)^(-1)X'y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        
        try:
            coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            coefficients = np.linalg.pinv(XtX) @ Xty
        
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
        
        # Calculate fitted values and residuals
        self.fitted_values = X_with_intercept @ coefficients
        self.residuals = y - self.fitted_values
        
        # Calculate R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate adjusted R-squared
        self.adj_r_squared = 1 - (ss_res / (self.n - self.p - 1)) / (ss_tot / (self.n - 1))
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        """
        X = np.array(X)
        return self.intercept + X @ self.coefficients
    
    def get_vif(self, X):
        """
        Calculate Variance Inflation Factors for multicollinearity detection.
        """
        vif_values = []
        for i in range(X.shape[1]):
            # Regress feature i on all other features
            other_features = np.delete(X, i, axis=1)
            target_feature = X[:, i]
            
            # Fit regression
            other_features_with_intercept = np.column_stack([np.ones(len(other_features)), other_features])
            try:
                coefs = np.linalg.solve(other_features_with_intercept.T @ other_features_with_intercept,
                                       other_features_with_intercept.T @ target_feature)
                fitted = other_features_with_intercept @ coefs
                residuals = target_feature - fitted
                
                # Calculate R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((target_feature - np.mean(target_feature)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
                vif_values.append(vif)
            except:
                vif_values.append(np.inf)
        
        return vif_values
    
    def stepwise_selection(self, X, y, method='forward', threshold_in=0.01, threshold_out=0.05):
        """
        Perform stepwise variable selection.
        
        Parameters:
        -----------
        X : array-like
            Independent variables
        y : array-like
            Dependent variable
        method : str, 'forward' or 'backward'
            Selection method
        threshold_in : float
            p-value threshold for adding variables
        threshold_out : float
            p-value threshold for removing variables
            
        Returns:
        --------
        selected_features : list
            Indices of selected features
        """
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        
        if method == 'forward':
            selected_features = []
            remaining_features = list(range(n_features))
            
            while remaining_features:
                best_score = -np.inf
                best_feature = None
                
                for feature in remaining_features:
                    # Add feature to current model
                    current_features = selected_features + [feature]
                    X_current = X[:, current_features]
                    
                    # Fit model
                    model = MultipleLinearRegression()
                    model.fit(X_current, y)
                    
                    # Calculate F-statistic
                    if len(selected_features) > 0:
                        X_reduced = X[:, selected_features]
                        model_reduced = MultipleLinearRegression()
                        model_reduced.fit(X_reduced, y)
                        
                        ss_res_full = np.sum(model.residuals ** 2)
                        ss_res_reduced = np.sum(model_reduced.residuals ** 2)
                        
                        if ss_res_reduced > ss_res_full:
                            f_stat = ((ss_res_reduced - ss_res_full) / 1) / (ss_res_full / (self.n - len(current_features) - 1))
                            p_value = 1 - stats.f.cdf(f_stat, 1, self.n - len(current_features) - 1)
                            
                            if p_value < threshold_in and model.adj_r_squared > best_score:
                                best_score = model.adj_r_squared
                                best_feature = feature
                    else:
                        # First feature
                        if model.adj_r_squared > best_score:
                            best_score = model.adj_r_squared
                            best_feature = feature
                
                if best_feature is not None:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                else:
                    break
        
        return selected_features


class PolynomialRegression:
    """
    Polynomial Regression Implementation
    
    Models non-linear relationships using polynomial terms:
    Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε
    
    Features:
    - Automatic polynomial feature generation
    - Degree selection using cross-validation
    - Overfitting prevention
    - Model comparison
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        Fit polynomial regression model.
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        # Create polynomial features
        X_poly = np.column_stack([X**i for i in range(1, self.degree + 1)])
        X_poly = np.column_stack([np.ones(len(X)), X_poly])
        
        # Solve normal equations
        XtX = X_poly.T @ X_poly
        Xty = X_poly.T @ y
        
        try:
            coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(XtX) @ Xty
        
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions using polynomial model.
        """
        X = np.array(X).flatten()
        X_poly = np.column_stack([X**i for i in range(1, self.degree + 1)])
        return self.intercept + X_poly @ self.coefficients
    
    @staticmethod
    def select_degree(X, y, max_degree=5, cv_folds=5):
        """
        Select optimal polynomial degree using cross-validation.
        """
        cv_scores = []
        
        for degree in range(1, max_degree + 1):
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = PolynomialRegression(degree)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                mse = mean_squared_error(y_val, y_pred)
                scores.append(mse)
            
            cv_scores.append(np.mean(scores))
        
        optimal_degree = np.argmin(cv_scores) + 1
        return optimal_degree, cv_scores


class LogisticRegressionModel:
    """
    Logistic Regression Implementation
    
    Models binary outcomes using logistic function:
    P(Y=1) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₚXₚ)))
    
    Features:
    - Maximum likelihood estimation
    - Odds ratio interpretation
    - Model evaluation metrics
    - Classification diagnostics
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def sigmoid(self, z):
        """
        Logistic function.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, max_iter=1000, learning_rate=0.01):
        """
        Fit logistic regression using gradient descent.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # Gradient descent
        for _ in range(max_iter):
            # Forward pass
            z = self.intercept + X @ self.coefficients
            predictions = self.sigmoid(z)
            
            # Gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = np.mean(predictions - y)
            
            # Update parameters
            self.coefficients -= learning_rate * dw
            self.intercept -= learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        """
        X = np.array(X)
        z = self.intercept + X @ self.coefficients
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary outcomes.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_odds_ratios(self):
        """
        Calculate odds ratios for coefficients.
        """
        return np.exp(self.coefficients)


def create_sample_data():
    """
    Create sample datasets for demonstration.
    """
    np.random.seed(42)
    
    # Simple linear regression data
    n = 100
    X_simple = np.random.uniform(0, 10, n)
    y_simple = 2 + 3 * X_simple + np.random.normal(0, 1, n)
    
    # Multiple regression data
    X1 = np.random.uniform(0, 10, n)
    X2 = np.random.uniform(0, 5, n)
    X3 = np.random.uniform(0, 3, n)
    y_multiple = 1 + 2 * X1 + 1.5 * X2 - 0.5 * X3 + np.random.normal(0, 0.5, n)
    X_multiple = np.column_stack([X1, X2, X3])
    
    # Polynomial regression data
    X_poly = np.random.uniform(-3, 3, n)
    y_poly = 2 + 3 * X_poly - 0.5 * X_poly**2 + np.random.normal(0, 0.5, n)
    
    # Logistic regression data
    X_log = np.random.normal(0, 1, n)
    log_odds = -1 + 2 * X_log
    prob = 1 / (1 + np.exp(-log_odds))
    y_log = np.random.binomial(1, prob, n)
    X_log = X_log.reshape(-1, 1)
    
    return {
        'simple': (X_simple, y_simple),
        'multiple': (X_multiple, y_multiple),
        'polynomial': (X_poly, y_poly),
        'logistic': (X_log, y_log)
    }


def demonstrate_simple_regression():
    """
    Demonstrate simple linear regression concepts.
    """
    print("=" * 60)
    print("SIMPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data()['simple']
    
    # Fit model
    model = SimpleLinearRegression()
    model.fit(X, y)
    
    # Model summary
    model.summary()
    
    # Diagnostic plots
    model.diagnostic_plots(X, y)
    
    # Prediction example
    X_new = np.array([2, 5, 8])
    predictions, lower_bound, upper_bound = model.prediction_interval(X_new)
    
    print("\nPREDICTION EXAMPLE:")
    for i, x in enumerate(X_new):
        print(f"X = {x:.1f}: Prediction = {predictions[i]:.2f} "
              f"(95% PI: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}])")


def demonstrate_multiple_regression():
    """
    Demonstrate multiple linear regression concepts.
    """
    print("\n" + "=" * 60)
    print("MULTIPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data()['multiple']
    
    # Fit model
    model = MultipleLinearRegression()
    model.fit(X, y)
    
    print(f"Model: Y = {model.intercept:.3f} + "
          f"{model.coefficients[0]:.3f}*X₁ + "
          f"{model.coefficients[1]:.3f}*X₂ + "
          f"{model.coefficients[2]:.3f}*X₃")
    print(f"R-squared: {model.r_squared:.4f}")
    print(f"Adjusted R-squared: {model.adj_r_squared:.4f}")
    
    # VIF analysis
    vif_values = model.get_vif(X)
    print("\nVARIANCE INFLATION FACTORS:")
    for i, vif in enumerate(vif_values):
        print(f"X_{i+1}: {vif:.2f}")
    
    # Stepwise selection
    selected_features = model.stepwise_selection(X, y, method='forward')
    print(f"\nSelected features (forward selection): {selected_features}")


def demonstrate_polynomial_regression():
    """
    Demonstrate polynomial regression concepts.
    """
    print("\n" + "=" * 60)
    print("POLYNOMIAL REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data()['polynomial']
    
    # Select optimal degree
    optimal_degree, cv_scores = PolynomialRegression.select_degree(X, y, max_degree=4)
    print(f"Optimal degree: {optimal_degree}")
    print("Cross-validation MSE by degree:")
    for degree, score in enumerate(cv_scores, 1):
        print(f"  Degree {degree}: {score:.4f}")
    
    # Fit optimal model
    model = PolynomialRegression(optimal_degree)
    model.fit(X, y)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    X_sorted = np.sort(X)
    y_pred = model.predict(X_sorted)
    plt.plot(X_sorted, y_pred, 'r-', linewidth=2, label=f'Polynomial (degree {optimal_degree})')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_logistic_regression():
    """
    Demonstrate logistic regression concepts.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data()['logistic']
    
    # Fit model
    model = LogisticRegressionModel()
    model.fit(X, y)
    
    print(f"Model: logit(P(Y=1)) = {model.intercept:.3f} + {model.coefficients[0]:.3f}*X")
    print(f"Odds ratio: {np.exp(model.coefficients[0]):.3f}")
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Model evaluation
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    
    # Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_regularization():
    """
    Demonstrate regularization methods.
    """
    print("\n" + "=" * 60)
    print("REGULARIZATION METHODS DEMONSTRATION")
    print("=" * 60)
    
    # Create data with multicollinearity
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = X1 + np.random.normal(0, 0.1, n)  # Highly correlated
    X3 = np.random.normal(0, 1, n)
    X = np.column_stack([X1, X2, X3])
    y = 1 + 2 * X1 + 1.5 * X2 - 0.5 * X3 + np.random.normal(0, 0.5, n)
    
    # Compare different methods
    methods = {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    for name, model in methods.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
        else:
            coefficients = model.coefficients if hasattr(model, 'coefficients') else [0, 0, 0]
        
        results[name] = {
            'MSE': mse,
            'R²': r2,
            'Coefficients': coefficients
        }
    
    # Display results
    print("MODEL COMPARISON:")
    print(f"{'Method':<15} {'MSE':<10} {'R²':<10} {'Coeff 1':<10} {'Coeff 2':<10} {'Coeff 3':<10}")
    print("-" * 70)
    for name, result in results.items():
        coeffs = result['Coefficients']
        print(f"{name:<15} {result['MSE']:<10.4f} {result['R²']:<10.4f} "
              f"{coeffs[0]:<10.4f} {coeffs[1]:<10.4f} {coeffs[2]:<10.4f}")


def demonstrate_real_world_example():
    """
    Demonstrate regression analysis with a realistic example.
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: HOUSE PRICE PREDICTION")
    print("=" * 60)
    
    # Create realistic house price data
    np.random.seed(42)
    n = 200
    
    # Generate features
    square_footage = np.random.uniform(800, 4000, n)
    bedrooms = np.random.poisson(3, n)
    age = np.random.exponential(10, n)
    distance_to_city = np.random.exponential(5, n)
    
    # Generate price with realistic relationships
    base_price = 100000
    price_per_sqft = 100
    bedroom_value = 15000
    age_penalty = 2000
    distance_penalty = 5000
    
    price = (base_price + 
             price_per_sqft * square_footage + 
             bedroom_value * bedrooms - 
             age_penalty * age - 
             distance_penalty * distance_to_city + 
             np.random.normal(0, 20000, n))
    
    # Create DataFrame
    data = pd.DataFrame({
        'price': price,
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'age': age,
        'distance_to_city': distance_to_city
    })
    
    print("DATA SUMMARY:")
    print(data.describe())
    
    # Correlation analysis
    print("\nCORRELATION MATRIX:")
    print(data.corr())
    
    # Fit multiple regression
    X = data[['square_footage', 'bedrooms', 'age', 'distance_to_city']].values
    y = data['price'].values
    
    model = MultipleLinearRegression()
    model.fit(X, y)
    
    print(f"\nREGRESSION MODEL:")
    print(f"Price = {model.intercept:.0f} + "
          f"{model.coefficients[0]:.1f}×SqFt + "
          f"{model.coefficients[1]:.0f}×Bedrooms + "
          f"{model.coefficients[2]:.0f}×Age + "
          f"{model.coefficients[3]:.0f}×Distance")
    print(f"R-squared: {model.r_squared:.4f}")
    print(f"Adjusted R-squared: {model.adj_r_squared:.4f}")
    
    # VIF analysis
    vif_values = model.get_vif(X)
    feature_names = ['Square Footage', 'Bedrooms', 'Age', 'Distance to City']
    print("\nMULTICOLLINEARITY ANALYSIS:")
    for name, vif in zip(feature_names, vif_values):
        print(f"{name}: VIF = {vif:.2f}")
    
    # Prediction example
    new_house = np.array([[2500, 4, 5, 3]])  # 2500 sqft, 4 bedrooms, 5 years old, 3 miles from city
    predicted_price = model.predict(new_house)[0]
    print(f"\nPREDICTION EXAMPLE:")
    print(f"New house: 2500 sqft, 4 bedrooms, 5 years old, 3 miles from city")
    print(f"Predicted price: ${predicted_price:,.0f}")


def main():
    """
    Main function to run all demonstrations.
    """
    print("REGRESSION ANALYSIS IMPLEMENTATION")
    print("Comprehensive demonstration of regression concepts and applications")
    print("=" * 80)
    
    # Run all demonstrations
    demonstrate_simple_regression()
    demonstrate_multiple_regression()
    demonstrate_polynomial_regression()
    demonstrate_logistic_regression()
    demonstrate_regularization()
    demonstrate_real_world_example()
    
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("✓ Simple Linear Regression with manual implementation")
    print("✓ Multiple Linear Regression with multicollinearity detection")
    print("✓ Polynomial Regression with degree selection")
    print("✓ Logistic Regression for binary classification")
    print("✓ Regularization methods (Ridge, Lasso, Elastic Net)")
    print("✓ Real-world house price prediction example")
    print("✓ Model diagnostics and assumption checking")
    print("✓ Variable selection and model comparison")
    print("✓ Statistical inference and confidence intervals")
    print("✓ Cross-validation and model evaluation")


if __name__ == "__main__":
    main() 