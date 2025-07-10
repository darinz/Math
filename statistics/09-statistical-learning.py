"""
Statistical Learning: Implementation of Core Concepts

This module implements the fundamental concepts of statistical learning including:
- Cross-validation techniques (K-fold, LOOCV, stratified)
- Bias-variance tradeoff analysis
- Model selection using information criteria
- Regularization methods (Ridge, Lasso, Elastic Net)
- Ensemble methods (Bagging, Random Forest, Boosting)
- Model evaluation metrics and techniques

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, KFold, LeaveOneOut, StratifiedKFold,
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression, make_classification, load_iris, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StatisticalLearning:
    """
    Comprehensive implementation of statistical learning concepts.
    
    This class provides methods for cross-validation, bias-variance analysis,
    model selection, regularization, ensemble methods, and model evaluation.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the StatisticalLearning class.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def demonstrate_cross_validation(self):
        """
        Demonstrate various cross-validation techniques.
        
        This method shows:
        - Holdout validation (train/test split)
        - K-fold cross-validation
        - Leave-one-out cross-validation (LOOCV)
        - Stratified cross-validation for classification
        """
        print("=" * 60)
        print("CROSS-VALIDATION TECHNIQUES")
        print("=" * 60)
        
        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.5, random_state=self.random_state)
        
        # 1. Holdout Validation (Train/Test Split)
        print("\n1. HOLDOUT VALIDATION")
        print("-" * 30)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Test R² Score: {test_score:.4f}")
        print(f"Overfitting indicator: {train_score - test_score:.4f}")
        
        # 2. K-Fold Cross-Validation
        print("\n2. K-FOLD CROSS-VALIDATION")
        print("-" * 30)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"CV Score Std: {cv_scores.std():.4f}")
        print(f"95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
              f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
        
        # 3. Leave-One-Out Cross-Validation
        print("\n3. LEAVE-ONE-OUT CROSS-VALIDATION")
        print("-" * 30)
        
        loo = LeaveOneOut()
        loo_scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
        
        print(f"LOOCV Mean Score: {loo_scores.mean():.4f}")
        print(f"LOOCV Score Std: {loo_scores.std():.4f}")
        print(f"Number of models trained: {len(loo_scores)}")
        
        # 4. Stratified Cross-Validation (for classification)
        print("\n4. STRATIFIED CROSS-VALIDATION")
        print("-" * 30)
        
        # Use classification dataset for stratified CV
        X_clf, y_clf = load_iris(return_X_y=True)
        # Use only two classes for binary classification
        mask = y_clf < 2
        X_clf = X_clf[mask]
        y_clf = y_clf[mask]
        
        clf_model = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        stratified_scores = cross_val_score(clf_model, X_clf, y_clf, cv=skf, scoring='accuracy')
        
        print(f"Stratified CV Accuracy Scores: {stratified_scores}")
        print(f"Mean Stratified CV Accuracy: {stratified_scores.mean():.4f}")
        print(f"Stratified CV Accuracy Std: {stratified_scores.std():.4f}")
        
        # Visualize CV results
        self._plot_cv_comparison(cv_scores, loo_scores, stratified_scores)
        
    def demonstrate_bias_variance_tradeoff(self):
        """
        Demonstrate the bias-variance tradeoff using polynomial regression.
        
        This method shows how model complexity affects bias and variance,
        and helps identify the optimal model complexity.
        """
        print("\n" + "=" * 60)
        print("BIAS-VARIANCE TRADEOFF ANALYSIS")
        print("=" * 60)
        
        # Generate true function with noise
        np.random.seed(self.random_state)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        true_function = 2 * X.flatten() + 1
        y = true_function + np.random.normal(0, 1, 100)
        
        # Test different polynomial degrees
        degrees = [1, 2, 3, 4, 5, 6]
        bias_squared = []
        variance = []
        total_error = []
        
        print("\nPolynomial Regression Analysis:")
        print("-" * 40)
        
        for degree in degrees:
            # Generate multiple datasets and fit models
            predictions = []
            for _ in range(50):  # 50 different datasets
                y_noisy = true_function + np.random.normal(0, 1, 100)
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y_noisy)
                pred = model.predict(X_poly)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            
            # Calculate bias², variance, and total error
            bias_sq = np.mean((mean_pred - true_function) ** 2)
            var = np.mean(np.var(predictions, axis=0))
            total_err = bias_sq + var + 1  # +1 for irreducible error
            
            bias_squared.append(bias_sq)
            variance.append(var)
            total_error.append(total_err)
            
            print(f"Degree {degree}: Bias²={bias_sq:.4f}, Variance={var:.4f}, Total Error={total_err:.4f}")
        
        # Find optimal degree
        optimal_degree = degrees[np.argmin(total_error)]
        print(f"\nOptimal polynomial degree: {optimal_degree}")
        
        # Visualize the tradeoff
        self._plot_bias_variance_tradeoff(degrees, bias_squared, variance, total_error)
        
    def demonstrate_model_selection(self):
        """
        Demonstrate model selection using information criteria and cross-validation.
        
        This method shows:
        - AIC and BIC calculations
        - Cross-validation for model selection
        - Grid search for hyperparameter tuning
        """
        print("\n" + "=" * 60)
        print("MODEL SELECTION TECHNIQUES")
        print("=" * 60)
        
        # Generate polynomial data
        np.random.seed(self.random_state)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        true_function = 2 * X.flatten() + 1
        y = true_function + np.random.normal(0, 1, 100)
        
        print("\n1. INFORMATION CRITERIA COMPARISON")
        print("-" * 40)
        
        # Test different polynomial degrees
        degrees = [1, 2, 3, 4, 5]
        aic_scores = []
        bic_scores = []
        cv_scores = []
        
        for degree in degrees:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Calculate predictions and residuals
            y_pred = model.predict(X_poly)
            residuals = y - y_pred
            n = len(y)
            k = X_poly.shape[1]  # number of parameters
            
            # Calculate log-likelihood (assuming normal residuals)
            mse = np.mean(residuals ** 2)
            log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * np.sum(residuals ** 2) / mse
            
            # Calculate AIC and BIC
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Calculate cross-validation score
            cv_score = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error').mean()
            
            aic_scores.append(aic)
            bic_scores.append(bic)
            cv_scores.append(-cv_score)  # Convert back to positive MSE
            
            print(f"Degree {degree}: AIC={aic:.2f}, BIC={bic:.2f}, CV MSE={-cv_score:.4f}")
        
        # Find best models
        best_aic_degree = degrees[np.argmin(aic_scores)]
        best_bic_degree = degrees[np.argmin(bic_scores)]
        best_cv_degree = degrees[np.argmin(cv_scores)]
        
        print(f"\nBest model by AIC: Degree {best_aic_degree}")
        print(f"Best model by BIC: Degree {best_bic_degree}")
        print(f"Best model by CV: Degree {best_cv_degree}")
        
        # 2. Grid Search for Hyperparameter Tuning
        print("\n2. GRID SEARCH HYPERPARAMETER TUNING")
        print("-" * 40)
        
        # Use SVM for classification example
        X_clf, y_clf = load_iris(return_X_y=True)
        # Use only two classes
        mask = y_clf < 2
        X_clf = X_clf[mask]
        y_clf = y_clf[mask]
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        
        svm = SVC(random_state=self.random_state)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_clf, y_clf)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        print(f"Best estimator: {grid_search.best_estimator_}")
        
        # Visualize grid search results
        self._plot_grid_search_results(grid_search)
        
    def demonstrate_regularization(self):
        """
        Demonstrate regularization techniques: Ridge, Lasso, and Elastic Net.
        
        This method shows how regularization helps prevent overfitting
        and can perform feature selection.
        """
        print("\n" + "=" * 60)
        print("REGULARIZATION TECHNIQUES")
        print("=" * 60)
        
        # Generate data with multicollinearity
        np.random.seed(self.random_state)
        n_samples, n_features = 100, 20
        
        # Create correlated features
        X = np.random.randn(n_samples, n_features)
        # Make some features correlated
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)
        X[:, 2] = X[:, 0] + 0.1 * np.random.randn(n_samples)
        
        # True coefficients (only first 5 are non-zero)
        true_coef = np.zeros(n_features)
        true_coef[:5] = [1.5, -2.0, 1.0, -1.5, 0.8]
        
        # Generate target
        y = X @ true_coef + np.random.normal(0, 0.5, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n1. LINEAR REGRESSION (NO REGULARIZATION)")
        print("-" * 40)
        
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_train_score = lr.score(X_train_scaled, y_train)
        lr_test_score = lr.score(X_test_scaled, y_test)
        
        print(f"Training R²: {lr_train_score:.4f}")
        print(f"Test R²: {lr_test_score:.4f}")
        print(f"Overfitting: {lr_train_score - lr_test_score:.4f}")
        
        print("\n2. RIDGE REGRESSION (L2 REGULARIZATION)")
        print("-" * 40)
        
        # Test different alpha values
        alphas = [0.01, 0.1, 1, 10, 100]
        ridge_scores = []
        
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            train_score = ridge.score(X_train_scaled, y_train)
            test_score = ridge.score(X_test_scaled, y_test)
            ridge_scores.append((alpha, train_score, test_score))
            
            print(f"Alpha {alpha}: Train R²={train_score:.4f}, Test R²={test_score:.4f}")
        
        print("\n3. LASSO REGRESSION (L1 REGULARIZATION)")
        print("-" * 40)
        
        lasso_scores = []
        lasso_coefs = []
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha)
            lasso.fit(X_train_scaled, y_train)
            train_score = lasso.score(X_train_scaled, y_train)
            test_score = lasso.score(X_test_scaled, y_test)
            lasso_scores.append((alpha, train_score, test_score))
            lasso_coefs.append(lasso.coef_)
            
            non_zero_coefs = np.sum(lasso.coef_ != 0)
            print(f"Alpha {alpha}: Train R²={train_score:.4f}, Test R²={test_score:.4f}, "
                  f"Non-zero coefs: {non_zero_coefs}")
        
        print("\n4. ELASTIC NET")
        print("-" * 40)
        
        # Test different alpha and l1_ratio combinations
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_net.fit(X_train_scaled, y_train)
        en_train_score = elastic_net.score(X_train_scaled, y_train)
        en_test_score = elastic_net.score(X_test_scaled, y_test)
        
        print(f"Elastic Net: Train R²={en_train_score:.4f}, Test R²={en_test_score:.4f}")
        
        # Visualize coefficient paths
        self._plot_regularization_comparison(alphas, lr.coef_, ridge_scores, lasso_coefs, true_coef)
        
    def demonstrate_ensemble_methods(self):
        """
        Demonstrate ensemble methods: Bagging, Random Forest, and Boosting.
        
        This method shows how combining multiple models can improve
        prediction accuracy and robustness.
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE METHODS")
        print("=" * 60)
        
        # Generate regression data
        X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, 
                              noise=0.5, random_state=self.random_state)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        print("\n1. BAGGING")
        print("-" * 20)
        
        # Bagging with decision trees
        bagging = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=50,
            random_state=self.random_state
        )
        bagging.fit(X_train, y_train)
        bagging_train_score = bagging.score(X_train, y_train)
        bagging_test_score = bagging.score(X_test, y_test)
        
        print(f"Bagging - Train R²: {bagging_train_score:.4f}")
        print(f"Bagging - Test R²: {bagging_test_score:.4f}")
        
        print("\n2. RANDOM FOREST")
        print("-" * 20)
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        rf.fit(X_train, y_train)
        rf_train_score = rf.score(X_train, y_train)
        rf_test_score = rf.score(X_test, y_test)
        
        print(f"Random Forest - Train R²: {rf_train_score:.4f}")
        print(f"Random Forest - Test R²: {rf_test_score:.4f}")
        
        # Feature importance
        feature_importance = rf.feature_importances_
        print(f"\nTop 5 most important features:")
        top_features = np.argsort(feature_importance)[-5:]
        for i, feature in enumerate(top_features):
            print(f"Feature {feature}: {feature_importance[feature]:.4f}")
        
        print("\n3. COMPARISON WITH SINGLE MODELS")
        print("-" * 40)
        
        # Single decision tree
        single_tree = DecisionTreeRegressor(max_depth=10, random_state=self.random_state)
        single_tree.fit(X_train, y_train)
        tree_train_score = single_tree.score(X_train, y_train)
        tree_test_score = single_tree.score(X_test, y_test)
        
        print(f"Single Tree - Train R²: {tree_train_score:.4f}")
        print(f"Single Tree - Test R²: {tree_test_score:.4f}")
        
        # Linear regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_train_score = lr.score(X_train, y_train)
        lr_test_score = lr.score(X_test, y_test)
        
        print(f"Linear Regression - Train R²: {lr_train_score:.4f}")
        print(f"Linear Regression - Test R²: {lr_test_score:.4f}")
        
        # Compare overfitting
        models = ['Linear', 'Single Tree', 'Bagging', 'Random Forest']
        train_scores = [lr_train_score, tree_train_score, bagging_train_score, rf_train_score]
        test_scores = [lr_test_score, tree_test_score, bagging_test_score, rf_test_score]
        
        print(f"\nOverfitting Analysis (Train R² - Test R²):")
        for model, train, test in zip(models, train_scores, test_scores):
            overfitting = train - test
            print(f"{model}: {overfitting:.4f}")
        
        # Visualize ensemble comparison
        self._plot_ensemble_comparison(models, train_scores, test_scores)
        
    def demonstrate_model_evaluation(self):
        """
        Demonstrate comprehensive model evaluation techniques.
        
        This method shows:
        - Classification metrics (accuracy, precision, recall, F1)
        - Regression metrics (MSE, RMSE, MAE, R²)
        - Confusion matrix visualization
        - Learning curves
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # 1. Classification Evaluation
        print("\n1. CLASSIFICATION EVALUATION")
        print("-" * 30)
        
        # Load breast cancer dataset
        X_clf, y_clf = load_breast_cancer(return_X_y=True)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.3, random_state=self.random_state
        )
        
        # Train multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state)
        }
        
        print("Classification Results:")
        print("-" * 20)
        
        for name, clf in classifiers.items():
            clf.fit(X_train_clf, y_train_clf)
            y_pred = clf.predict(X_test_clf)
            
            accuracy = accuracy_score(y_test_clf, y_pred)
            precision = precision_score(y_test_clf, y_pred, average='weighted')
            recall = recall_score(y_test_clf, y_pred, average='weighted')
            f1 = f1_score(y_test_clf, y_pred, average='weighted')
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print()
        
        # 2. Regression Evaluation
        print("2. REGRESSION EVALUATION")
        print("-" * 30)
        
        # Generate regression data
        X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.5, 
                                      random_state=self.random_state)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=self.random_state
        )
        
        # Train multiple regressors
        regressors = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        }
        
        print("Regression Results:")
        print("-" * 20)
        
        for name, reg in regressors.items():
            reg.fit(X_train_reg, y_train_reg)
            y_pred = reg.predict(X_test_reg)
            
            mse = mean_squared_error(y_test_reg, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_reg, y_pred)
            r2 = r2_score(y_test_reg, y_pred)
            
            print(f"{name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print()
        
        # 3. Confusion Matrix Visualization
        print("3. CONFUSION MATRIX")
        print("-" * 20)
        
        # Use Random Forest for confusion matrix
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf_clf.fit(X_train_clf, y_train_clf)
        y_pred_rf = rf_clf.predict(X_test_clf)
        
        cm = confusion_matrix(y_test_clf, y_pred_rf)
        print("Confusion Matrix:")
        print(cm)
        
        # Visualize confusion matrix and learning curves
        self._plot_evaluation_metrics(cm, X_train_clf, y_train_clf, rf_clf)
        
    def _plot_cv_comparison(self, cv_scores, loo_scores, stratified_scores):
        """Plot cross-validation comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # K-fold CV
        axes[0].hist(cv_scores, bins=10, alpha=0.7, color='skyblue')
        axes[0].axvline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
        axes[0].set_title('K-Fold Cross-Validation')
        axes[0].set_xlabel('R² Score')
        axes[0].legend()
        
        # LOOCV
        axes[1].hist(loo_scores, bins=20, alpha=0.7, color='lightgreen')
        axes[1].axvline(loo_scores.mean(), color='red', linestyle='--', label=f'Mean: {loo_scores.mean():.3f}')
        axes[1].set_title('Leave-One-Out CV')
        axes[1].set_xlabel('R² Score')
        axes[1].legend()
        
        # Stratified CV
        axes[2].hist(stratified_scores, bins=10, alpha=0.7, color='orange')
        axes[2].axvline(stratified_scores.mean(), color='red', linestyle='--', label=f'Mean: {stratified_scores.mean():.3f}')
        axes[2].set_title('Stratified Cross-Validation')
        axes[2].set_xlabel('Accuracy')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def _plot_bias_variance_tradeoff(self, degrees, bias_squared, variance, total_error):
        """Plot bias-variance tradeoff."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(degrees, bias_squared, 'o-', label='Bias²', linewidth=2, markersize=8)
        plt.plot(degrees, variance, 's-', label='Variance', linewidth=2, markersize=8)
        plt.plot(degrees, total_error, '^-', label='Total Error', linewidth=2, markersize=8)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error')
        plt.title('Bias-Variance Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight optimal degree
        optimal_idx = np.argmin(total_error)
        plt.axvline(degrees[optimal_idx], color='red', linestyle='--', alpha=0.7)
        plt.text(degrees[optimal_idx], plt.ylim()[1], f'Optimal: {degrees[optimal_idx]}', 
                rotation=90, verticalalignment='top')
        
        plt.subplot(2, 1, 2)
        plt.plot(degrees, bias_squared, 'o-', label='Bias²', linewidth=2, markersize=8)
        plt.plot(degrees, variance, 's-', label='Variance', linewidth=2, markersize=8)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error')
        plt.title('Bias vs Variance Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_grid_search_results(self, grid_search):
        """Plot grid search results."""
        results = grid_search.cv_results_
        
        # Extract scores and parameters
        scores = results['mean_test_score'].reshape(len(grid_search.param_grid['C']), 
                                                 len(grid_search.param_grid['gamma']))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(scores, cmap='viridis', aspect='auto')
        plt.colorbar(label='CV Accuracy')
        plt.xticks(range(len(grid_search.param_grid['gamma'])), grid_search.param_grid['gamma'])
        plt.yticks(range(len(grid_search.param_grid['C'])), grid_search.param_grid['C'])
        plt.xlabel('Gamma')
        plt.ylabel('C')
        plt.title('Grid Search Results')
        
        # Annotate best result
        best_idx = np.unravel_index(np.argmax(scores), scores.shape)
        plt.annotate(f'Best: {scores[best_idx]:.3f}', 
                    xy=(best_idx[1], best_idx[0]), 
                    xytext=(best_idx[1], best_idx[0]),
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.show()
        
    def _plot_regularization_comparison(self, alphas, lr_coef, ridge_scores, lasso_coefs, true_coef):
        """Plot regularization comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coefficient comparison
        axes[0, 0].plot(range(len(lr_coef)), lr_coef, 'o-', label='Linear Regression', linewidth=2)
        axes[0, 0].plot(range(len(true_coef)), true_coef, 's-', label='True Coefficients', linewidth=2)
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Coefficient Value')
        axes[0, 0].set_title('Linear Regression vs True Coefficients')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ridge coefficients
        for i, alpha in enumerate(alphas):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            axes[0, 1].plot(range(len(ridge.coef_)), ridge.coef_, 'o-', 
                           label=f'Alpha={alpha}', linewidth=2)
        axes[0, 1].plot(range(len(true_coef)), true_coef, 's-', label='True Coefficients', linewidth=2)
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Coefficient Value')
        axes[0, 1].set_title('Ridge Regression Coefficients')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Lasso coefficients
        for i, alpha in enumerate(alphas):
            axes[1, 0].plot(range(len(lasso_coefs[i])), lasso_coefs[i], 'o-', 
                           label=f'Alpha={alpha}', linewidth=2)
        axes[1, 0].plot(range(len(true_coef)), true_coef, 's-', label='True Coefficients', linewidth=2)
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Coefficient Value')
        axes[1, 0].set_title('Lasso Regression Coefficients')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance comparison
        ridge_train = [score[1] for score in ridge_scores]
        ridge_test = [score[2] for score in ridge_scores]
        lasso_train = [score[1] for score in lasso_scores]
        lasso_test = [score[2] for score in lasso_scores]
        
        x = range(len(alphas))
        axes[1, 1].plot(x, ridge_train, 'o-', label='Ridge Train', linewidth=2)
        axes[1, 1].plot(x, ridge_test, 's-', label='Ridge Test', linewidth=2)
        axes[1, 1].plot(x, lasso_train, '^-', label='Lasso Train', linewidth=2)
        axes[1, 1].plot(x, lasso_test, 'd-', label='Lasso Test', linewidth=2)
        axes[1, 1].set_xlabel('Alpha Index')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Regularization Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_ensemble_comparison(self, models, train_scores, test_scores):
        """Plot ensemble method comparison."""
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        overfitting = np.array(train_scores) - np.array(test_scores)
        plt.bar(x, overfitting, alpha=0.8, color='red')
        plt.xlabel('Models')
        plt.ylabel('Overfitting (Train R² - Test R²)')
        plt.title('Overfitting Analysis')
        plt.xticks(x, models, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_evaluation_metrics(self, cm, X_train, y_train, model):
        """Plot evaluation metrics including confusion matrix and learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, cv=5, 
            scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[1].plot(train_sizes_abs, train_mean, 'o-', label='Training Score', linewidth=2)
        axes[1].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        axes[1].plot(train_sizes_abs, val_mean, 's-', label='Cross-validation Score', linewidth=2)
        axes[1].fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        axes[1].set_xlabel('Training Examples')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Learning Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate all statistical learning concepts.
    
    This function runs comprehensive demonstrations of:
    - Cross-validation techniques
    - Bias-variance tradeoff
    - Model selection methods
    - Regularization techniques
    - Ensemble methods
    - Model evaluation
    """
    print("STATISTICAL LEARNING: COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the statistical learning class
    sl = StatisticalLearning(random_state=42)
    
    # Run all demonstrations
    sl.demonstrate_cross_validation()
    sl.demonstrate_bias_variance_tradeoff()
    sl.demonstrate_model_selection()
    sl.demonstrate_regularization()
    sl.demonstrate_ensemble_methods()
    sl.demonstrate_model_evaluation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Cross-validation provides reliable performance estimates")
    print("2. Bias-variance tradeoff guides model complexity choice")
    print("3. Regularization prevents overfitting")
    print("4. Ensemble methods often improve performance")
    print("5. Multiple evaluation metrics provide comprehensive assessment")
    print("6. Model selection requires balancing fit and complexity")

if __name__ == "__main__":
    main() 