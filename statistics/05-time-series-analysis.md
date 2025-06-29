# Time Series Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-blue.svg)](https://facebook.github.io/prophet/)

Time series analysis deals with data points collected over time. This chapter covers trend analysis, seasonality, forecasting models, and their applications in AI/ML.

## Table of Contents
- [Time Series Components](#time-series-components)
- [Stationarity](#stationarity)
- [Autocorrelation](#autocorrelation)
- [Moving Averages](#moving-averages)
- [ARIMA Models](#arima-models)
- [Seasonal Decomposition](#seasonal-decomposition)
- [Forecasting](#forecasting)
- [Practical Applications](#practical-applications)

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)
```

## Time Series Components

### Decomposing Time Series

```python
def generate_time_series_data(n=200):
    """Generate synthetic time series with trend, seasonality, and noise"""
    t = np.arange(n)
    
    # Trend component (linear + quadratic)
    trend = 0.1 * t + 0.001 * t**2
    
    # Seasonal component (annual seasonality)
    seasonal_period = 12
    seasonal = 5 * np.sin(2 * np.pi * t / seasonal_period) + 2 * np.cos(2 * np.pi * t / seasonal_period)
    
    # Cyclical component (longer period)
    cyclical_period = 50
    cyclical = 3 * np.sin(2 * np.pi * t / cyclical_period)
    
    # Random component
    random = np.random.normal(0, 1, n)
    
    # Combine components
    time_series = trend + seasonal + cyclical + random
    
    # Create datetime index
    dates = pd.date_range('2020-01-01', periods=n, freq='M')
    
    return pd.Series(time_series, index=dates), trend, seasonal, cyclical, random

ts_data, trend, seasonal, cyclical, random = generate_time_series_data()

print("Time Series Components")
print(f"Data shape: {ts_data.shape}")
print(f"Date range: {ts_data.index[0]} to {ts_data.index[-1]}")

# Visualize components
plt.figure(figsize=(15, 10))

# Original time series
plt.subplot(4, 1, 1)
plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=1)
plt.title('Original Time Series')
plt.ylabel('Value')

# Trend component
plt.subplot(4, 1, 2)
plt.plot(ts_data.index, trend, 'r-', linewidth=2)
plt.title('Trend Component')
plt.ylabel('Value')

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(ts_data.index, seasonal, 'g-', linewidth=2)
plt.title('Seasonal Component')
plt.ylabel('Value')

# Cyclical component
plt.subplot(4, 1, 4)
plt.plot(ts_data.index, cyclical, 'orange', linewidth=2)
plt.title('Cyclical Component')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\nComponent Statistics:")
print(f"Trend - Mean: {np.mean(trend):.3f}, Std: {np.std(trend):.3f}")
print(f"Seasonal - Mean: {np.mean(seasonal):.3f}, Std: {np.std(seasonal):.3f}")
print(f"Cyclical - Mean: {np.mean(cyclical):.3f}, Std: {np.std(cyclical):.3f}")
print(f"Random - Mean: {np.mean(random):.3f}, Std: {np.std(random):.3f}")
```

## Stationarity

### Testing for Stationarity

```python
def test_stationarity(timeseries):
    """Test stationarity using multiple methods"""
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(timeseries)
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical_values = adf_result[4]
    
    # KPSS test
    kpss_result = kpss(timeseries)
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical_values = kpss_result[3]
    
    return {
        'adf_statistic': adf_statistic,
        'adf_pvalue': adf_pvalue,
        'adf_critical_values': adf_critical_values,
        'kpss_statistic': kpss_statistic,
        'kpss_pvalue': kpss_pvalue,
        'kpss_critical_values': kpss_critical_values
    }

# Test original series
original_stationarity = test_stationarity(ts_data)

print("Stationarity Tests - Original Series")
print(f"ADF Test:")
print(f"  Statistic: {original_stationarity['adf_statistic']:.4f}")
print(f"  P-value: {original_stationarity['adf_pvalue']:.4f}")
print(f"  Critical values: {original_stationarity['adf_critical_values']}")

print(f"\nKPSS Test:")
print(f"  Statistic: {original_stationarity['kpss_statistic']:.4f}")
print(f"  P-value: {original_stationarity['kpss_pvalue']:.4f}")
print(f"  Critical values: {original_stationarity['kpss_critical_values']}")

# Test differenced series
ts_diff = ts_data.diff().dropna()
diff_stationarity = test_stationarity(ts_diff)

print(f"\nStationarity Tests - Differenced Series")
print(f"ADF Test:")
print(f"  Statistic: {diff_stationarity['adf_statistic']:.4f}")
print(f"  P-value: {diff_stationarity['adf_pvalue']:.4f}")

print(f"\nKPSS Test:")
print(f"  Statistic: {diff_stationarity['kpss_statistic']:.4f}")
print(f"  P-value: {diff_stationarity['kpss_pvalue']:.4f}")

# Visualize stationarity
plt.figure(figsize=(15, 10))

# Original series
plt.subplot(3, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=1)
plt.title('Original Time Series')
plt.ylabel('Value')

plt.subplot(3, 2, 2)
plt.hist(ts_data.values, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Original Series')
plt.xlabel('Value')
plt.ylabel('Frequency')

# First difference
plt.subplot(3, 2, 3)
plt.plot(ts_diff.index, ts_diff.values, 'g-', linewidth=1)
plt.title('First Difference')
plt.ylabel('Value')

plt.subplot(3, 2, 4)
plt.hist(ts_diff.values, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of First Difference')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Second difference
ts_diff2 = ts_diff.diff().dropna()
plt.subplot(3, 2, 5)
plt.plot(ts_diff2.index, ts_diff2.values, 'r-', linewidth=1)
plt.title('Second Difference')
plt.ylabel('Value')

plt.subplot(3, 2, 6)
plt.hist(ts_diff2.values, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Second Difference')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Autocorrelation

### ACF and PACF Analysis

```python
def analyze_autocorrelation(timeseries, max_lag=40):
    """Analyze autocorrelation and partial autocorrelation"""
    
    # Calculate ACF and PACF
    acf_values = acf(timeseries, nlags=max_lag)
    pacf_values = pacf(timeseries, nlags=max_lag)
    
    # Confidence intervals (95%)
    confidence_interval = 1.96 / np.sqrt(len(timeseries))
    
    return acf_values, pacf_values, confidence_interval

acf_vals, pacf_vals, ci = analyze_autocorrelation(ts_data)

print("Autocorrelation Analysis")
print(f"Confidence interval: ±{ci:.3f}")

# Visualize ACF and PACF
plt.figure(figsize=(15, 5))

# ACF plot
plt.subplot(1, 2, 1)
lags = np.arange(len(acf_vals))
plt.bar(lags, acf_vals, alpha=0.7, color='skyblue', edgecolor='black')
plt.axhline(ci, color='red', linestyle='--', alpha=0.7, label=f'95% CI: {ci:.3f}')
plt.axhline(-ci, color='red', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linestyle='-', alpha=0.7)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.legend()

# PACF plot
plt.subplot(1, 2, 2)
plt.bar(lags, pacf_vals, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axhline(ci, color='red', linestyle='--', alpha=0.7, label=f'95% CI: {ci:.3f}')
plt.axhline(-ci, color='red', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linestyle='-', alpha=0.7)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.legend()

plt.tight_layout()
plt.show()

# Using statsmodels built-in functions
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts_data, ax=axes[0], lags=40)
plot_pacf(ts_data, ax=axes[1], lags=40)
plt.tight_layout()
plt.show()
```

## Moving Averages

### Simple and Exponential Moving Averages

```python
def calculate_moving_averages(timeseries, windows=[5, 10, 20]):
    """Calculate different types of moving averages"""
    
    moving_averages = {}
    
    # Simple Moving Average (SMA)
    for window in windows:
        moving_averages[f'SMA_{window}'] = timeseries.rolling(window=window).mean()
    
    # Exponential Moving Average (EMA)
    for alpha in [0.1, 0.3, 0.5]:
        moving_averages[f'EMA_{alpha}'] = timeseries.ewm(alpha=alpha).mean()
    
    # Weighted Moving Average (WMA)
    for window in [5, 10]:
        weights = np.arange(1, window + 1)
        wma = timeseries.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        moving_averages[f'WMA_{window}'] = wma
    
    return moving_averages

ma_results = calculate_moving_averages(ts_data)

# Visualize moving averages
plt.figure(figsize=(15, 10))

# Original data with SMAs
plt.subplot(2, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'b-', alpha=0.7, label='Original', linewidth=1)
for window in [5, 10, 20]:
    plt.plot(ts_data.index, ma_results[f'SMA_{window}'], 
             linewidth=2, label=f'SMA {window}')
plt.title('Simple Moving Averages')
plt.ylabel('Value')
plt.legend()

# Original data with EMAs
plt.subplot(2, 2, 2)
plt.plot(ts_data.index, ts_data.values, 'b-', alpha=0.7, label='Original', linewidth=1)
for alpha in [0.1, 0.3, 0.5]:
    plt.plot(ts_data.index, ma_results[f'EMA_{alpha}'], 
             linewidth=2, label=f'EMA α={alpha}')
plt.title('Exponential Moving Averages')
plt.ylabel('Value')
plt.legend()

# Original data with WMAs
plt.subplot(2, 2, 3)
plt.plot(ts_data.index, ts_data.values, 'b-', alpha=0.7, label='Original', linewidth=1)
for window in [5, 10]:
    plt.plot(ts_data.index, ma_results[f'WMA_{window}'], 
             linewidth=2, label=f'WMA {window}')
plt.title('Weighted Moving Averages')
plt.ylabel('Value')
plt.legend()

# Comparison of different methods
plt.subplot(2, 2, 4)
plt.plot(ts_data.index, ts_data.values, 'b-', alpha=0.7, label='Original', linewidth=1)
plt.plot(ts_data.index, ma_results['SMA_10'], 'r-', linewidth=2, label='SMA 10')
plt.plot(ts_data.index, ma_results['EMA_0.3'], 'g-', linewidth=2, label='EMA 0.3')
plt.plot(ts_data.index, ma_results['WMA_10'], 'orange', linewidth=2, label='WMA 10')
plt.title('Comparison of Moving Averages')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate performance metrics
def evaluate_moving_averages(original, predictions):
    """Evaluate moving average performance"""
    # Remove NaN values for comparison
    valid_mask = ~np.isnan(predictions)
    if np.sum(valid_mask) == 0:
        return {'mse': np.nan, 'mae': np.nan, 'mape': np.nan}
    
    original_valid = original[valid_mask]
    predictions_valid = predictions[valid_mask]
    
    mse = mean_squared_error(original_valid, predictions_valid)
    mae = mean_absolute_error(original_valid, predictions_valid)
    mape = np.mean(np.abs((original_valid - predictions_valid) / original_valid)) * 100
    
    return {'mse': mse, 'mae': mae, 'mape': mape}

print("Moving Average Performance Evaluation")
for name, ma_series in ma_results.items():
    metrics = evaluate_moving_averages(ts_data.values, ma_series.values)
    print(f"{name}: MSE={metrics['mse']:.3f}, MAE={metrics['mae']:.3f}, MAPE={metrics['mape']:.2f}%")
```

## ARIMA Models

### ARIMA Model Fitting

```python
def fit_arima_models(timeseries, orders):
    """Fit multiple ARIMA models with different parameters"""
    
    models = {}
    results = {}
    
    for order in orders:
        try:
            # Fit ARIMA model
            model = ARIMA(timeseries, order=order)
            fitted_model = model.fit()
            
            models[order] = fitted_model
            results[order] = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.llf,
                'residuals': fitted_model.resid
            }
            
        except Exception as e:
            print(f"Error fitting ARIMA{order}: {e}")
            continue
    
    return models, results

# Define different ARIMA orders to try
arima_orders = [
    (1, 1, 0), (0, 1, 1), (1, 1, 1),
    (2, 1, 0), (0, 1, 2), (2, 1, 2),
    (1, 0, 1), (2, 0, 2), (1, 0, 0), (0, 0, 1)
]

arima_models, arima_results = fit_arima_models(ts_data, arima_orders)

print("ARIMA Model Comparison")
print("Order\t\tAIC\t\tBIC\t\tLog-Likelihood")
print("-" * 50)
for order, result in arima_results.items():
    print(f"{order}\t\t{result['aic']:.2f}\t\t{result['bic']:.2f}\t\t{result['log_likelihood']:.2f}")

# Find best model
best_order = min(arima_results.keys(), key=lambda x: arima_results[x]['aic'])
best_model = arima_models[best_order]

print(f"\nBest ARIMA model: {best_order}")
print(f"AIC: {arima_results[best_order]['aic']:.2f}")
print(f"BIC: {arima_results[best_order]['bic']:.2f}")

# Model diagnostics
plt.figure(figsize=(15, 10))

# Residuals
residuals = arima_results[best_order]['residuals']
plt.subplot(2, 3, 1)
plt.plot(residuals.index, residuals.values, 'b-', linewidth=1)
plt.title('Residuals')
plt.ylabel('Residual')

# Residuals histogram
plt.subplot(2, 3, 2)
plt.hist(residuals.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')

# Q-Q plot
plt.subplot(2, 3, 3)
stats.probplot(residuals.values, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

# ACF of residuals
plt.subplot(2, 3, 4)
plot_acf(residuals, ax=plt.gca(), lags=20)
plt.title('ACF of Residuals')

# PACF of residuals
plt.subplot(2, 3, 5)
plot_pacf(residuals, ax=plt.gca(), lags=20)
plt.title('PACF of Residuals')

# Residuals vs fitted
fitted_values = ts_data - residuals
plt.subplot(2, 3, 6)
plt.scatter(fitted_values, residuals, alpha=0.7, color='lightgreen')
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.tight_layout()
plt.show()
```

## Seasonal Decomposition

### STL Decomposition

```python
def perform_seasonal_decomposition(timeseries, period=12):
    """Perform seasonal decomposition using STL"""
    
    # STL decomposition
    decomposition = seasonal_decompose(timeseries, period=period, extrapolate_trend='freq')
    
    return decomposition

decomposition = perform_seasonal_decomposition(ts_data)

# Visualize decomposition
plt.figure(figsize=(15, 10))

# Original series
plt.subplot(4, 1, 1)
plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=1)
plt.title('Original Time Series')
plt.ylabel('Value')

# Trend
plt.subplot(4, 1, 2)
plt.plot(decomposition.trend.index, decomposition.trend.values, 'r-', linewidth=2)
plt.title('Trend Component')
plt.ylabel('Value')

# Seasonal
plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal.index, decomposition.seasonal.values, 'g-', linewidth=2)
plt.title('Seasonal Component')
plt.ylabel('Value')

# Residual
plt.subplot(4, 1, 4)
plt.plot(decomposition.resid.index, decomposition.resid.values, 'orange', linewidth=1)
plt.title('Residual Component')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Analyze seasonal patterns
seasonal_data = decomposition.seasonal.values
seasonal_period = 12

# Extract seasonal pattern
seasonal_pattern = []
for i in range(seasonal_period):
    pattern_values = seasonal_data[i::seasonal_period]
    seasonal_pattern.append(np.mean(pattern_values))

print("Seasonal Pattern Analysis")
print(f"Seasonal period: {seasonal_period}")
print("Seasonal pattern (monthly averages):")
for i, pattern in enumerate(seasonal_pattern):
    print(f"  Month {i+1}: {pattern:.3f}")

# Visualize seasonal pattern
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
months = range(1, seasonal_period + 1)
plt.bar(months, seasonal_pattern, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Month')
plt.ylabel('Seasonal Effect')
plt.title('Seasonal Pattern')
plt.xticks(months)

plt.subplot(1, 2, 2)
plt.plot(months, seasonal_pattern, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Month')
plt.ylabel('Seasonal Effect')
plt.title('Seasonal Pattern (Line Plot)')
plt.xticks(months)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Forecasting

### Time Series Forecasting

```python
def perform_forecasting(model, steps=24):
    """Perform forecasting using fitted ARIMA model"""
    
    # Generate forecast
    forecast = model.forecast(steps=steps)
    forecast_conf = model.get_forecast(steps=steps)
    
    # Get confidence intervals
    conf_int = forecast_conf.conf_int()
    
    return forecast, conf_int

# Perform forecasting
forecast_values, conf_intervals = perform_forecasting(best_model, steps=24)

# Create forecast index
last_date = ts_data.index[-1]
forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                              periods=24, freq='M')

print("Forecasting Results")
print(f"Forecast period: {forecast_index[0]} to {forecast_index[-1]}")
print(f"Number of forecast steps: {len(forecast_values)}")

# Visualize forecast
plt.figure(figsize=(15, 8))

# Historical data
plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=2, label='Historical Data')

# Forecast
plt.plot(forecast_index, forecast_values, 'r-', linewidth=2, label='Forecast')

# Confidence intervals
plt.fill_between(forecast_index, 
                 conf_intervals.iloc[:, 0], 
                 conf_intervals.iloc[:, 1], 
                 alpha=0.3, color='red', label='95% Confidence Interval')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Forecast evaluation metrics
def evaluate_forecast(actual, predicted):
    """Evaluate forecast performance"""
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {'mse': mse, 'mae': mae, 'mape': mape}

# For demonstration, let's create a test set
test_size = 12
train_data = ts_data[:-test_size]
test_data = ts_data[-test_size:]

# Fit model on training data
train_model = ARIMA(train_data, order=best_order).fit()

# Forecast on test period
test_forecast = train_model.forecast(steps=test_size)

# Evaluate forecast
forecast_metrics = evaluate_forecast(test_data.values, test_forecast.values)

print("Forecast Evaluation")
print(f"MSE: {forecast_metrics['mse']:.3f}")
print(f"MAE: {forecast_metrics['mae']:.3f}")
print(f"MAPE: {forecast_metrics['mape']:.2f}%")

# Visualize forecast evaluation
plt.figure(figsize=(12, 6))

plt.plot(test_data.index, test_data.values, 'b-', linewidth=2, label='Actual')
plt.plot(test_data.index, test_forecast.values, 'r-', linewidth=2, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Forecast vs Actual (Test Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Practical Applications

### Stock Price Analysis

```python
def simulate_stock_prices(n_days=252):
    """Simulate stock price data"""
    np.random.seed(42)
    
    # Generate daily returns
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
    
    # Generate price series
    initial_price = 100
    prices = [initial_price]
    
    for return_val in daily_returns:
        new_price = prices[-1] * (1 + return_val)
        prices.append(new_price)
    
    # Create datetime index
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    price_series = pd.Series(prices[1:], index=dates)
    
    return price_series

stock_prices = simulate_stock_prices()

print("Stock Price Analysis")
print(f"Data shape: {stock_prices.shape}")
print(f"Date range: {stock_prices.index[0]} to {stock_prices.index[-1]}")
print(f"Initial price: ${stock_prices.iloc[0]:.2f}")
print(f"Final price: ${stock_prices.iloc[-1]:.2f}")
print(f"Total return: {((stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1) * 100:.2f}%")

# Analyze stock prices
plt.figure(figsize=(15, 10))

# Price series
plt.subplot(2, 3, 1)
plt.plot(stock_prices.index, stock_prices.values, 'b-', linewidth=1)
plt.title('Stock Price Series')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)

# Returns
returns = stock_prices.pct_change().dropna()
plt.subplot(2, 3, 2)
plt.plot(returns.index, returns.values, 'g-', linewidth=1)
plt.title('Daily Returns')
plt.ylabel('Return')
plt.xticks(rotation=45)

# Returns distribution
plt.subplot(2, 3, 3)
plt.hist(returns.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Returns Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')

# Moving averages
sma_20 = stock_prices.rolling(window=20).mean()
sma_50 = stock_prices.rolling(window=50).mean()
plt.subplot(2, 3, 4)
plt.plot(stock_prices.index, stock_prices.values, 'b-', alpha=0.7, label='Price', linewidth=1)
plt.plot(sma_20.index, sma_20.values, 'r-', linewidth=2, label='SMA 20')
plt.plot(sma_50.index, sma_50.values, 'g-', linewidth=2, label='SMA 50')
plt.title('Price with Moving Averages')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)

# Volatility (rolling standard deviation)
volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
plt.subplot(2, 3, 5)
plt.plot(volatility.index, volatility.values, 'orange', linewidth=2)
plt.title('Rolling Volatility (20-day)')
plt.ylabel('Volatility')
plt.xticks(rotation=45)

# ACF of returns
plt.subplot(2, 3, 6)
plot_acf(returns, ax=plt.gca(), lags=20)
plt.title('ACF of Returns')

plt.tight_layout()
plt.show()

# Test for stationarity in returns
returns_stationarity = test_stationarity(returns)
print(f"\nReturns Stationarity Test:")
print(f"ADF p-value: {returns_stationarity['adf_pvalue']:.4f}")
print(f"KPSS p-value: {returns_stationarity['kpss_pvalue']:.4f}")
```

### Sales Forecasting

```python
def simulate_sales_data(n_months=60):
    """Simulate monthly sales data with trend and seasonality"""
    np.random.seed(42)
    
    # Generate time index
    months = np.arange(n_months)
    
    # Components
    trend = 1000 + 50 * months  # Linear trend
    seasonal = 200 * np.sin(2 * np.pi * months / 12) + 100 * np.cos(2 * np.pi * months / 12)
    noise = np.random.normal(0, 100, n_months)
    
    # Combine components
    sales = trend + seasonal + noise
    
    # Create datetime index
    dates = pd.date_range('2019-01-01', periods=n_months, freq='M')
    sales_series = pd.Series(sales, index=dates)
    
    return sales_series

sales_data = simulate_sales_data()

print("Sales Forecasting Analysis")
print(f"Data shape: {sales_data.shape}")
print(f"Date range: {sales_data.index[0]} to {sales_data.index[-1]}")
print(f"Average monthly sales: {sales_data.mean():.0f}")
print(f"Sales growth: {((sales_data.iloc[-1] / sales_data.iloc[0]) - 1) * 100:.1f}%")

# Analyze sales data
plt.figure(figsize=(15, 10))

# Sales series
plt.subplot(2, 3, 1)
plt.plot(sales_data.index, sales_data.values, 'b-', linewidth=1)
plt.title('Monthly Sales')
plt.ylabel('Sales')
plt.xticks(rotation=45)

# Seasonal decomposition
sales_decomposition = perform_seasonal_decomposition(sales_data)
plt.subplot(2, 3, 2)
plt.plot(sales_decomposition.seasonal.index, sales_decomposition.seasonal.values, 'g-', linewidth=2)
plt.title('Seasonal Component')
plt.ylabel('Seasonal Effect')
plt.xticks(rotation=45)

# Trend
plt.subplot(2, 3, 3)
plt.plot(sales_decomposition.trend.index, sales_decomposition.trend.values, 'r-', linewidth=2)
plt.title('Trend Component')
plt.ylabel('Trend')
plt.xticks(rotation=45)

# Year-over-year growth
yoy_growth = sales_data.pct_change(periods=12) * 100
plt.subplot(2, 3, 4)
plt.plot(yoy_growth.index, yoy_growth.values, 'purple', linewidth=2)
plt.title('Year-over-Year Growth')
plt.ylabel('Growth (%)')
plt.xticks(rotation=45)

# Monthly averages
monthly_avg = sales_data.groupby(sales_data.index.month).mean()
plt.subplot(2, 3, 5)
plt.bar(monthly_avg.index, monthly_avg.values, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(1, 13))

# ACF
plt.subplot(2, 3, 6)
plot_acf(sales_data, ax=plt.gca(), lags=24)
plt.title('ACF of Sales')

plt.tight_layout()
plt.show()

# Fit ARIMA model for sales forecasting
sales_arima = ARIMA(sales_data, order=(1, 1, 1)).fit()
sales_forecast, sales_conf = perform_forecasting(sales_arima, steps=12)

print(f"\nSales Forecast (Next 12 Months):")
forecast_df = pd.DataFrame({
    'Forecast': sales_forecast.values,
    'Lower_CI': sales_conf.iloc[:, 0],
    'Upper_CI': sales_conf.iloc[:, 1]
}, index=sales_forecast.index)

print(forecast_df.round(0))

# Visualize sales forecast
plt.figure(figsize=(12, 6))

plt.plot(sales_data.index, sales_data.values, 'b-', linewidth=2, label='Historical Sales')
plt.plot(sales_forecast.index, sales_forecast.values, 'r-', linewidth=2, label='Forecast')
plt.fill_between(sales_forecast.index, 
                 sales_conf.iloc[:, 0], 
                 sales_conf.iloc[:, 1], 
                 alpha=0.3, color='red', label='95% Confidence Interval')

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.show()
```

## Practice Problems

1. **Trend Analysis**: Create functions to detect and analyze different types of trends (linear, exponential, polynomial).

2. **Seasonality Detection**: Implement methods to automatically detect seasonal patterns and their periods.

3. **Forecast Evaluation**: Build comprehensive forecast evaluation frameworks with multiple metrics.

4. **Anomaly Detection**: Develop time series anomaly detection methods using statistical and machine learning approaches.

## Further Reading

- "Time Series Analysis: Forecasting and Control" by Box, Jenkins, Reinsel, and Ljung
- "Forecasting: Principles and Practice" by Rob J. Hyndman and George Athanasopoulos
- "Time Series Analysis: Univariate and Multivariate Methods" by William W.S. Wei
- "Applied Time Series Analysis" by Wayne A. Woodward, Henry L. Gray, and Alan C. Elliott

## Key Takeaways

- **Time series components** include trend, seasonality, cyclical, and random components
- **Stationarity** is crucial for many time series models and can be tested using ADF and KPSS tests
- **Autocorrelation** analysis helps identify patterns and guide model selection
- **Moving averages** provide smoothing and trend identification
- **ARIMA models** are powerful for forecasting stationary time series
- **Seasonal decomposition** separates time series into interpretable components
- **Forecasting** requires careful model selection and evaluation
- **Real-world applications** include stock prices, sales forecasting, and economic indicators

In the next chapter, we'll explore multivariate statistics, including principal component analysis, factor analysis, and clustering techniques. 