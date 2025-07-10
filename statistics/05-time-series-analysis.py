"""
Time Series Analysis Implementation

This module provides comprehensive implementations of time series analysis concepts,
including decomposition, stationarity, autocorrelation, moving averages, ARIMA models,
seasonal decomposition, forecasting, and practical applications.

Key Concepts Covered:
- Time Series Components: Trend, seasonality, cyclical, and random components
- Stationarity: Testing and achieving stationarity
- Autocorrelation: ACF and PACF analysis
- Moving Averages: SMA, EMA, and WMA
- ARIMA Models: AutoRegressive Integrated Moving Average models
- Seasonal Decomposition: STL and classical decomposition
- Forecasting: Multiple forecasting methods and evaluation
- Practical Applications: Stock prices, sales forecasting, economic indicators

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import detrend
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TimeSeriesComponents:
    """
    Time Series Components Analysis
    
    Implements decomposition of time series into trend, seasonal, cyclical, and random components.
    Supports both additive and multiplicative models.
    """
    
    def __init__(self, method='additive'):
        self.method = method
        self.trend = None
        self.seasonal = None
        self.residual = None
        
    def decompose(self, data, period=None):
        """
        Decompose time series into components.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        period : int, optional
            Seasonal period (e.g., 12 for monthly data)
            
        Returns:
        --------
        self : object
            Decomposed components
        """
        if period is None:
            # Try to detect period automatically
            period = self._detect_period(data)
        
        # Use statsmodels seasonal decomposition
        decomposition = seasonal_decompose(data, period=period, extrapolate_trend='freq')
        
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        
        return self
    
    def _detect_period(self, data):
        """
        Detect seasonal period using autocorrelation.
        """
        acf_values = acf(data, nlags=len(data)//2)
        # Find the first peak after lag 1
        peaks = []
        for i in range(2, len(acf_values)):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                peaks.append(i)
        
        if peaks:
            return peaks[0]
        else:
            return 12  # Default to monthly
    
    def plot_components(self, data, title="Time Series Decomposition"):
        """
        Plot the decomposed components.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original data
        axes[0].plot(data, label='Original')
        axes[0].set_title('Original Time Series')
        axes[0].legend()
        
        # Trend
        axes[1].plot(self.trend, label='Trend', color='red')
        axes[1].set_title('Trend Component')
        axes[1].legend()
        
        # Seasonal
        axes[2].plot(self.seasonal, label='Seasonal', color='green')
        axes[2].set_title('Seasonal Component')
        axes[2].legend()
        
        # Residual
        axes[3].plot(self.residual, label='Residual', color='orange')
        axes[3].set_title('Residual Component')
        axes[3].legend()
        
        plt.tight_layout()
        plt.show()


class StationarityTests:
    """
    Stationarity Testing and Transformation
    
    Implements Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity,
    along with differencing and transformation methods.
    """
    
    def __init__(self):
        self.adf_result = None
        self.kpss_result = None
        
    def test_stationarity(self, data, alpha=0.05):
        """
        Perform ADF and KPSS tests for stationarity.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        # ADF Test
        adf_stat, adf_pvalue, adf_critical = adfuller(data, regression='ct')
        
        # KPSS Test
        kpss_stat, kpss_pvalue, kpss_critical = kpss(data, regression='ct')
        
        self.adf_result = {
            'statistic': adf_stat,
            'pvalue': adf_pvalue,
            'critical_values': adf_critical,
            'is_stationary': adf_pvalue < alpha
        }
        
        self.kpss_result = {
            'statistic': kpss_stat,
            'pvalue': kpss_pvalue,
            'critical_values': kpss_critical,
            'is_stationary': kpss_pvalue > alpha
        }
        
        return {
            'adf': self.adf_result,
            'kpss': self.kpss_result,
            'conclusion': self._interpret_results()
        }
    
    def _interpret_results(self):
        """
        Interpret ADF and KPSS test results.
        """
        adf_stationary = self.adf_result['is_stationary']
        kpss_stationary = self.kpss_result['is_stationary']
        
        if adf_stationary and kpss_stationary:
            return "Series is stationary"
        elif not adf_stationary and not kpss_stationary:
            return "Series is non-stationary"
        else:
            return "Inconclusive - may need differencing"
    
    def difference(self, data, order=1):
        """
        Apply differencing to make series stationary.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        order : int
            Order of differencing
            
        Returns:
        --------
        array : Differenced series
        """
        diff_data = data.copy()
        for _ in range(order):
            diff_data = np.diff(diff_data)
        
        return diff_data
    
    def seasonal_difference(self, data, period):
        """
        Apply seasonal differencing.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        period : int
            Seasonal period
            
        Returns:
        --------
        array : Seasonally differenced series
        """
        return data[period:] - data[:-period]


class AutocorrelationAnalysis:
    """
    Autocorrelation Analysis
    
    Implements ACF and PACF analysis for time series modeling.
    """
    
    def __init__(self):
        self.acf_values = None
        self.pacf_values = None
        self.acf_confint = None
        self.pacf_confint = None
        
    def compute_acf_pacf(self, data, nlags=40, alpha=0.05):
        """
        Compute ACF and PACF values.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        nlags : int
            Number of lags to compute
        alpha : float
            Significance level for confidence intervals
            
        Returns:
        --------
        dict : ACF and PACF results
        """
        # Compute ACF
        self.acf_values, self.acf_confint = acf(data, nlags=nlags, alpha=alpha, fft=True)
        
        # Compute PACF
        self.pacf_values, self.pacf_confint = pacf(data, nlags=nlags, alpha=alpha)
        
        return {
            'acf': self.acf_values,
            'pacf': self.pacf_values,
            'acf_confint': self.acf_confint,
            'pacf_confint': self.pacf_confint
        }
    
    def plot_acf_pacf(self, data, nlags=40, alpha=0.05):
        """
        Plot ACF and PACF.
        """
        results = self.compute_acf_pacf(data, nlags, alpha)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF Plot
        lags = range(len(results['acf']))
        axes[0].stem(lags, results['acf'])
        axes[0].axhline(y=0, linestyle='--', color='black')
        axes[0].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='red', alpha=0.5)
        axes[0].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='red', alpha=0.5)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
        
        # PACF Plot
        axes[1].stem(lags, results['pacf'])
        axes[1].axhline(y=0, linestyle='--', color='black')
        axes[1].axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='red', alpha=0.5)
        axes[1].axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='red', alpha=0.5)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
        
        plt.tight_layout()
        plt.show()


class MovingAverages:
    """
    Moving Averages Implementation
    
    Implements Simple Moving Average (SMA), Exponential Moving Average (EMA),
    and Weighted Moving Average (WMA).
    """
    
    def __init__(self):
        self.sma_values = None
        self.ema_values = None
        self.wma_values = None
        
    def simple_moving_average(self, data, window):
        """
        Compute Simple Moving Average.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        window : int
            Window size
            
        Returns:
        --------
        array : SMA values
        """
        self.sma_values = pd.Series(data).rolling(window=window).mean().values
        return self.sma_values
    
    def exponential_moving_average(self, data, alpha):
        """
        Compute Exponential Moving Average.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        alpha : float
            Smoothing parameter (0 < alpha <= 1)
            
        Returns:
        --------
        array : EMA values
        """
        self.ema_values = pd.Series(data).ewm(alpha=alpha).mean().values
        return self.ema_values
    
    def weighted_moving_average(self, data, weights):
        """
        Compute Weighted Moving Average.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        weights : array-like
            Weights for the moving average
            
        Returns:
        --------
        array : WMA values
        """
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        self.wma_values = pd.Series(data).rolling(window=len(weights)).apply(
            lambda x: np.dot(x, weights), raw=True
        ).values
        
        return self.wma_values
    
    def plot_moving_averages(self, data, window=20, alpha=0.2, weights=None):
        """
        Plot different types of moving averages.
        """
        if weights is None:
            weights = [0.5, 0.3, 0.2]  # Default weights
        
        sma = self.simple_moving_average(data, window)
        ema = self.exponential_moving_average(data, alpha)
        wma = self.weighted_moving_average(data, weights)
        
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Original Data', alpha=0.7)
        plt.plot(sma, label=f'SMA (window={window})', linewidth=2)
        plt.plot(ema, label=f'EMA (α={alpha})', linewidth=2)
        plt.plot(wma, label=f'WMA (weights={weights})', linewidth=2)
        plt.title('Moving Averages Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class ARIMAModel:
    """
    ARIMA Model Implementation
    
    Implements AutoRegressive Integrated Moving Average models for time series forecasting.
    """
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, data):
        """
        Fit ARIMA model to data.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        self : object
            Fitted model
        """
        self.fitted_model = ARIMA(data, order=self.order).fit()
        return self
    
    def forecast(self, steps=10):
        """
        Generate forecasts.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        array : Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_summary(self):
        """
        Get model summary.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting summary")
        
        return self.fitted_model.summary()
    
    def diagnostic_plots(self):
        """
        Create diagnostic plots for the fitted model.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before diagnostics")
        
        self.fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.show()


class ForecastingMethods:
    """
    Multiple Forecasting Methods
    
    Implements various forecasting methods including naive, seasonal naive,
    moving average, and exponential smoothing.
    """
    
    def __init__(self):
        self.forecasts = {}
        
    def naive_forecast(self, data, steps=1):
        """
        Naive forecast: next value equals last value.
        """
        forecast = np.full(steps, data[-1])
        self.forecasts['naive'] = forecast
        return forecast
    
    def seasonal_naive_forecast(self, data, period, steps=1):
        """
        Seasonal naive forecast: next value equals value from same period last cycle.
        """
        forecast = []
        for i in range(steps):
            forecast.append(data[-(period - (i % period))])
        
        self.forecasts['seasonal_naive'] = np.array(forecast)
        return np.array(forecast)
    
    def moving_average_forecast(self, data, window, steps=1):
        """
        Moving average forecast.
        """
        ma_value = np.mean(data[-window:])
        forecast = np.full(steps, ma_value)
        self.forecasts['moving_average'] = forecast
        return forecast
    
    def exponential_smoothing_forecast(self, data, alpha, steps=1):
        """
        Exponential smoothing forecast.
        """
        # Simple exponential smoothing
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
        
        forecast = np.full(steps, smoothed[-1])
        self.forecasts['exponential_smoothing'] = forecast
        return forecast
    
    def evaluate_forecasts(self, actual, predicted):
        """
        Evaluate forecast accuracy.
        
        Parameters:
        -----------
        actual : array-like
            Actual values
        predicted : array-like
            Predicted values
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }


def create_sample_time_series():
    """
    Create sample time series data for demonstration.
    """
    np.random.seed(42)
    
    # Generate time index
    n = 200
    time = np.arange(n)
    
    # Create components
    trend = 0.1 * time + 10  # Linear trend
    seasonal = 5 * np.sin(2 * np.pi * time / 12)  # Annual seasonality
    noise = np.random.normal(0, 1, n)  # Random noise
    
    # Combine components
    data = trend + seasonal + noise
    
    return data, time


def demonstrate_time_series_analysis():
    """
    Demonstrate comprehensive time series analysis.
    """
    print("=" * 60)
    print("TIME SERIES ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    data, time = create_sample_time_series()
    
    print("1. TIME SERIES COMPONENTS ANALYSIS")
    print("-" * 40)
    
    # Decompose time series
    decomposer = TimeSeriesComponents()
    decomposer.decompose(data, period=12)
    decomposer.plot_components(data)
    
    print("2. STATIONARITY TESTING")
    print("-" * 40)
    
    # Test stationarity
    stationarity_tester = StationarityTests()
    results = stationarity_tester.test_stationarity(data)
    
    print(f"ADF Test:")
    print(f"  Statistic: {results['adf']['statistic']:.4f}")
    print(f"  p-value: {results['adf']['pvalue']:.4f}")
    print(f"  Stationary: {results['adf']['is_stationary']}")
    
    print(f"\nKPSS Test:")
    print(f"  Statistic: {results['kpss']['statistic']:.4f}")
    print(f"  p-value: {results['kpss']['pvalue']:.4f}")
    print(f"  Stationary: {results['kpss']['is_stationary']}")
    
    print(f"\nConclusion: {results['conclusion']}")
    
    # Apply differencing if needed
    if not results['adf']['is_stationary']:
        print("\nApplying first differencing...")
        diff_data = stationarity_tester.difference(data)
        diff_results = stationarity_tester.test_stationarity(diff_data)
        print(f"Differenced series stationary: {diff_results['adf']['is_stationary']}")
    
    print("\n3. AUTOCORRELATION ANALYSIS")
    print("-" * 40)
    
    # ACF and PACF analysis
    acf_analyzer = AutocorrelationAnalysis()
    acf_analyzer.plot_acf_pacf(data)
    
    print("4. MOVING AVERAGES")
    print("-" * 40)
    
    # Moving averages
    ma_analyzer = MovingAverages()
    ma_analyzer.plot_moving_averages(data)
    
    print("5. ARIMA MODELING")
    print("-" * 40)
    
    # Fit ARIMA model
    arima_model = ARIMAModel(order=(1, 1, 1))
    arima_model.fit(data)
    
    print("ARIMA Model Summary:")
    print(arima_model.get_summary())
    
    # Generate forecasts
    forecast = arima_model.forecast(steps=10)
    print(f"\nARIMA Forecast (next 10 steps):")
    print(forecast)
    
    print("\n6. FORECASTING METHODS COMPARISON")
    print("-" * 40)
    
    # Compare different forecasting methods
    forecaster = ForecastingMethods()
    
    # Split data for evaluation
    train_data = data[:-20]
    test_data = data[-20:]
    
    # Generate forecasts
    naive_forecast = forecaster.naive_forecast(train_data, steps=20)
    seasonal_forecast = forecaster.seasonal_naive_forecast(train_data, period=12, steps=20)
    ma_forecast = forecaster.moving_average_forecast(train_data, window=10, steps=20)
    es_forecast = forecaster.exponential_smoothing_forecast(train_data, alpha=0.3, steps=20)
    
    # Evaluate forecasts
    methods = ['naive', 'seasonal_naive', 'moving_average', 'exponential_smoothing']
    forecasts = [naive_forecast, seasonal_forecast, ma_forecast, es_forecast]
    
    print("Forecast Evaluation:")
    for method, forecast in zip(methods, forecasts):
        metrics = forecaster.evaluate_forecasts(test_data, forecast)
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label='Original Data', alpha=0.7)
    plt.plot(time[-20:], test_data, label='Test Data', linewidth=2)
    
    colors = ['red', 'green', 'blue', 'orange']
    for method, forecast, color in zip(methods, forecasts, colors):
        plt.plot(time[-20:], forecast, label=f'{method.replace("_", " ").title()}', 
                color=color, linestyle='--')
    
    plt.title('Forecasting Methods Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_real_world_example():
    """
    Demonstrate time series analysis with a realistic example.
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: STOCK PRICE ANALYSIS")
    print("=" * 60)
    
    # Create realistic stock price data
    np.random.seed(42)
    n = 252  # Trading days in a year
    
    # Generate stock price with trend, volatility, and mean reversion
    returns = np.random.normal(0.0005, 0.02, n)  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))  # Stock price
    
    # Add some trend and seasonality
    trend = np.linspace(0, 0.1, n)  # Upward trend
    seasonal = 0.02 * np.sin(2 * np.pi * np.arange(n) / 21)  # Monthly seasonality
    
    price = price * (1 + trend + seasonal)
    
    print("STOCK PRICE ANALYSIS:")
    print(f"Initial price: ${price[0]:.2f}")
    print(f"Final price: ${price[-1]:.2f}")
    print(f"Total return: {((price[-1] / price[0]) - 1) * 100:.2f}%")
    
    # Technical analysis with moving averages
    ma_analyzer = MovingAverages()
    
    # Short-term and long-term moving averages
    sma_20 = ma_analyzer.simple_moving_average(price, 20)
    sma_50 = ma_analyzer.simple_moving_average(price, 50)
    
    # Plot with moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(price, label='Stock Price', alpha=0.7)
    plt.plot(sma_20, label='20-day SMA', linewidth=2)
    plt.plot(sma_50, label='50-day SMA', linewidth=2)
    
    # Identify golden cross and death cross
    golden_cross = np.where(sma_20 > sma_50)[0]
    death_cross = np.where(sma_20 < sma_50)[0]
    
    if len(golden_cross) > 0:
        plt.scatter(golden_cross, price[golden_cross], color='green', 
                   s=50, label='Golden Cross', zorder=5)
    if len(death_cross) > 0:
        plt.scatter(death_cross, price[death_cross], color='red', 
                   s=50, label='Death Cross', zorder=5)
    
    plt.title('Stock Price with Moving Averages')
    plt.xlabel('Trading Day')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Volatility analysis
    returns = np.diff(np.log(price))
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    print(f"\nVOLATILITY ANALYSIS:")
    print(f"Annualized volatility: {volatility:.2%}")
    
    # Stationarity test on returns
    stationarity_tester = StationarityTests()
    returns_results = stationarity_tester.test_stationarity(returns)
    print(f"Returns stationary: {returns_results['adf']['is_stationary']}")
    
    # ACF analysis of returns
    acf_analyzer = AutocorrelationAnalysis()
    acf_analyzer.plot_acf_pacf(returns, nlags=20)
    
    print("\nTECHNICAL INDICATORS:")
    
    # Support and resistance levels
    support = np.percentile(price, 25)
    resistance = np.percentile(price, 75)
    
    print(f"Support level (25th percentile): ${support:.2f}")
    print(f"Resistance level (75th percentile): ${resistance:.2f}")
    print(f"Current price: ${price[-1]:.2f}")
    
    if price[-1] < support:
        print("Price below support - potential buy signal")
    elif price[-1] > resistance:
        print("Price above resistance - potential sell signal")
    else:
        print("Price within normal range")


def main():
    """
    Main function to run all demonstrations.
    """
    print("TIME SERIES ANALYSIS IMPLEMENTATION")
    print("Comprehensive demonstration of time series concepts and applications")
    print("=" * 80)
    
    # Run demonstrations
    demonstrate_time_series_analysis()
    demonstrate_real_world_example()
    
    print("\n" + "=" * 80)
    print("TIME SERIES ANALYSIS DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Concepts Demonstrated:")
    print("✓ Time series decomposition (trend, seasonal, residual)")
    print("✓ Stationarity testing (ADF, KPSS)")
    print("✓ Autocorrelation analysis (ACF, PACF)")
    print("✓ Moving averages (SMA, EMA, WMA)")
    print("✓ ARIMA modeling and forecasting")
    print("✓ Multiple forecasting methods comparison")
    print("✓ Real-world stock price analysis")
    print("✓ Technical analysis indicators")
    print("✓ Forecast evaluation metrics")
    print("✓ Model diagnostics and validation")


if __name__ == "__main__":
    main() 