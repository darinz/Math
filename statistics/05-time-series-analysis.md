# Time Series Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13+-blue.svg)](https://www.statsmodels.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-blue.svg)](https://facebook.github.io/prophet/)

## Introduction

Time series analysis deals with data points collected over time. This chapter covers trend analysis, seasonality, forecasting models, and their applications in AI/ML.

### Why Time Series Analysis Matters

Time series analysis is essential for understanding temporal patterns and making predictions about future values. It's crucial for:

1. **Business Forecasting**: Sales, demand, and revenue predictions
2. **Financial Analysis**: Stock prices, exchange rates, and market trends
3. **Economic Modeling**: GDP, inflation, and unemployment rates
4. **Scientific Research**: Climate data, population growth, and medical monitoring
5. **Quality Control**: Manufacturing processes and system monitoring

### Characteristics of Time Series Data

Time series data has unique properties that distinguish it from cross-sectional data:

1. **Temporal Ordering**: Observations are ordered by time
2. **Dependencies**: Current values depend on past values
3. **Trends**: Long-term systematic changes
4. **Seasonality**: Regular periodic patterns
5. **Noise**: Random fluctuations and measurement error

### Types of Time Series

1. **Continuous**: Stock prices, temperature readings
2. **Discrete**: Daily sales counts, monthly unemployment rates
3. **Regular**: Observations at fixed intervals (hourly, daily, monthly)
4. **Irregular**: Observations at variable intervals

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

The examples in this chapter use Python libraries for time series analysis. We'll work with both theoretical concepts and practical implementations to build intuition and computational skills.

## Time Series Components

Time series data can be decomposed into several fundamental components that help us understand the underlying patterns and structure.

### Understanding Time Series Decomposition

Think of a time series as a complex signal that can be broken down into simpler, interpretable parts. Just as a musical chord can be decomposed into individual notes, a time series can be decomposed into trend, seasonality, cycles, and noise.

#### Intuitive Example: Retail Sales

Consider monthly retail sales data:
- **Trend**: Overall growth in sales over years
- **Seasonality**: Higher sales in December (holiday season)
- **Cycles**: Economic boom/bust cycles affecting consumer spending
- **Random**: Unpredictable events (weather, promotions, etc.)

### Mathematical Decomposition

The **classical decomposition** model represents a time series as:

```math
Y_t = T_t + S_t + C_t + R_t
```

Where:
- $`Y_t`$ = Observed value at time t
- $`T_t`$ = Trend component (long-term movement)
- $`S_t`$ = Seasonal component (periodic patterns)
- $`C_t`$ = Cyclical component (irregular cycles)
- $`R_t`$ = Random/Residual component (unexplained variation)

#### Additive vs. Multiplicative Models

**Additive Model:**
```math
Y_t = T_t + S_t + C_t + R_t
```

**Multiplicative Model:**
```math
Y_t = T_t \times S_t \times C_t \times R_t
```

**Log-Additive Model:**
```math
\log(Y_t) = \log(T_t) + \log(S_t) + \log(C_t) + \log(R_t)
```

#### Choosing the Right Model

**Additive Model**: When seasonal variations are constant in magnitude
- **Example**: Temperature data (same degree variation each season)

**Multiplicative Model**: When seasonal variations are proportional to trend
- **Example**: Sales data (percentage changes with growth)

**Log-Additive**: Transforms multiplicative model to additive form
- **Example**: Economic data with exponential growth

### Trend Component ($`T_t`$)

The trend represents the long-term systematic change in the series.

#### Understanding Trends

Trends capture the underlying direction of the time series, ignoring short-term fluctuations. They can be:
- **Upward**: Growing population, increasing sales
- **Downward**: Declining manufacturing, decreasing crime rates
- **Flat**: Stable processes, mature markets

#### Mathematical Properties

1. **Monotonicity**: Trend should be smooth and systematic
2. **Persistence**: Changes should be gradual, not abrupt
3. **Global Nature**: Trend affects the entire series

#### Common Trend Models

**Linear Trend:**
```math
T_t = \beta_0 + \beta_1 t + \epsilon_t
```

**Quadratic Trend:**
```math
T_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \epsilon_t
```

**Exponential Trend:**
```math
T_t = \beta_0 e^{\beta_1 t} + \epsilon_t
```

**Logistic Trend:**
```math
T_t = \frac{L}{1 + e^{-k(t-t_0)}} + \epsilon_t
```

Where:
- $`L`$ = maximum level (carrying capacity)
- $`k`$ = growth rate
- $`t_0`$ = inflection point

#### Example: Population Growth

**Linear**: Population grows by constant amount each year
**Exponential**: Population grows by constant percentage each year
**Logistic**: Population grows rapidly, then levels off (S-curve)

#### Trend Estimation Methods

**1. Moving Average:**
```math
\hat{T}_t = \frac{1}{2k+1} \sum_{i=-k}^{k} Y_{t+i}
```

**2. Exponential Smoothing:**
```math
\hat{T}_t = \alpha Y_t + (1-\alpha) \hat{T}_{t-1}
```

**3. Linear Regression:**
```math
\hat{T}_t = \hat{\beta}_0 + \hat{\beta}_1 t
```

**4. Polynomial Regression:**
```math
\hat{T}_t = \hat{\beta}_0 + \hat{\beta}_1 t + \hat{\beta}_2 t^2 + \cdots + \hat{\beta}_p t^p
```

#### Example: Moving Average Calculation

For monthly sales data with k=2:
- **January**: $`\hat{T}_{Jan} = \frac{Y_{Nov} + Y_{Dec} + Y_{Jan} + Y_{Feb} + Y_{Mar}}{5}`$
- **February**: $`\hat{T}_{Feb} = \frac{Y_{Dec} + Y_{Jan} + Y_{Feb} + Y_{Mar} + Y_{Apr}}{5}`$

### Seasonal Component (S_t)

Seasonality represents regular, periodic patterns that repeat at fixed intervals.

#### Understanding Seasonality

Seasonality captures predictable patterns that repeat regularly:
- **Daily**: Traffic patterns, electricity usage
- **Weekly**: Restaurant sales (weekend vs. weekday)
- **Monthly**: Payroll cycles, utility bills
- **Quarterly**: Business reporting cycles
- **Yearly**: Weather patterns, holiday effects

#### Mathematical Properties

1. **Periodicity**: $`S_t = S_{t+s}`$ where s is the seasonal period
2. **Zero Sum**: $`\sum_{i=1}^{s} S_i = 0`$ (additive model)
3. **Product Unity**: $`\prod_{i=1}^{s} S_i = 1`$ (multiplicative model)

#### Example: Monthly Seasonality

For monthly data (s=12):
- **January effect**: $`S_1`$ (same every January)
- **February effect**: $`S_2`$ (same every February)
- **...**
- **December effect**: $`S_{12}`$ (same every December)

#### Seasonal Models

**Deterministic Seasonal:**
```math
S_t = \sum_{j=1}^{s} \alpha_j D_{j,t}
```

Where $`D_{j,t}`$ are seasonal dummy variables.

**Harmonic Seasonal:**
```math
S_t = \sum_{j=1}^{k} [A_j \cos(2\pi j t/s) + B_j \sin(2\pi j t/s)]
```

#### Example: Quarterly Dummy Variables

For quarterly data:
- **Q1**: $`D_{1,t} = 1`$ if t is Q1, 0 otherwise
- **Q2**: $`D_{2,t} = 1`$ if t is Q2, 0 otherwise
- **Q3**: $`D_{3,t} = 1`$ if t is Q3, 0 otherwise
- **Q4**: $`D_{4,t} = 1`$ if t is Q4, 0 otherwise

#### Seasonal Estimation

**1. Seasonal Subseries Method:**
```math
\hat{S}_j = \frac{1}{n_j} \sum_{i=1}^{n_j} (Y_{i,j} - \bar{Y})
```

**2. Seasonal Moving Average:**
```math
\hat{S}_t = \frac{1}{s} \sum_{i=0}^{s-1} Y_{t-i} - \hat{T}_t
```

#### Example: Monthly Sales Seasonality

For retail sales:
- **December**: $`\hat{S}_{Dec} = +500`$ (holiday boost)
- **January**: $`\hat{S}_{Jan} = -200`$ (post-holiday decline)
- **Summer months**: $`\hat{S}_{Jun-Aug} = +100`$ (vacation spending)

### Cyclical Component (C_t)

Cycles represent irregular, non-seasonal patterns that occur over longer periods.

#### Understanding Cycles

Cycles are different from seasonality because they:
- Don't repeat at fixed intervals
- Have variable amplitude and duration
- Are often related to economic or business cycles

#### Examples of Cycles

- **Business cycles**: Boom and bust periods
- **Product life cycles**: Introduction, growth, maturity, decline
- **Economic cycles**: Recession and expansion periods
- **Technology cycles**: Innovation waves

#### Mathematical Properties

1. **Non-periodic**: Cycles don't repeat at fixed intervals
2. **Variable Amplitude**: Cycle strength can vary over time
3. **Economic Nature**: Often related to business cycles

#### Cyclical Models

**ARMA Process:**
```math
C_t = \phi_1 C_{t-1} + \phi_2 C_{t-2} + \cdots + \phi_p C_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}
```

**Spectral Analysis:**
```math
C_t = \int_{-\pi}^{\pi} e^{i\omega t} dZ(\omega)
```

Where $`dZ(\omega)`$ is the spectral measure.

#### Example: Business Cycle

**Expansion phase**: Economic growth, increasing employment
**Peak**: Maximum economic activity
**Recession**: Economic decline, decreasing employment
**Trough**: Minimum economic activity

### Random Component ($`R_t`$)

The residual component captures unexplained variation.

#### Understanding Random Components

Random components represent:
- **Measurement error**: Instrument precision limitations
- **Model misspecification**: Incomplete understanding of relationships
- **Unpredictable events**: Natural disasters, policy changes
- **Inherent randomness**: Stochastic processes

#### Mathematical Properties

1. **Zero Mean**: $`E[R_t] = 0`$
2. **Constant Variance**: $`\text{Var}(R_t) = \sigma^2`$
3. **Uncorrelated**: $`\text{Cov}(R_t, R_{t-k}) = 0`$ for $`k \neq 0`$

#### Residual Analysis

```math
R_t = Y_t - \hat{T}_t - \hat{S}_t - \hat{C}_t
```

#### Example: Residual Interpretation

For stock price model:
- **Small residuals**: Model captures most variation
- **Large residuals**: Model misses important patterns
- **Patterned residuals**: Model assumptions violated

## Stationarity

Stationarity is a fundamental concept in time series analysis that ensures the statistical properties of the series remain constant over time.

### Understanding Stationarity

A stationary time series has:
- **Constant mean**: No trend
- **Constant variance**: No heteroscedasticity
- **Constant autocorrelation**: No changing patterns

#### Intuitive Example: Temperature vs. Stock Prices

**Temperature**: Non-stationary (trends, seasonal patterns)
**Stock returns**: Stationary (random fluctuations around zero)

### Types of Stationarity

#### Strict Stationarity

A process is strictly stationary if the joint distribution of any set of observations is invariant to time shifts.

**Mathematical Definition:**
For any $`k`$ and any $`h`$:
```math
F(Y_{t_1}, Y_{t_2}, \ldots, Y_{t_k}) = F(Y_{t_1+h}, Y_{t_2+h}, \ldots, Y_{t_k+h})
```

#### Weak Stationarity (Second-Order Stationarity)

A process is weakly stationary if:
1. **Constant mean**: $`E[Y_t] = \mu`$ for all t
2. **Constant variance**: $`\text{Var}(Y_t) = \sigma^2`$ for all t
3. **Constant autocovariance**: $`\text{Cov}(Y_t, Y_{t-k}) = \gamma_k`$ for all t

#### Example: Stationary vs. Non-stationary

**Stationary**: White noise, random walk differences
**Non-stationary**: Trends, seasonal patterns, changing variance

### Testing for Stationarity

#### Augmented Dickey-Fuller (ADF) Test

The ADF test checks for unit roots in the time series.

**Test Equation:**
```math
\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \epsilon_t
```

**Hypotheses:**
- $`H_0`$: Series has unit root (non-stationary)
- $`H_1`$: Series is stationary

**Decision Rule:**
- Reject $`H_0`$ if p-value < α (series is stationary)
- Fail to reject $`H_0`$ if p-value ≥ α (series is non-stationary)

#### Example: ADF Test Results

For monthly sales data:
- **Test statistic**: -2.45
- **p-value**: 0.12
- **Critical values**: -3.43 (1%), -2.86 (5%), -2.57 (10%)
- **Conclusion**: Fail to reject null (non-stationary)

#### KPSS Test

The KPSS test has opposite hypotheses to ADF.

**Hypotheses:**
- $`H_0`$: Series is stationary
- $`H_1`$: Series has unit root (non-stationary)

**Decision Rule:**
- Reject $`H_0`$ if test statistic > critical value (non-stationary)
- Fail to reject $`H_0`$ if test statistic ≤ critical value (stationary)

### Making Series Stationary

#### Differencing

**First Difference:**
```math
\Delta Y_t = Y_t - Y_{t-1}
```

**Second Difference:**
```math
\Delta^2 Y_t = \Delta Y_t - \Delta Y_{t-1} = Y_t - 2Y_{t-1} + Y_{t-2}
```

#### Example: Stock Prices vs. Returns

**Stock prices**: Non-stationary (trending upward)
**Stock returns**: Stationary (random fluctuations around zero)

#### Seasonal Differencing

For seasonal data with period s:
```math
\Delta_s Y_t = Y_t - Y_{t-s}
```

#### Example: Monthly Sales

**Original series**: Non-stationary (trend + seasonality)
**Seasonally differenced**: Removes seasonality, may still have trend
**First difference of seasonal difference**: Removes both trend and seasonality

#### Transformation

**Log Transformation:**
```math
Z_t = \log(Y_t)
```

**Box-Cox Transformation:**
```math
Z_t = \begin{cases}
\frac{Y_t^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(Y_t) & \text{if } \lambda = 0
\end{cases}
```

## Autocorrelation

Autocorrelation measures the correlation between observations at different time lags.

### Understanding Autocorrelation

Autocorrelation helps us understand:
- How current values relate to past values
- The memory of the time series
- Appropriate model specifications

#### Intuitive Example: Temperature

**Positive autocorrelation**: Today's temperature similar to yesterday's
**Negative autocorrelation**: Hot day followed by cool day
**No autocorrelation**: Temperature independent of previous days

### Autocorrelation Function (ACF)

#### Mathematical Definition

The autocorrelation function measures correlation at lag k:

```math
\rho_k = \frac{\text{Cov}(Y_t, Y_{t-k})}{\sqrt{\text{Var}(Y_t) \text{Var}(Y_{t-k})}} = \frac{\gamma_k}{\gamma_0}
```

Where $`\gamma_k = \text{Cov}(Y_t, Y_{t-k})`$ is the autocovariance at lag k.

#### Sample ACF

For a sample of size n:

```math
\hat{\rho}_k = \frac{\sum_{t=k+1}^{n} (Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{n} (Y_t - \bar{Y})^2}
```

#### Example: ACF Calculation

For monthly sales data:
- **Lag 1**: $`\hat{\rho}_1 = 0.85`$ (strong positive correlation)
- **Lag 2**: $`\hat{\rho}_2 = 0.72`$ (moderate positive correlation)
- **Lag 12**: $`\hat{\rho}_{12} = 0.65`$ (seasonal correlation)

### Partial Autocorrelation Function (PACF)

#### Understanding PACF

PACF measures the correlation between $`Y_t`$ and $`Y_{t-k}`$ after removing the effects of intermediate observations.

#### Mathematical Definition

```math
\phi_{kk} = \text{Corr}(Y_t - \hat{Y}_t, Y_{t-k} - \hat{Y}_{t-k})
```

Where $`\hat{Y}_t`$ is the linear prediction of $`Y_t`$ from $`Y_{t-1}, Y_{t-2}, \ldots, Y_{t-k+1}`$.

#### Example: PACF vs. ACF

**ACF**: Shows total correlation including indirect effects
**PACF**: Shows direct correlation, controlling for intermediate lags

### ACF and PACF Analysis

#### Model Identification

**AR(p) Process:**
- ACF: Tails off exponentially
- PACF: Cuts off after lag p

**MA(q) Process:**
- ACF: Cuts off after lag q
- PACF: Tails off exponentially

**ARMA(p,q) Process:**
- ACF: Tails off after lag q
- PACF: Tails off after lag p

#### Example: Model Identification

For monthly sales:
- **ACF**: Significant at lags 1, 2, 12, 24
- **PACF**: Significant at lags 1, 2
- **Interpretation**: AR(2) with seasonal component

#### Confidence Intervals

For white noise, approximately 95% of ACF/PACF values should fall within:
```math
\pm \frac{1.96}{\sqrt{n}}
```

Where n is the sample size.

## Moving Averages

Moving averages are fundamental tools for smoothing time series and identifying trends.

### Understanding Moving Averages

Moving averages help us:
- **Smooth out noise**: Reduce random fluctuations
- **Identify trends**: Reveal underlying patterns
- **Detect seasonality**: Highlight periodic patterns
- **Forecast**: Provide baseline predictions

#### Intuitive Example: Stock Prices

**Daily prices**: Noisy, hard to see trends
**30-day moving average**: Smooth trend becomes visible
**200-day moving average**: Long-term trend emerges

### Simple and Exponential Moving Averages

#### Simple Moving Average (SMA)

**Mathematical Definition:**
```math
\text{SMA}_t = \frac{1}{k} \sum_{i=0}^{k-1} Y_{t-i}
```

Where k is the window size.

#### Example: 5-Day SMA

For daily stock prices:
- **Day 5**: $`\text{SMA}_5 = \frac{P_1 + P_2 + P_3 + P_4 + P_5}{5}`$
- **Day 6**: $`\text{SMA}_6 = \frac{P_2 + P_3 + P_4 + P_5 + P_6}{5}`$

#### Properties of SMA

1. **Equal weights**: All observations weighted equally
2. **Lag**: SMA lags behind the original series
3. **Smoothing**: Reduces noise and volatility
4. **Window effect**: Larger windows = more smoothing

#### Exponential Moving Average (EMA)

**Mathematical Definition:**
```math
\text{EMA}_t = \alpha Y_t + (1-\alpha) \text{EMA}_{t-1}
```

Where $`\alpha`$ is the smoothing parameter (0 < α ≤ 1).

#### Example: EMA Calculation

For α = 0.2:
- **Day 1**: $`\text{EMA}_1 = 0.2 \times Y_1 + 0.8 \times \text{EMA}_0`$
- **Day 2**: $`\text{EMA}_2 = 0.2 \times Y_2 + 0.8 \times \text{EMA}_1`$

#### Properties of EMA

1. **Exponential weights**: Recent observations weighted more heavily
2. **Less lag**: Responds faster to changes
3. **Smoothing**: Still reduces noise
4. **Parameter effect**: Larger α = less smoothing, more responsiveness

#### Comparison: SMA vs. EMA

**SMA**: Equal weights, more lag, more smoothing
**EMA**: Exponential weights, less lag, faster response

### Weighted Moving Average (WMA)

#### Mathematical Definition

```math
\text{WMA}_t = \sum_{i=0}^{k-1} w_i Y_{t-i}
```

Where $`w_i`$ are weights that sum to 1.

#### Example: 3-Point WMA

```math
\text{WMA}_t = 0.5 Y_t + 0.3 Y_{t-1} + 0.2 Y_{t-2}
```

### Moving Average Applications

#### Trend Identification

**Short-term trend**: 10-day moving average
**Medium-term trend**: 50-day moving average
**Long-term trend**: 200-day moving average

#### Support and Resistance

**Support**: Price level where stock tends to stop falling
**Resistance**: Price level where stock tends to stop rising
**Moving averages**: Often act as dynamic support/resistance

#### Example: Trading Signals

**Golden Cross**: Short-term MA crosses above long-term MA (bullish)
**Death Cross**: Short-term MA crosses below long-term MA (bearish)

## ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models are powerful tools for time series forecasting.

### Understanding ARIMA Models

ARIMA models combine:
- **Autoregression**: Current value depends on past values
- **Integration**: Differencing to achieve stationarity
- **Moving Average**: Current value depends on past errors

#### Intuitive Example: Sales Forecasting

**Autoregression**: This month's sales depend on last month's sales
**Moving Average**: This month's sales depend on last month's forecast error
**Integration**: Use sales growth rates instead of absolute sales

### ARIMA Model Structure

#### Mathematical Definition

ARIMA(p,d,q) model:

```math
(1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p)(1 - B)^d Y_t = (1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q)\epsilon_t
```

Where:
- $`p`$ = order of autoregression
- $`d`$ = degree of differencing
- $`q`$ = order of moving average
- $`B`$ = backshift operator ($`BY_t = Y_{t-1}`$)

#### Expanded Form

```math
Y_t = \mu + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}
```

#### Example: ARIMA(1,1,1)

```math
\Delta Y_t = \phi_1 \Delta Y_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1}
```

### Model Identification

#### Box-Jenkins Methodology

1. **Stationarity**: Check if series is stationary
2. **Differencing**: Apply differencing if needed
3. **Model Selection**: Use ACF/PACF to identify p and q
4. **Estimation**: Estimate parameters
5. **Diagnostics**: Check residuals
6. **Forecasting**: Generate predictions

#### Example: Model Selection

For monthly sales data:
- **Original series**: Non-stationary (ADF p-value = 0.15)
- **First difference**: Stationary (ADF p-value = 0.001)
- **ACF**: Significant at lags 1, 12
- **PACF**: Significant at lags 1, 2
- **Model**: ARIMA(2,1,1) with seasonal component

### ARIMA Model Fitting

#### Parameter Estimation

**Maximum Likelihood Estimation:**
```math
L(\phi, \theta, \sigma^2) = \prod_{t=1}^{n} f(Y_t | Y_{t-1}, Y_{t-2}, \ldots)
```

**Least Squares Estimation:**
```math
\min_{\phi, \theta} \sum_{t=1}^{n} \epsilon_t^2
```

#### Example: AR(1) Estimation

For AR(1) model: $`Y_t = \phi Y_{t-1} + \epsilon_t`$

**OLS estimator:**
```math
\hat{\phi} = \frac{\sum_{t=2}^{n} Y_t Y_{t-1}}{\sum_{t=2}^{n} Y_{t-1}^2}
```

#### Model Diagnostics

**Residual Analysis:**
1. **Normality**: Q-Q plot, Shapiro-Wilk test
2. **Independence**: Ljung-Box test
3. **Homoscedasticity**: Residual plot

**Example: Diagnostic Results**
- **Ljung-Box p-value**: 0.45 (residuals independent)
- **Shapiro-Wilk p-value**: 0.12 (residuals normal)
- **Conclusion**: Model adequate

### Seasonal ARIMA (SARIMA)

#### Mathematical Definition

SARIMA(p,d,q)(P,D,Q,s):

```math
\phi_p(B)\Phi_P(B^s)(1-B)^d(1-B^s)^D Y_t = \theta_q(B)\Theta_Q(B^s)\epsilon_t
```

Where:
- $`(P,D,Q,s)`$ = seasonal orders
- $`s`$ = seasonal period

#### Example: Monthly Data

For monthly sales with seasonal pattern:
- **Non-seasonal**: ARIMA(1,1,1)
- **Seasonal**: ARIMA(1,1,1)(1,1,1,12)
- **Total model**: 12 parameters

## Seasonal Decomposition

Seasonal decomposition separates a time series into trend, seasonal, and residual components.

### Understanding Seasonal Decomposition

Seasonal decomposition helps us:
- **Understand patterns**: Identify trend and seasonal effects
- **Forecast**: Use components for prediction
- **Anomaly detection**: Identify unusual observations
- **Data preprocessing**: Remove seasonality for analysis

#### Intuitive Example: Electricity Usage

**Trend**: Increasing usage over years
**Seasonal**: Higher usage in summer (air conditioning)
**Residual**: Unusual weather, holidays, etc.

### STL Decomposition

STL (Seasonal and Trend decomposition using Loess) is a robust method for seasonal decomposition.

#### Mathematical Framework

**Additive Model:**
```math
Y_t = T_t + S_t + R_t
```

**Multiplicative Model:**
```math
Y_t = T_t \times S_t \times R_t
```

#### STL Algorithm

1. **Trend Extraction**: Apply loess smoothing to remove seasonality
2. **Seasonal Extraction**: Apply loess smoothing to detrended series
3. **Residual Calculation**: Subtract trend and seasonal components

#### Example: STL Parameters

**Trend window**: 13 months (for monthly data)
**Seasonal window**: 7 months
**Robust**: True (handles outliers)

### Classical Decomposition

#### Additive Decomposition

1. **Trend**: Centered moving average
2. **Seasonal**: Average of detrended values by season
3. **Residual**: Original minus trend minus seasonal

#### Multiplicative Decomposition

1. **Trend**: Centered moving average
2. **Seasonal**: Ratio of original to trend, averaged by season
3. **Residual**: Original divided by trend and seasonal

#### Example: Monthly Sales Decomposition

**Original**: $`Y_t = 1000 + 50t + 200S_t + 50R_t`$
**Trend**: $`T_t = 1000 + 50t`$ (linear growth)
**Seasonal**: $`S_t`$ (12 monthly effects)
**Residual**: $`R_t`$ (random variation)

### X-13ARIMA-SEATS

X-13ARIMA-SEATS is a comprehensive seasonal adjustment program.

#### Features

1. **Multiple decomposition methods**
2. **Outlier detection**
3. **Calendar effects**
4. **Quality diagnostics**

#### Example: GDP Data

**Original**: Quarterly GDP with seasonal patterns
**Adjusted**: GDP without seasonal effects
**Trend**: Long-term economic growth
**Seasonal**: Regular quarterly patterns

## Forecasting

Forecasting uses historical data to predict future values of a time series.

### Understanding Forecasting

Forecasting involves:
- **Model selection**: Choosing appropriate forecasting method
- **Parameter estimation**: Fitting model to historical data
- **Forecast generation**: Predicting future values
- **Forecast evaluation**: Assessing prediction accuracy

#### Intuitive Example: Weather Forecasting

**Historical data**: Past temperature, humidity, pressure
**Model**: Weather prediction algorithms
**Forecast**: Tomorrow's temperature
**Evaluation**: Compare prediction to actual temperature

### Time Series Forecasting

#### Forecasting Methods

**1. Naive Forecast:**
```math
\hat{Y}_{t+1} = Y_t
```

**2. Seasonal Naive:**
```math
\hat{Y}_{t+1} = Y_{t-s+1}
```

**3. Moving Average:**
```math
\hat{Y}_{t+1} = \frac{1}{k} \sum_{i=0}^{k-1} Y_{t-i}
```

**4. Exponential Smoothing:**
```math
\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha) \hat{Y}_t
```

#### Example: Sales Forecasting

**Last month**: 1000 units
**Naive forecast**: 1000 units
**3-month average**: 950 units
**Exponential smoothing**: 980 units

### ARIMA Forecasting

#### Forecast Equation

For ARIMA(p,d,q) model:

```math
\hat{Y}_{t+h} = \mu + \phi_1 \hat{Y}_{t+h-1} + \cdots + \phi_p \hat{Y}_{t+h-p} + \theta_1 \epsilon_{t+h-1} + \cdots + \theta_q \epsilon_{t+h-q}
```

#### Example: AR(1) Forecast

For AR(1): $`Y_t = 0.8 Y_{t-1} + \epsilon_t`$

**1-step ahead**: $`\hat{Y}_{t+1} = 0.8 Y_t`$
**2-step ahead**: $`\hat{Y}_{t+2} = 0.8 \hat{Y}_{t+1} = 0.64 Y_t`$

#### Forecast Intervals

**95% Forecast Interval:**
```math
\hat{Y}_{t+h} \pm 1.96 \sqrt{\text{Var}(\hat{Y}_{t+h})}
```

### Forecast Evaluation

#### Accuracy Metrics

**Mean Absolute Error (MAE):**
```math
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |Y_i - \hat{Y}_i|
```

**Mean Squared Error (MSE):**
```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
```

**Root Mean Squared Error (RMSE):**
```math
\text{RMSE} = \sqrt{\text{MSE}}
```

**Mean Absolute Percentage Error (MAPE):**
```math
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{Y_i - \hat{Y}_i}{Y_i}\right|
```

#### Example: Forecast Comparison

**Model A**: MAE = 50, RMSE = 65, MAPE = 5%
**Model B**: MAE = 45, RMSE = 60, MAPE = 4.5%
**Conclusion**: Model B performs better

### Advanced Forecasting Methods

#### Prophet

Prophet is Facebook's forecasting tool for time series with strong seasonal patterns.

**Model Components:**
```math
y(t) = g(t) + s(t) + h(t) + \epsilon_t
```

Where:
- $`g(t)`$ = trend
- $`s(t)`$ = seasonality
- $`h(t)`$ = holidays
- $`\epsilon_t`$ = error

#### Neural Network Methods

**LSTM (Long Short-Term Memory):**
- Captures long-term dependencies
- Handles non-linear relationships
- Requires large datasets

**Example: Stock Price Prediction**
- **Input**: Past 60 days of prices
- **Output**: Next day's price
- **Architecture**: LSTM with 50 hidden units

## Practical Applications

### Stock Price Analysis

#### Technical Analysis

**Moving Averages:**
- **Golden Cross**: 50-day MA crosses above 200-day MA
- **Death Cross**: 50-day MA crosses below 200-day MA

**Support and Resistance:**
- **Support**: Price level where stock stops falling
- **Resistance**: Price level where stock stops rising

#### Example: Apple Stock Analysis

**Data**: Daily closing prices for 2 years
**Trend**: Upward trend with volatility
**Seasonality**: Quarterly earnings effects
**Forecast**: 30-day ahead prediction

### Sales Forecasting

#### Business Applications

**Retail Sales:**
- **Trend**: Overall business growth
- **Seasonality**: Holiday effects, weather patterns
- **Cycles**: Economic conditions
- **Random**: Promotions, events

#### Example: E-commerce Sales

**Data**: Daily sales for 1 year
**Model**: SARIMA(1,1,1)(1,1,1,7) for weekly seasonality
**Forecast**: Next 30 days of sales
**Accuracy**: 85% within 10% of actual

### Economic Indicators

#### GDP Forecasting

**Components:**
- **Trend**: Long-term economic growth
- **Seasonality**: Quarterly patterns
- **Cycles**: Business cycles
- **Random**: Policy changes, shocks

#### Example: Quarterly GDP

**Data**: Quarterly GDP for 20 years
**Model**: ARIMA(2,1,2) with seasonal adjustment
**Forecast**: Next 4 quarters
**Use**: Economic policy planning

### Weather Forecasting

#### Meteorological Applications

**Temperature Prediction:**
- **Trend**: Climate change effects
- **Seasonality**: Annual temperature cycles
- **Cycles**: El Niño/La Niña patterns
- **Random**: Daily weather variations

#### Example: Daily Temperature

**Data**: Daily temperatures for 5 years
**Model**: SARIMA(1,0,1)(1,1,1,365) for annual seasonality
**Forecast**: Next 7 days
**Accuracy**: 90% within 2°C

## Practice Problems

### Problem 1: Trend Analysis

**Objective**: Create functions to detect and analyze different types of trends.

**Tasks**:
1. Implement linear, quadratic, and exponential trend detection
2. Create trend strength measures
3. Add trend change point detection
4. Include visualization of trend components
5. Add statistical tests for trend significance

**Example Implementation**:
```python
def trend_analysis(time_series, method='linear'):
    """
    Analyze trend in time series data.
    
    Returns: trend_type, trend_strength, change_points
    """
    # Implementation here
```

### Problem 2: Seasonality Detection

**Objective**: Implement methods to automatically detect seasonal patterns.

**Tasks**:
1. Create seasonal period detection algorithms
2. Implement seasonal strength measures
3. Add seasonal decomposition methods
4. Include seasonal adjustment procedures
5. Add visualization of seasonal components

### Problem 3: Forecast Evaluation

**Objective**: Build comprehensive forecast evaluation frameworks.

**Tasks**:
1. Implement multiple accuracy metrics (MAE, RMSE, MAPE)
2. Create forecast interval calculations
3. Add backtesting procedures
4. Include model comparison tools
5. Add forecast visualization

### Problem 4: Anomaly Detection

**Objective**: Develop time series anomaly detection methods.

**Tasks**:
1. Implement statistical anomaly detection
2. Add machine learning approaches
3. Create real-time detection systems
4. Include multiple detection algorithms
5. Add anomaly visualization

### Problem 5: Real-World Time Series Analysis

**Objective**: Apply time series analysis to real datasets.

**Tasks**:
1. Choose a dataset (financial, economic, environmental)
2. Perform comprehensive exploratory analysis
3. Build and compare multiple forecasting models
4. Conduct thorough model diagnostics
5. Write comprehensive analysis report

## Further Reading

### Books
- **"Time Series Analysis: Forecasting and Control"** by Box, Jenkins, Reinsel, and Ljung
- **"Forecasting: Principles and Practice"** by Rob J. Hyndman and George Athanasopoulos
- **"Time Series Analysis: Univariate and Multivariate Methods"** by William W.S. Wei
- **"Applied Time Series Analysis"** by Wayne A. Woodward, Henry L. Gray, and Alan C. Elliott
- **"Time Series Analysis and Its Applications"** by Robert H. Shumway and David S. Stoffer

### Online Resources
- **Rob Hyndman's Blog**: Forecasting principles and practice
- **Time Series Analysis**: Comprehensive online course
- **Coursera**: Time Series Analysis course
- **edX**: Statistical Learning course

### Advanced Topics
- **State Space Models**: Kalman filtering and smoothing
- **GARCH Models**: Volatility modeling for financial data
- **Neural Networks**: LSTM and GRU for time series
- **Bayesian Time Series**: Probabilistic forecasting
- **Multivariate Time Series**: Vector ARIMA and cointegration

## Key Takeaways

### Fundamental Concepts
- **Time series components** include trend, seasonality, cyclical, and random components
- **Stationarity** is crucial for many time series models and can be tested using ADF and KPSS tests
- **Autocorrelation** analysis helps identify patterns and guide model selection
- **Moving averages** provide smoothing and trend identification
- **ARIMA models** are powerful for forecasting stationary time series
- **Seasonal decomposition** separates time series into interpretable components
- **Forecasting** requires careful model selection and evaluation
- **Real-world applications** include stock prices, sales forecasting, and economic indicators

### Mathematical Tools
- **Decomposition methods** separate time series into interpretable components
- **Stationarity tests** ensure appropriate model assumptions
- **Autocorrelation functions** identify temporal dependencies
- **Moving averages** provide smoothing and trend extraction
- **ARIMA models** combine autoregression and moving averages
- **Forecast evaluation** metrics assess prediction accuracy

### Applications
- **Financial analysis** uses time series for stock price prediction
- **Business forecasting** predicts sales, demand, and revenue
- **Economic modeling** analyzes GDP, inflation, and employment
- **Environmental monitoring** tracks climate and pollution data
- **Quality control** monitors manufacturing processes

### Best Practices
- **Always check stationarity** before applying ARIMA models
- **Use multiple evaluation metrics** for forecast assessment
- **Consider seasonal patterns** in decomposition and modeling
- **Validate models** with out-of-sample testing
- **Interpret results** in context of domain knowledge

### Next Steps
In the following chapters, we'll build on time series foundations to explore:
- **Multivariate Statistics**: Principal component analysis and factor analysis
- **Analysis of Variance**: Comparing means across multiple groups
- **Nonparametric Methods**: When assumptions are violated
- **Advanced Topics**: Specialized methods for complex data structures

Remember that time series analysis is not just about fitting models to data—it's about understanding temporal patterns, making informed predictions, and extracting meaningful insights from time-ordered observations. The methods and concepts covered in this chapter provide the foundation for sophisticated temporal data analysis and evidence-based forecasting. 