import polars as pl
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from data import load_data
import pandas as pd

def plot_volatility_clustering(returns, title):
    """Plot volatility clustering and ACF of squared returns"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot returns
    ax1.plot(returns)
    ax1.set_title(f'{title} - Returns Over Time')
    ax1.set_ylabel('Returns')
    
    # Plot volatility
    volatility = np.sqrt(returns**2)
    ax2.plot(volatility)
    ax2.set_title('Volatility Over Time')
    ax2.set_ylabel('Volatility')
    
    # Plot squared returns ACF
    squared_returns = returns**2
    lags = 40
    acf = np.array([1] + [np.corrcoef(squared_returns[:-i], squared_returns[i:])[0,1] 
                         for i in range(1, lags)])
    
    ax3.bar(range(lags), acf)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.axhline(y=1.96/np.sqrt(len(returns)), color='r', linestyle='--')
    ax3.axhline(y=-1.96/np.sqrt(len(returns)), color='r', linestyle='--')
    ax3.set_title('ACF of Squared Returns (Volatility Clustering)')
    ax3.set_xlabel('Lag')
    
    plt.tight_layout()
    plt.savefig('volatility_clustering.png')
    plt.close()

def analyze_residuals(model_fit, returns):
    """Analyze model residuals"""
    residuals = model_fit.resid/np.sqrt(model_fit.conditional_volatility)
    
    # Normality test
    _, p_value = stats.jarque_bera(residuals)
    
    # Plot residuals diagnostics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Standardized residuals
    ax1.plot(residuals)
    ax1.set_title('Standardized Residuals')
    
    # QQ plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q plot')
    
    # Histogram
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title(f'Histogram (JB p-value: {p_value:.4f})')
    
    # ACF of absolute residuals
    lags = 20
    acf = np.array([1] + [np.corrcoef(np.abs(residuals)[:-i], np.abs(residuals)[i:])[0,1] 
                         for i in range(1, lags)])
    ax4.bar(range(lags), acf)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
    ax4.axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
    ax4.set_title('ACF of Absolute Residuals')
    
    plt.tight_layout()
    plt.savefig('residuals_diagnostics.png')
    plt.close()
    
    return p_value

def calculate_volatility_persistence(returns, window_sizes=[5, 10, 22, 66]):
    """Calculate volatility persistence using different window sizes"""
    results = []
    returns_series = pd.Series(returns)
    
    # Ensure we have enough data
    min_required = max(window_sizes) * 2
    if len(returns) < min_required:
        print(f"Warning: Need at least {min_required} observations for volatility persistence calculation")
        return
    
    for window in window_sizes:
        # Calculate rolling volatility
        rolling_vol = returns_series.rolling(window, min_periods=1).std()
        # Remove NaN values
        valid_vol = rolling_vol.dropna()
        
        if len(valid_vol) > window:
            # Calculate autocorrelation
            autocorr = np.corrcoef(valid_vol[window:].values, valid_vol[:-window].values)[0,1]
            results.append((window, autocorr))
        else:
            results.append((window, np.nan))
    
    print("\nVolatility Persistence:")
    for window, autocorr in results:
        if np.isnan(autocorr):
            print(f"{window}-day volatility autocorrelation: insufficient data")
        else:
            print(f"{window}-day volatility autocorrelation: {autocorr:.4f}")

# Step 1: Load Ethereum Price Data and check data quality
print("Loading Ethereum price data...")
df = load_data(lags=0, for_training=False)
eth_data = df.filter(pl.col('coin') == 'ETH')

# Print data info
print("\nData Overview:")
print(f"Date range: {eth_data.get_column('timestamp').min()} to {eth_data.get_column('timestamp').max()}")
print(f"Number of observations: {eth_data.shape[0]}")
print(f"Number of missing values: {eth_data.null_count().sum()}")

# Calculate returns and check for issues
eth_data = eth_data.with_columns([
    pl.col('close_price').pct_change().alias('Returns'),
    pl.col('timestamp').diff().alias('time_diff')
]).drop_nulls()

# Check for data quality
print("\nData Quality Check:")
print(f"Average time between observations: {eth_data.get_column('time_diff').mean()} seconds")
print(f"Number of observations after cleaning: {eth_data.shape[0]}")

# Get numpy array of returns
returns_np = eth_data.get_column('Returns').to_numpy()
squared_returns = returns_np**2

# Volatility statistics with checks
print("\nVolatility Statistics:")
print(f"Number of non-zero returns: {np.count_nonzero(returns_np)}")
print(f"Proportion of zero returns: {(returns_np == 0).mean():.2%}")
print(f"Mean Volatility: {np.sqrt(squared_returns.mean()):.6f}")
print(f"Max Volatility: {np.sqrt(squared_returns.max()):.6f}")
print(f"Volatility of Volatility: {np.sqrt(np.var(squared_returns)):.6f}")
print(f"Volatility Skewness: {stats.skew(squared_returns):.6f}")
print(f"Volatility Kurtosis: {stats.kurtosis(squared_returns):.6f}")

# Perform Engle's ARCH test with more lags
lags_to_test = [5, 10, 22]
print("\nEngle's ARCH test results:")
for lag in lags_to_test:
    arch_test = het_arch(returns_np, nlags=lag)
    lm_stat, p_value, f_stat, f_p_value = arch_test
    print(f"\nLags {lag}:")
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"LM test p-value: {p_value:.4f}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"F-test p-value: {f_p_value:.4f}")

# Plot volatility analysis
plot_volatility_clustering(returns_np, 'Ethereum')

# Calculate volatility persistence
calculate_volatility_persistence(returns_np)

# Additional return statistics
print("\nReturn Statistics:")
print(f"Mean: {returns_np.mean():.6f}")
print(f"Std Dev: {returns_np.std():.6f}")
print(f"Skewness: {stats.skew(returns_np):.6f}")
print(f"Kurtosis: {stats.kurtosis(returns_np):.6f}")

# Step 2: Train/Validate/Test Split
print("\nSplitting data into train, validation, and test sets...")
total_rows = eth_data.shape[0]
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.15)

train_data = eth_data.slice(0, train_size)
val_data = eth_data.slice(train_size, val_size)
test_data = eth_data.slice(train_size + val_size)

print(f"Train size: {train_data.shape[0]}, Validation size: {val_data.shape[0]}, Test size: {test_data.shape[0]}")

def evaluate_garch_model(train, val, p, q):
    """
    Fits a GARCH model on the training data and evaluates it on the validation data.
    Returns the model results and RMSE of the volatility forecasts.
    """
    returns = train.get_column('Returns').to_numpy()
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='t')  # Using Student's t distribution
    results = model.fit(disp='off')
    
    # Print model diagnostics
    print(f"\nGARCH({p},{q}) Model Summary:")
    print(f"AIC: {results.aic:.2f}")
    print(f"BIC: {results.bic:.2f}")
    print(f"Log-Likelihood: {results.loglikelihood:.2f}")
    
    # Forecast volatility on the validation set
    val_returns = val.get_column('Returns').to_numpy()
    forecasts = results.forecast(start=0, horizon=len(val_returns))
    val_forecast = forecasts.variance.iloc[-len(val_returns):].mean(axis=1)
    
    # Calculate RMSE between actual squared returns and forecasted variance
    actual_var = val_returns ** 2
    rmse = np.sqrt(mean_squared_error(actual_var, val_forecast))
    
    return results, rmse

# Step 4: Evaluate Different GARCH Models
print("\nEvaluating GARCH models...")
models = [(1, 1), (1, 2), (2, 1), (2, 2)]
results = []

for p, q in models:
    model_results, rmse = evaluate_garch_model(train_data, val_data, p, q)
    results.append((p, q, rmse, model_results.aic, model_results.bic))

# Create a DataFrame to compare models
model_comparison = pl.DataFrame({
    'p': [r[0] for r in results],
    'q': [r[1] for r in results],
    'RMSE': [r[2] for r in results],
    'AIC': [r[3] for r in results],
    'BIC': [r[4] for r in results]
})
print("\nModel Comparison:")
print(model_comparison)

# Select the best model based on AIC
best_model_idx = model_comparison.get_column('AIC').arg_min()
best_model = model_comparison.row(best_model_idx)
best_p, best_q = int(best_model[0]), int(best_model[1])
print(f"\nBest GARCH Model: p={best_p}, q={best_q}")
print(f"AIC: {best_model[3]:.2f}")
print(f"BIC: {best_model[4]:.2f}")
print(f"RMSE: {best_model[2]:.6f}")

# Step 5: Fit the Best Model on Combined Train + Validation Data
print("\nFitting the best model on combined train + validation data...")
combined_train_val = pl.concat([train_data, val_data])
returns = combined_train_val.get_column('Returns').to_numpy()
best_garch = arch_model(returns, vol='Garch', p=best_p, q=best_q, dist='t')
best_results = best_garch.fit(disp='off')

# Analyze residuals
jb_pvalue = analyze_residuals(best_results, returns)
print(f"\nJarque-Bera test p-value for standardized residuals: {jb_pvalue:.4f}")

# Print model parameters
print("\nModel Parameters:")
print(best_results.summary().tables[1])

# Step 6: Evaluate the Best Model on the Test Set
print("\nEvaluating the best model on the test set...")
test_returns = test_data.get_column('Returns').to_numpy()
test_forecasts = best_results.forecast(start=0, horizon=len(test_returns))
test_forecast_var = test_forecasts.variance.iloc[-len(test_returns):].mean(axis=1)

# Calculate RMSE on the test set
actual_test_var = test_returns ** 2
test_rmse = np.sqrt(mean_squared_error(actual_test_var, test_forecast_var))
print(f"Test RMSE: {test_rmse}")

# Step 7: Plot the Forecasted vs Actual Volatility
plt.figure(figsize=(12, 6))
timestamps = test_data.get_column('timestamp').to_numpy()

# Plot actual volatility
plt.plot(timestamps, np.sqrt(actual_test_var) * 100, label='Actual Volatility', alpha=0.6)
plt.plot(timestamps, np.sqrt(test_forecast_var) * 100, label='Forecasted Volatility', linestyle='--')
plt.title('Actual vs Forecasted Volatility (Test Set)')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('forecasted_vs_actual_volatility.png')
plt.close()