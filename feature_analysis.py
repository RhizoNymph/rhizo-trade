import glob
import polars as pl
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming you have your preprocessing.py file
from preprocessing import *
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white, normal_ad
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera

# Expected schema definition
expected_schema = {
    "exchange": pl.Utf8,
    "coin": pl.Utf8,
    "product": pl.Utf8,
    "timestamp": pl.Int64,
    "open_price": pl.Float64,
    "high_price": pl.Float64,
    "low_price": pl.Float64,
    "close_price": pl.Float64,
    "coin_volume": pl.Float64,
    "dollar_volume": pl.Float64,
    "buy_trades": pl.Int64,
    "sell_trades": pl.Int64,
    "total_trades": pl.Int64,
    "buy_coin_volume": pl.Float64,
    "sell_coin_volume": pl.Float64,
    "buy_dollar_volume": pl.Float64,
    "sell_dollar_volume": pl.Float64,
}

# Function to enforce schema
def enforce_schema(df):
    return df.select(
        [pl.col(col_name).cast(col_type) for col_name, col_type in expected_schema.items()]
    )

# Function to load and preprocess data
def load_data(data_dir='data/velo/spot/binance/1d'):
    files = glob.glob(f'{data_dir}/*.parquet')
    df = pl.concat([
        enforce_schema(pl.read_parquet(f)) for f in files
    ])

    # Sort by coin and timestamp
    df = df.sort(['coin', 'timestamp'])

    df = df.group_by('coin').map_groups(lambda group: calculate_returns(group, frequency=data_dir[-1]))
    df = df.group_by('coin').map_groups(lambda group: calculate_trading_features(group))

    df = df.drop_nulls()

    return df

# Function to calculate VIF
def calculate_vif(data: pl.DataFrame, threshold: float = 5.0) -> pl.Series:
    # Convert to NumPy for statsmodels compatibility
    X = data.to_numpy()

    # Add a constant to the features for statsmodels
    X_const = sm.add_constant(X)

    # Calculate VIF using statsmodels
    vifs = []
    for i in range(X_const.shape[1]):
        if i == 0:
            vifs.append(np.nan)  # VIF for constant term is undefined
            continue

        y = X_const[:, i]
        X_i = np.delete(X_const, i, axis=1)

        model = sm.OLS(y, X_i).fit()
        r_squared = model.rsquared

        if 1 - r_squared < 1e-10:
            vif = np.inf
        else:
            vif = 1 / (1 - r_squared)
        vifs.append(vif)

    # Create a Polars Series with feature names and VIFs
    vif_series = pl.Series("VIF", vifs[1:], dtype=pl.Float64)

    return vif_series

# Function for VIF-based feature selection
def select_features_vif_polars(data: pl.DataFrame, threshold: float = 5.0) -> list:
    selected_features = list(data.columns)
    removed_features = []

    while True:
        vif_data = data.select(selected_features)
        vif_series = calculate_vif(vif_data, threshold)

        max_vif = vif_series.max()
        if max_vif > threshold:
            feature_to_remove = vif_data.columns[vif_series.arg_max()]
            selected_features.remove(feature_to_remove)
            removed_features.append(feature_to_remove)
        else:
            break

    print("Removed features due to high VIF:", removed_features)
    return selected_features

df = load_data()

# Select features and target variable
X = df.select(["timestamp", "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d", "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"])
y = df.select("future_return_14d")

# Perform VIF-based feature selection
selected_features = select_features_vif_polars(X, threshold=5)
X_selected = X.select(selected_features)

# Split the data into training and testing sets (time-series split)
train_size = 0.8
split_idx = int(len(X_selected) * train_size)

X_train = X_selected[:split_idx]
X_test = X_selected[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler for the test set

# Convert scaled data back to Polars DataFrame
X_train_scaled = pl.DataFrame(X_train_scaled, schema=X_selected.columns)
X_test_scaled = pl.DataFrame(X_test_scaled, schema=X_selected.columns)

# Create interaction terms (degree=2)
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_interaction = interaction.fit_transform(X_train_scaled)
X_test_interaction = interaction.transform(X_test_scaled)
feature_names = interaction.get_feature_names_out(X_train_scaled.columns)

# Convert back to Polars DataFrame with feature names
X_train_interaction = pl.DataFrame(X_train_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)
X_test_interaction = pl.DataFrame(X_test_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)

# Add a constant term for statsmodels
X_train_const = sm.add_constant(X_train_interaction.to_numpy())
X_test_const = sm.add_constant(X_test_interaction.to_numpy())

# Fit the GLSAR model
rho = 5  # Starting with AR(2) based on previous Ljung-Box results. Change as needed.
model = sm.GLSAR(y_train.to_numpy(), X_train_const, rho=rho)
glsar_model = model.iterative_fit(maxiter=5)

# Print the model summary
print(glsar_model.summary())

# --- Validation ---

# 1. Residual Analysis

# a) Autocorrelation (Ljung-Box Test)
lb_test = acorr_ljungbox(glsar_model.resid, lags=[10], return_df=True)  # Check lags up to 10
print("\nLjung-Box test for autocorrelation:")
print(lb_test)

# b) Heteroskedasticity (White Test)
white_test = het_white(glsar_model.resid, glsar_model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
white_results = dict(zip(labels, white_test))
print("\nWhite test for heteroskedasticity:")
print(white_results)

# c) Normality (Jarque-Bera and Anderson-Darling Tests)
jb_test = jarque_bera(glsar_model.resid)
labels = ['Jarque-Bera Test Statistic', 'Jarque-Bera p-value', 'Skew', 'Kurtosis']
jb_results = dict(zip(labels, jb_test))
print("\nJarque-Bera test for normality:")
print(jb_results)

ad_test = normal_ad(glsar_model.resid)
labels = ['Anderson-Darling Test Statistic', 'Anderson-Darling p-value']
ad_results = dict(zip(labels, ad_test))
print("\nAnderson-Darling test for normality:")
print(ad_results)

# d) Residual Plots

plt.figure(figsize=(12, 8))

# Time Series Plot of Residuals
plt.subplot(2, 2, 1)
plt.plot(glsar_model.resid)
plt.title("Time Series Plot of Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")

# Histogram of Residuals
plt.subplot(2, 2, 2)
plt.hist(glsar_model.resid, bins=30)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")

# ACF of Residuals
plt.subplot(2, 2, 3)
plot_acf(glsar_model.resid, lags=20, ax=plt.gca())  # Use plt.gca() to plot on current axes
plt.title("ACF of Residuals")

# PACF of Residuals
plt.subplot(2, 2, 4)
plot_pacf(glsar_model.resid, lags=20, ax=plt.gca()) # Use plt.gca() to plot on current axes
plt.title("PACF of Residuals")

plt.tight_layout()
plt.savefig("GLS Residual Diagnostics.png")

# 2. Out-of-Sample Performance

# Predictions on the test set
predictions = glsar_model.predict(exog=X_test_const)

# a) Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error (MAE): {mae}")

# b) Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# c) Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

mape_val = mape(y_test.to_numpy(), predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape_val:.2f}%")

# d) Theil's U
def theils_u(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)))

theils_u_val = theils_u(y_test, predictions)
print(f"Theil's U: {theils_u_val}")

# --- Model Improvement ---

# Example: Add lagged values of an existing feature (e.g., return_1d)

lagged_features = ["return_1d"]
for i in range(1, 4):
    for feature in lagged_features:
        X_train = X_train.with_columns(pl.col(feature).shift(i).alias(f"{feature}_lag_{i}"))
        X_test = X_test.with_columns(pl.col(feature).shift(i).alias(f"{feature}_lag_{i}"))

# Remove rows with NaNs that were introduced due to lagging
X_train = X_train.drop_nulls()
X_test = X_test.drop_nulls()

# Need to adjust y_train and y_test lengths to be the same length as X_train and X_test after removing nulls
y_train = y_train.slice(3)
y_test = y_test.slice(3)

# Rescale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back to Polars DataFrame
X_train_scaled = pl.DataFrame(X_train_scaled, schema=X_train.columns)
X_test_scaled = pl.DataFrame(X_test_scaled, schema=X_test.columns)

# Recreate interaction terms
X_train_interaction = interaction.fit_transform(X_train_scaled)
X_test_interaction = interaction.transform(X_test_scaled)
feature_names = interaction.get_feature_names_out(X_train_scaled.columns)

# Convert back to Polars DataFrame with feature names
X_train_interaction = pl.DataFrame(X_train_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)
X_test_interaction = pl.DataFrame(X_test_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)

# Add a constant term for statsmodels
X_train_const = sm.add_constant(X_train_interaction.to_numpy())
X_test_const = sm.add_constant(X_test_interaction.to_numpy())

# Refit the GLSAR model
model = sm.GLSAR(y_train.to_numpy(), X_train_const, rho=rho)
glsar_model = model.iterative_fit(maxiter=5)

# Print the model summary
print(glsar_model.summary())

# --- Validation of the Improved Model ---

# 1. Residual Analysis

# a) Autocorrelation (Ljung-Box Test)
lb_test = acorr_ljungbox(glsar_model.resid, lags=[10], return_df=True)
print("\nLjung-Box test for autocorrelation:")
print(lb_test)

# b) Heteroskedasticity (White Test)
white_test = het_white(glsar_model.resid, glsar_model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
white_results = dict(zip(labels, white_test))
print("\nWhite test for heteroskedasticity:")
print(white_results)

# c) Normality (Jarque-Bera and Anderson-Darling Tests)
jb_test = jarque_bera(glsar_model.resid)
labels = ['Jarque-Bera Test Statistic', 'Jarque-Bera p-value', 'Skew', 'Kurtosis']
jb_results = dict(zip(labels, jb_test))
print("\nJarque-Bera test for normality:")
print(jb_results)

ad_test = normal_ad(glsar_model.resid)
labels = ['Anderson-Darling Test Statistic', 'Anderson-Darling p-value']
ad_results = dict(zip(labels, ad_test))
print("\nAnderson-Darling test for normality:")
print(ad_results)

plt.figure(figsize=(12, 8))

# Time Series Plot of Residuals
plt.subplot(2, 2, 1)
plt.plot(glsar_model.resid)
plt.title("Time Series Plot of Residuals")
plt.xlabel("Time")
plt.ylabel("Residual")

# Histogram of Residuals
plt.subplot(2, 2, 2)
plt.hist(glsar_model.resid, bins=30)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")

# ACF of Residuals
plt.subplot(2, 2, 3)
plot_acf(glsar_model.resid, lags=20, ax=plt.gca())  # Use plt.gca() to plot on current axes
plt.title("ACF of Residuals")

# PACF of Residuals
plt.subplot(2, 2, 4)
plot_pacf(glsar_model.resid, lags=20, ax=plt.gca()) # Use plt.gca() to plot on current axes
plt.title("PACF of Residuals")

plt.tight_layout()
plt.savefig("Improved GLS Residual Diagnostics.png")

# 2. Out-of-Sample Performance

# Predictions on the test set
predictions = glsar_model.predict(exog=X_test_const)

# a) Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error (MAE): {mae}")

# b) Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# c) Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

mape_val = mape(y_test.to_numpy(), predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape_val:.2f}%")

# d) Theil's U
def theils_u(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)))

theils_u_val = theils_u(y_test, predictions)
print(f"Theil's U: {theils_u_val}")