import glob
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import xgboost as xgb
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.metrics import make_scorer

# Assuming you have your preprocessing.py file
from preprocessing import *

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

# Add lagged values of an existing feature (e.g., return_1d)
lagged_features = ["return_1d"]
for i in range(1, 4):
    for feature in lagged_features:
        X_selected = X_selected.with_columns(pl.col(feature).shift(i).alias(f"{feature}_lag_{i}"))

# Remove rows with NaNs that were introduced due to lagging
X_selected = X_selected.drop_nulls()

# Need to adjust y lengths to be the same length as X after removing nulls
y = y.slice(3)

# --- XGBoost Model Implementation ---

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
X_test_scaled = scaler.transform(X_test)

# Convert scaled data to pandas for xgboost
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
y_train = y_train.to_pandas().squeeze()  # Convert to pandas Series
y_test = y_test.to_pandas().squeeze()

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Define custom scorer
def theils_u_scorer(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    theils_u_val = np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)))
    return -theils_u_val  # Invert to treat it as a loss to minimize

# Create a scorer using make_scorer
custom_scorer = make_scorer(theils_u_scorer)

# Set up TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Set up GridSearchCV with the custom scorer
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring=custom_scorer, cv=tscv, verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Best model
best_xgb_model = grid_search.best_estimator_

# --- Model Evaluation ---

# Predictions
y_train_pred = best_xgb_model.predict(X_train_scaled)
y_test_pred = best_xgb_model.predict(X_test_scaled)

# 1. In-Sample Performance
print("\nIn-Sample Performance:")
print(f"  Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  Train R^2: {r2_score(y_train, y_train_pred):.4f}")

# Custom metrics
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def theils_u(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2)))

print(f"  Train MAPE: {mape(y_train, y_train_pred):.2f}%")
print(f"  Train Theil's U: {theils_u(y_train, y_train_pred):.4f}")

# 2. Out-of-Sample Performance
print("\nOut-of-Sample Performance:")
print(f"  Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  Test R^2: {r2_score(y_test, y_test_pred):.4f}")
print(f"  Test MAPE: {mape(y_test, y_test_pred):.2f}%")
print(f"  Test Theil's U: {theils_u(y_test, y_test_pred):.4f}")

# --- Feature Importance ---

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_xgb_model, importance_type='gain', max_num_features=20)
plt.title("Feature Importance (Gain)")
plt.savefig("XGBoost Feature Importance.png")

# --- Residual Analysis ---
# (Note: This is more relevant for regression models like GLSAR, but can still provide some insights)

residuals = y_test - y_test_pred

# Plot residuals
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("XGBoost Residual Analysis.png")

print("\n--- XGBoost Model Training Complete ---")