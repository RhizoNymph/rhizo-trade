import glob
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from preprocessing import *

from schema import expected_schema, enforce_schema

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

# Load the data
df = load_data()

# Select features and target variable (keep 'timestamp' for later)
X = df.select(["timestamp", "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d", "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"])
y = df.select("future_return_14d")

# Perform VIF-based feature selection
selected_features = select_features_vif_polars(X.drop("timestamp"), threshold=5)
selected_features = ["timestamp"] + selected_features  # Add 'timestamp' back
X_selected = X.select(selected_features)

# --- Adapt the data for walk-forward cross-validation ---

# Make sure 'timestamp' is not included in feature scaling or interaction terms
features_for_scaling = [col for col in selected_features if col != 'timestamp']

# Create interaction terms (degree=2) for the selected features
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_interaction = interaction.fit_transform(X_selected.select(features_for_scaling))
feature_names = interaction.get_feature_names_out(features_for_scaling)

# Convert back to Polars DataFrame with feature names
X_interaction = pl.DataFrame(X_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)

# Scale the interaction features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_interaction)

# Convert scaled data back to Polars DataFrame
X_scaled = pl.DataFrame(X_scaled, schema=X_interaction.columns)

# Add 'timestamp' back to the scaled features
X_scaled = X_scaled.with_columns(X_selected.select("timestamp"))

# Concatenate the scaled features and the target variable
data = pl.concat([X_scaled, y], how='horizontal')

# Convert the combined DataFrame to a NumPy array
data_np = data.to_numpy()

# Separate the features (including 'timestamp') and the target variable
X_np = data_np[:, :-1]  # All columns except the last one
y_np = data_np[:, -1]   # The last column

# --- Set up XGBoost parameters and hyperparameter grid ---

model_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'nthread': 4,
    'seed': 42,
    'device': 'cuda'
}

hyperparameter_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'gamma': [0, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# --- Walk-forward cross-validation for XGBoost ---

def time_series_walk_forward_cv_xgboost(features, target, model_params, hyperparameter_grid, n_splits, test_size):
    """
    Performs walk-forward cross-validation for time series data with hyperparameter tuning,
    progressively adding validation points one at a time during evaluation.

    Specialized for XGBoost.
    """
    from itertools import product
    import time

    # Split data into training/validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=test_size, shuffle=False
    )

    # Convert data to NumPy arrays and move to GPU
    X_train_val = torch.tensor(X_train_val, dtype=torch.float32, device='cuda')
    y_train_val = torch.tensor(y_train_val, dtype=torch.float32, device='cuda').view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32, device='cuda')
    y_test = torch.tensor(y_test, dtype=torch.float32, device='cuda').view(-1, 1)

    # Prepare for walk-forward validation
    n_train_val = len(X_train_val)
    fold_size = n_train_val // n_splits

    # Ensure at least one sample in the initial training set
    if fold_size == 0:
        raise ValueError("Fold size cannot be zero. Increase n_train_val or decrease n_splits.")

    # Store results
    train_rmses = []
    val_rmses = []
    best_avg_val_rmse = float('inf')
    best_model_params = None

    # Generate all hyperparameter combinations
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Walk-forward cross-validation
    for i in range(n_splits):
        print(f"Starting fold {i+1} of {n_splits}")

        # Split into initial training and validation sets for this fold
        start_train = 0
        end_train = max(1, i * fold_size)
        start_val = end_train
        end_val = start_val + fold_size

        # Hyperparameter tuning for this fold
        for hyperparam_idx, hyperparams in enumerate(hyperparameter_combinations):
            fold_start_time = time.time()
            print(f"  Starting hyperparameter combination {hyperparam_idx+1} of {len(hyperparameter_combinations)} (Fold {i+1})")

            fold_train_rmses = []
            fold_val_rmses = []

            # Progressively add validation points to the training set
            X_train_fold = X_train_val[start_train:end_train]
            y_train_fold = y_train_val[start_train:end_train]
            for j in range(len(X_train_val[start_val:end_val])):
                # 1. Data scaling (fit on training, transform both)
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train_fold.cpu())
                X_val_scaled = scaler_X.transform(X_train_val[start_val + j:start_val + j + 1].cpu())

                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train_fold.cpu())

                # Convert scaled data to tensors and move to GPU
                X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device='cuda')
                X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32, device='cuda')
                y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32, device='cuda')

                # 2. Model training
                current_model_params = model_params.copy()
                current_model_params.update(hyperparams)

                model = xgb.XGBRegressor(**current_model_params)
                model.fit(X_train_scaled, y_train_scaled.ravel()) # Train on GPU

                # 3. Predictions
                y_train_pred = model.predict(X_train_scaled) # Predict on GPU
                y_val_pred = model.predict(X_val_scaled) # Predict on GPU

                # 4. Inverse transform predictions
                y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
                y_val_pred = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
                y_train_fold_rescaled = scaler_y.inverse_transform(y_train_scaled.cpu()).flatten()

                # Convert predictions back to tensors on GPU
                y_train_pred = torch.tensor(y_train_pred, dtype=torch.float32, device='cuda')
                y_val_pred = torch.tensor(y_val_pred, dtype=torch.float32, device='cuda')
                y_train_fold_rescaled = torch.tensor(y_train_fold_rescaled, dtype=torch.float32, device='cuda')

                # 5. Evaluate
                train_rmse = np.sqrt(mean_squared_error(y_train_fold_rescaled.cpu(), y_train_pred.cpu()))
                val_rmse = np.sqrt(mean_squared_error(y_train_val[start_val + j:start_val + j + 1].cpu(), y_val_pred.cpu()))

                # Add the current validation point to the training set
                X_train_fold = torch.cat([X_train_fold, X_train_val[start_val + j:start_val + j + 1]])
                y_train_fold = torch.cat([y_train_fold, y_train_val[start_val + j:start_val + j + 1]])

                fold_train_rmses.append(train_rmse)
                fold_val_rmses.append(val_rmse)

            # Calculate average RMSE for this hyperparameter combination in this fold
            avg_fold_train_rmse = np.mean(fold_train_rmses)
            avg_fold_val_rmse = np.mean(fold_val_rmses)

            # 6. Store fold results for this hyperparameter combination
            train_rmses.append(avg_fold_train_rmse)
            val_rmses.append(avg_fold_val_rmse)

            # 7. Update best model and hyperparameters (based on average validation RMSE)
            if avg_fold_val_rmse < best_avg_val_rmse:
                best_avg_val_rmse = avg_fold_val_rmse
                best_model_params = current_model_params

            fold_end_time = time.time()
            print(f"  Finished hyperparameter combination {hyperparam_idx+1} in {fold_end_time - fold_start_time:.2f} seconds")
            print(f"    Average Train RMSE: {avg_fold_train_rmse:.4f}, Average Validation RMSE: {avg_fold_val_rmse:.4f}")

    # 8. Train best model on the entire training + validation set and evaluate on the test set
    scaler_X_final = StandardScaler()
    X_train_val_scaled = scaler_X_final.fit_transform(X_train_val.cpu())
    X_test_scaled = scaler_X_final.transform(X_test.cpu())

    scaler_y_final = StandardScaler()
    y_train_val_scaled = scaler_y_final.fit_transform(y_train_val.cpu())

    # Convert scaled data to tensors and move to GPU
    X_train_val_scaled = torch.tensor(X_train_val_scaled, dtype=torch.float32, device='cuda')
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device='cuda')
    y_train_val_scaled = torch.tensor(y_train_val_scaled, dtype=torch.float32, device='cuda')

    best_model = xgb.XGBRegressor(**best_model_params)
    best_model.fit(X_train_val_scaled, y_train_val_scaled.ravel())

    y_test_pred = best_model.predict(X_test_scaled)

    y_test_pred = scaler_y_final.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Convert predictions back to tensors on GPU
    y_test_pred = torch.tensor(y_test_pred, dtype=torch.float32, device='cuda')

    test_rmse = np.sqrt(mean_squared_error(y_test.cpu(), y_test_pred.cpu()))

    # 9. Return results
    avg_train_rmse = np.mean(train_rmses)
    avg_val_rmse = np.mean(val_rmses)

    return avg_train_rmse, avg_val_rmse, test_rmse, best_model_params, best_avg_val_rmse

# --- Perform walk-forward cross-validation for XGBoost ---

avg_train_rmse, avg_val_rmse, test_rmse, best_model_params, best_avg_val_rmse = time_series_walk_forward_cv_xgboost(
    features=X_np,
    target=y_np,
    model_params=model_params,
    hyperparameter_grid=hyperparameter_grid,
    n_splits=5,
    test_size=0.2
)

print("\nXGBoost Cross-Validation Results:")
print(f"  Average Training RMSE: {avg_train_rmse:.4f}")
print(f"  Average Validation RMSE: {avg_val_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Best Hyperparameters: {best_model_params}")
print(f"  Best Average Validation RMSE: {best_avg_val_rmse:.4f}")