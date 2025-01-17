"""
XGBoost Time Series Cross Validation

This module implements time series cross-validation for XGBoost models with various
feature engineering and evaluation capabilities.
"""

# Standard library imports
import logging
import multiprocessing
import os
import random
import time
from datetime import datetime
from itertools import product

# Third-party imports
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from preprocessing import select_features_vif_polars
from data import load_data

# Configuration
pl.Config.set_tbl_rows(100)
pl.Config(tbl_cols=10)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Schema definition
EXPECTED_SCHEMA = {
    "exchange": pl.Utf8,
    "coin": pl.Utf8,
    "product": pl.Utf8,
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "trade_count": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
}

# Feature engineering functions
def calculate_returns(group, future_periods=[1, 3, 5, 7, 14, 30], 
                     past_periods=[1, 3, 5, 7, 14, 30], frequency='d', 
                     lags=7, winsorize=True):
    """
    Calculate various return-based features for time series data.
    
    Args:
        group (pl.DataFrame): Input dataframe
        future_periods (list): Periods for future returns
        past_periods (list): Periods for past returns
        frequency (str): Time frequency ('d' for daily)
        lags (int): Number of lagged periods
        winsorize (bool): Whether to winsorize outliers
        
    Returns:
        tuple: (pl.DataFrame with calculated features, list of feature names)
    """
    existing_columns = set(group.columns)
    feature_names = []
    
    # Calculate past returns
    for period in past_periods:
        col_name = f'return_{period}{frequency}'
        group = group.with_columns(
            (pl.col('close_price') / pl.col('close_price').shift(period) - 1).alias(col_name)
        )
        feature_names.append(col_name)

    if winsorize:
        # Identify return columns to winsorize
        return_cols = [col for col in group.columns if 'return' in col]
        
        # Winsorize within each group
        for col in return_cols:
            # Calculate percentiles within the group
            group = group.with_columns(
                pl.when(pl.col(col).is_not_null())
                .then(pl.col(col).clip(pl.col(col).quantile(0.01), pl.col(col).quantile(0.99)))
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Calculate standard deviation of past returns
    for period in past_periods[1:]:
        std_col_name = f'return_std_{period}{frequency}'
        group = group.with_columns(
            pl.col(f'return_{period}{frequency}').rolling_std(period).alias(std_col_name)
        )
        feature_names.append(std_col_name)

    # Calculate future returns
    for period in future_periods:
        col_name = f'future_return_{period}{frequency}'
        group = group.with_columns(
            (pl.col('close_price').shift(-period) / pl.col('close_price') - 1).alias(col_name)
        )
        feature_names.append(col_name)
    
    # Add lags for features
    lagged_features = []
    for col in feature_names:
        for lag in range(1, lags + 1):
            lag_col_name = f'{col}_lag_{lag}'
            group = group.with_columns(pl.col(col).shift(lag).alias(lag_col_name))
            lagged_features.append(lag_col_name)
    
    feature_names.extend(lagged_features)
    return group, feature_names

def calculate_trading_features(group, lags=7):
    """Calculate trading-related features."""
    feature_names = []
    
    if 'buy_coin_volume' in group.columns:
        trading_features = {
            'coin_volume_bs_ratio': (pl.col('buy_coin_volume') / pl.col('sell_coin_volume')),
            'trades_bs_ratio': (pl.col('buy_trades') / pl.col('sell_trades')),
            'total_coin_volume': (pl.col('buy_coin_volume') + pl.col('sell_coin_volume')),
            'total_trades': (pl.col('buy_trades') + pl.col('sell_trades'))
        }
        
        for name, expr in trading_features.items():
            group = group.with_columns(expr.alias(name))
            feature_names.append(name)
    
    # Add lags for features
    lagged_features = []
    for col in feature_names:
        for lag in range(1, lags + 1):
            lag_col_name = f'{col}_lag_{lag}'
            group = group.with_columns(pl.col(col).shift(lag).alias(lag_col_name))
            lagged_features.append(lag_col_name)
    
    feature_names.extend(lagged_features)
    return group, feature_names

def calculate_time_features(group, lags=7):
    """Calculate time-based features."""
    feature_names = []
    
    # Convert Unix timestamp (in milliseconds) to datetime
    group = group.with_columns(
        pl.col('timestamp').map_elements(
            lambda x: datetime.utcfromtimestamp(x / 1000), return_dtype=pl.Datetime
        ).alias('datetime')
    )
    
    # Extract time features
    time_features = {
        'day_of_week': pl.col('datetime').dt.weekday(),
        'day_of_month': pl.col('datetime').dt.day(),
        'month_of_year': pl.col('datetime').dt.month()
    }
    
    for name, expr in time_features.items():
        group = group.with_columns(expr.alias(name))
        feature_names.append(name)
    
    group = group.drop('datetime')
    
    # Add lags for features
    lagged_features = []
    for col in feature_names:
        for lag in range(1, lags + 1):
            lag_col_name = f'{col}_lag_{lag}'
            group = group.with_columns(pl.col(col).shift(lag).alias(lag_col_name))
            lagged_features.append(lag_col_name)
    
    feature_names.extend(lagged_features)
    return group, feature_names

# Function to calculate VIF
def calculate_vif(data: pl.DataFrame, threshold: float = 5.0):
    """Calculate Variance Inflation Factor for feature selection."""
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
def select_features_vif_polars(data: pl.DataFrame, threshold: float = 5.0):
    """Select features based on VIF threshold."""
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

# Data preprocessing functions
def enforce_schema(df):
    """Enforce the expected schema on the dataframe."""
    return df.select(
        [pl.col(col_name).cast(col_type) for col_name, col_type in EXPECTED_SCHEMA.items()]
    )

def preprocess_data(X, selected_features=None, poly_transform=None, 
                   scaler=None, for_training=True):
    """
    Preprocess features including feature selection, polynomial features, and scaling.
    
    Args:
        X (pl.DataFrame): Input features
        selected_features (list): List of features to use
        poly_transform (PolynomialFeatures): Fitted polynomial transformer
        scaler (StandardScaler): Fitted scaler
        for_training (bool): Whether this is for training data
        
    Returns:
        tuple: (processed_features, scaler, poly_transform, selected_features)
    """
    # Feature selection
    if selected_features is None:
        selected_features = select_features_vif_polars(X.drop("timestamp"), threshold=5)
        selected_features = ["timestamp"] + selected_features  
    X_selected = X.select(selected_features)

    # Separate 'timestamp' from the features to be transformed
    features_for_transform = [col for col in selected_features if col != 'timestamp']
    X_features = X_selected.select(features_for_transform)
    X_timestamp = X_selected.select("timestamp")

    # Convert to numpy for sklearn transformers
    X_features_np = X_features.to_numpy()
    X_timestamp_np = X_timestamp.to_numpy()

    # Apply polynomial features
    if poly_transform is None:
        poly_transform = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly_transform.fit_transform(X_features_np)
    else:
        X_poly = poly_transform.transform(X_features_np)

    # Scale the features (excluding timestamp)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)  
    else:
        X_scaled = scaler.transform(X_poly)

    # Combine scaled features with timestamp
    X_scaled_df = pl.DataFrame(X_scaled)
    timestamp_series = pl.Series("timestamp", X_timestamp_np.flatten().tolist())

    # Ensure the number of rows matches
    assert X_scaled.shape[0] == len(X_timestamp_np), "Mismatch in number of rows."

    # Get feature names after polynomial transformation
    poly_feature_names = poly_transform.get_feature_names_out(features_for_transform)

    # Add 'timestamp' to the list of feature names
    all_feature_names = list(poly_feature_names)

    # Assign the correct column names
    X_scaled_df.columns = all_feature_names

    return X_scaled_df, scaler, poly_transform, selected_features

def load_and_preprocess_data(target_period=7):
    """
    Load and preprocess the complete dataset.
    
    Args:
        target_period (int): Number of days for future return prediction (e.g., 7 for 7-day returns)
    
    Returns:
        tuple: (numpy array of data, scaler, poly_transform, selected_features, target_col)
    """
    df = load_data()
    all_features = []

    # Calculate returns and get feature names
    df, return_features = df.group_by('coin').map_groups(
        lambda group: calculate_returns(group, frequency='d')
    )
    all_features.extend(return_features)

    # Calculate trading features if available
    df, trading_features = df.group_by('coin').map_groups(
        lambda group: calculate_trading_features(group)
    )
    all_features.extend(trading_features)

    # Calculate time features
    df, time_features = df.group_by('coin').map_groups(
        lambda group: calculate_time_features(group)
    )
    all_features.extend(time_features)

    # Add timestamp to features
    all_features = ['timestamp'] + all_features

    # Select features and target
    target_col = f'future_return_{target_period}d'
    X = df.select(all_features)
    y = df.select(target_col)

    # Preprocess the data
    X_processed, scaler, poly_transform, selected_features = preprocess_data(X)

    # Combine features and target
    data = pl.concat([pl.DataFrame(X_processed), y], how='horizontal')
    
    return data.to_numpy(), scaler, poly_transform, selected_features, target_col

# Model training and evaluation
def time_series_walk_forward_cv_xgboost_parallel(
        features, target, model_params, hyperparameter_grid, 
        selected_features, mode='dynamic', window_type='expanding',
        n_folds=5, validation_days=30, min_training_days=90,
        test_size=0.1, gap_days=7):
    """
    Perform time series walk-forward cross-validation with XGBoost.
    
    Args:
        features (np.ndarray): Feature matrix
        target (np.ndarray): Target values
        model_params (dict): XGBoost model parameters
        hyperparameter_grid (dict): Grid of hyperparameters to search
        selected_features (list): Selected feature names
        mode (str): Cross-validation mode ('dynamic' or 'static')
        window_type (str): Window type for training ('expanding' or 'sliding')
        n_folds (int): Number of cross-validation folds
        validation_days (int): Number of days in validation set
        min_training_days (int): Minimum number of training days
        test_size (float): Proportion of data for test set
        gap_days (int): Gap between train and validation sets
        
    Returns:
        tuple: (best_params, cv_results, test_predictions, feature_importance)
    """
    logging.info("Starting time series walk-forward cross-validation")
    logging.info(f"Data shape: Features {features.shape}, Target {target.shape}")
    
    # Split into train-val and final test
    total_days = len(features)
    test_days = int(total_days * test_size)
    X_train_val = features[:-test_days]
    y_train_val = target[:-test_days]
    X_test = features[-test_days:]
    y_test = target[-test_days:]
    
    logging.info(f"Train-val size: {len(X_train_val)}, Test size: {len(X_test)}")
    
    # Calculate number of validation windows
    if mode == 'dynamic':
        total_train_val_days = len(X_train_val)
        n_splits = (total_train_val_days - min_training_days - gap_days) // validation_days
        logging.info(f"Using dynamic mode with {n_splits} validation windows of {validation_days} days each")
        logging.info(f"Minimum training size: {min_training_days} days")
        logging.info(f"Gap between train and validation sets: {gap_days} days")
    elif mode == 'fixed':   
        n_splits = n_folds
        logging.info(f"Using fixed mode with {n_splits} folds")
    else:
        raise ValueError("Invalid mode. Choose 'dynamic' or 'fixed'.")
    
    val_rmses = []
    best_val_rmse = float('inf')
    best_model_params = None
    all_predictions = []

    rmse_by_hyperparams = {}
    
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
    logging.info(f"Testing {len(hyperparameter_combinations)} hyperparameter combinations")
    
    # Fit the scaler on the entire training data
    scaler = StandardScaler()
    scaler.fit(X_train_val) 
    
    args_list = []
    for fold_idx in range(n_splits):
        if mode == 'dynamic':
            # Calculate window boundaries for dynamic mode
            end_val = total_train_val_days - (n_splits - fold_idx - 1) * validation_days
            start_val = end_val - validation_days
            
            if window_type == 'expanding':
                # Expanding window: training set grows over time
                start_train = 0  
                end_train = start_val - gap_days
            elif window_type == 'sliding':
                # Sliding window: training set has a fixed size
                start_train = start_val - min_training_days - gap_days
                end_train = start_val - gap_days
            else:
                raise ValueError("Invalid window_type. Choose 'expanding' or 'sliding'.")
            
            if end_train - start_train < min_training_days:
                continue  
        elif mode == 'fixed':
            # Calculate window boundaries for fixed mode
            val_window_size = len(X_train_val) // n_splits
            start_val = fold_idx * val_window_size
            end_val = (fold_idx + 1) * val_window_size if fold_idx < n_splits - 1 else len(X_train_val)
            start_train = 0
            end_train = start_val - gap_days
            
            if end_train - start_train < 1:
                continue  
        
        X_train_fold = X_train_val[start_train:end_train]
        y_train_fold = y_train_val[start_train:end_train]
        X_val_fold = X_train_val[start_val:end_val]
        y_val_fold = y_train_val[start_val:end_val]

        num_samples = 480
        if len(hyperparameter_combinations) < num_samples:
            sampled_combinations = hyperparameter_combinations
        else:
            sampled_combinations = random.sample(hyperparameter_combinations, num_samples)

        for hyperparams in sampled_combinations:
            start_time_fold = time.time()
            args_list.append((hyperparams, fold_idx, X_train_fold, y_train_fold, 
                            X_val_fold, y_val_fold, model_params, start_time_fold, scaler))
    
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn')

        with multiprocessing.Pool() as pool:
            logging.info("Starting parallel processing of folds")
            results = pool.map(train_and_evaluate, args_list)

        logging.info("Processing results from all folds")
        for val_rmse, hyperparams, fold_idx, duration, predictions in results:
            val_rmses.append(val_rmse)
            all_predictions.extend(predictions)

            hyperparams_tuple = tuple(sorted(hyperparams.items()))

            if hyperparams_tuple not in rmse_by_hyperparams:
                rmse_by_hyperparams[hyperparams_tuple] = []
            rmse_by_hyperparams[hyperparams_tuple].append(val_rmse)

            print(f"Fold {fold_idx+1}, Hyperparams: {hyperparams}, Val RMSE: {val_rmse:.4f}, Time: {duration:.2f}s")

        # Create predictions DataFrame and sort by timestamp
        predictions_df = pl.DataFrame(all_predictions)
        predictions_df = predictions_df.sort("timestamp")
        
        print("\nLatest Validation Predictions:")
        print(predictions_df.tail(10))

        # Calculate average RMSE for each hyperparameter combination
        # Calculate average RMSE for each hyperparameter combination
        avg_rmse_by_hyperparams = {}
        for hyperparams_tuple, rmses in rmse_by_hyperparams.items():
            avg_rmse_by_hyperparams[hyperparams_tuple] = np.mean(rmses)

        # Sort the models by average RMSE and select the top 5
        sorted_hyperparams = sorted(avg_rmse_by_hyperparams.items(), key=lambda x: x[1])[:5]
        top_5_hyperparams = [item[0] for item in sorted_hyperparams]
        top_5_avg_rmses = [item[1] for item in sorted_hyperparams]

        logging.info(f"Top 5 average RMSEs: {top_5_avg_rmses}")
        logging.info(f"Top 5 hyperparameters: {top_5_hyperparams}")

        # Scale the entire training data
        X_train_val_scaled = scaler.transform(X_train_val)
        X_train_val_scaled = torch.tensor(X_train_val_scaled, dtype=torch.float32, device='cuda')

        # Ensure y_train_val is of type float32
        y_train_val = y_train_val.astype(np.float32)
        y_train_val = torch.tensor(y_train_val, dtype=torch.float32, device='cuda')

        # Split the training data into training and validation sets
        split_idx = int(0.8 * len(X_train_val_scaled)) 
        X_train_final = X_train_val_scaled[:split_idx]
        y_train_final = y_train_val[:split_idx]
        X_val_final = X_train_val_scaled[split_idx:]
        y_val_final = y_train_val[split_idx:]

        # Train the top 5 models on all data except the test set
        top_5_models = []
        for hyperparams_tuple in top_5_hyperparams:
            # Create a copy of the base model parameters
            model_params_copy = model_params.copy()  # Use your original `model_params` here
            # Update with the best hyperparameters for this model
            model_params_copy.update(dict(hyperparams_tuple))
            # Add early stopping rounds
            model_params_copy['early_stopping_rounds'] = 10
            
            # Initialize and train the model
            model = xgb.XGBRegressor(**model_params_copy)
            model.fit(
                X_train_final.cpu().numpy(), y_train_final.cpu().numpy(),
                eval_set=[(X_val_final.cpu().numpy(), y_val_final.cpu().numpy())],
                verbose=False
            )
            
            # Save the trained model
            top_5_models.append(model)

        # Save the top 5 models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for i, model in enumerate(top_5_models):
            model_path = os.path.join(model_dir, f"xgboost_model_{timestamp}_top_{i+1}.json")
            model.save_model(model_path)
            logging.info(f"Saved top {i+1} model to {model_path}")

        # Scale the test data
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device='cuda')

        # Make predictions on the test set using all top 5 models
        test_predictions = []
        for model in top_5_models:
            y_test_pred = model.predict(X_test_scaled.cpu().numpy())
            test_predictions.append(y_test_pred)

        # Average the predictions
        y_test_pred_avg = np.mean(test_predictions, axis=0)

        # Calculate RMSE and R² for the averaged predictions
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_avg))
        test_r2 = r2_score(y_test, y_test_pred_avg)

        logging.info(f"Final Test Set RMSE (Averaged): {test_rmse:.4f}")
        logging.info(f"Final Test Set R² (Averaged): {test_r2:.4f}")

        # Load the current data for final predictions
        from data import load_data
        current_data = load_data(for_training=False)

        cols = ["timestamp", "day_of_week", "day_of_month", "month_of_year",
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"]
    
        features = [
        "day_of_week", "day_of_month", "month_of_year",
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"
        ]
        lags=7

        lag_features = [f"{feature}_lag_{lag}" for feature in features for lag in range(1, lags + 1)]

        cols = cols + lag_features

        if 'buy_coin_volume' in current_data.columns:
            cols = cols + ["coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"]
        else:
            cols = cols + ["volume"]

        # Define X_current by selecting relevant features from current_data
        X_current = current_data.select(cols + ["coin"])
        X_current = X_current.drop_nulls()

        # Get the coin identifiers
        current_coins = X_current["coin"].to_numpy()
        X_current = X_current.drop("coin")

        # Log the columns of X_current
        logging.info(f"Selected features: {selected_features}")
        logging.info(f"X_current columns: {X_current.columns}")

        # Ensure all selected features are present in X_current
        missing_features = [col for col in selected_features if col not in X_current.columns]
        if missing_features:
            logging.warning(f"Missing features in X_current: {missing_features}")
            # Add missing features with default values (e.g., 0)
            for feature in missing_features:
                print(f"feature: {feature} missing")
                exit()

        # Filter features using selected_features
        X_current_selected = X_current.select(selected_features)

        # Exclude 'timestamp' from transformations
        features_for_transform = [col for col in selected_features if col != 'timestamp']
        X_current_features = X_current_selected.select(features_for_transform).to_numpy()

        # Apply polynomial features using the same poly_transform object
        X_current_poly = poly_transform.transform(X_current_features)

        # Scale the features using the same scaler object
        X_current_scaled = scaler.transform(X_current_poly)

        # Combine scaled features with timestamp
        X_current_timestamp = X_current_selected.select("timestamp").to_numpy()
        X_current_processed = np.hstack([X_current_scaled, X_current_timestamp])

        # Make predictions on the current data using all top 5 models
        current_predictions = []
        for model in top_5_models:
            y_current_pred = model.predict(X_current_processed[:, :-1])  # Drop 'timestamp' before prediction
            current_predictions.append(y_current_pred)

        # Average the predictions
        y_current_pred_avg = np.mean(current_predictions, axis=0)

        # Create the final predictions dictionary
        current_predictions_avg = {
            'timestamp': X_current_timestamp.flatten(),
            'coin': current_coins,  
            'predicted_future_return_14d': y_current_pred_avg  
        }

        # Convert to Polars DataFrame
        current_predictions_df = pl.DataFrame(current_predictions_avg)

        # Get only most current predictions of most recent timestamp
        current_predictions_df = current_predictions_df.filter(pl.col("timestamp") == current_predictions_df["timestamp"].max())

        # Sort by predicted_future_return_14d (descending order)
        current_predictions_df = current_predictions_df.sort("predicted_future_return_14d", descending=True)

        print("\nCurrent Predictions Sorted by Predicted (Descending):")
        print(current_predictions_df.head(20))
        
        logging.info("Validation complete!")

        y_test_pred = y_test_pred  # If y_test_pred is a tensor

        plot_residuals_predictions_and_rmse_distribution(y_test, y_test_pred, test_rmse, val_rmses, predictions_df)

        return test_rmse, best_model_params, predictions_df, current_predictions_df
        
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    TARGET_PERIOD = 7  # Change this to predict different future return periods
    
    # Load and preprocess data
    data_np, scaler, poly_transform, selected_features, target_col = load_and_preprocess_data(target_period=TARGET_PERIOD)
    
    # Define model parameters and hyperparameter grid
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    hyperparameter_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Run cross-validation and get results
    best_params, cv_results, test_predictions, feature_importance = \
        time_series_walk_forward_cv_xgboost_parallel(
            features=data_np,
            target=data_np[:, -1],  # Last column is target
            model_params=model_params,
            hyperparameter_grid=hyperparameter_grid,
            selected_features=selected_features,
            mode='dynamic',
            window_type='expanding',
            n_folds=5,
            validation_days=30,
            min_training_days=90,
            test_size=0.1,
            gap_days=7
        )
    
    logging.info(f"Training model to predict {target_col}")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Cross-validation results: {cv_results}")
    logging.info(f"Feature importance: {feature_importance}")

def train_and_evaluate(args):
    import torch
    import logging
    import xgboost as xgb
    import time
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    """Train and evaluate the model for a given fold and hyperparameters."""
    hyperparams, fold_idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold, model_params, start_time_fold, scaler = args
    
    logging.info(f"Starting fold {fold_idx+1} evaluation with hyperparams: {hyperparams}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Scale the data using the provided scaler
    X_train_scaled = scaler.transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    # Update model parameters with hyperparameters
    current_model_params = model_params.copy()
    current_model_params.update(hyperparams)
    current_model_params['early_stopping_rounds'] = 5
    
    model = xgb.XGBRegressor(**current_model_params)
    
    # Train the model
    model.fit(
        X_train_scaled, y_train_fold,
        eval_set=[(X_val_scaled, y_val_fold)],
        verbose=False
    )
    
    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred)) # This is RMSE
    
    # Store predictions
    predictions = [{
        'timestamp': X_val_fold[i, -1],
        'actual': y_val_fold[i],
        'predicted': y_val_pred[i],
        'error': abs(y_val_fold[i] - y_val_pred[i]) # Individual error (not RMSE)
    } for i in range(len(y_val_fold))]
    
    return val_rmse, hyperparams, fold_idx, time.time() - start_time_fold, predictions

# Visualization functions
def plot_residuals_predictions_and_rmse_distribution(
        y_test, y_test_pred, test_rmse, val_rmses, predictions_df):
    """
    Create comprehensive diagnostic plots for model evaluation.
    
    Args:
        y_test (np.ndarray): True test values
        y_test_pred (np.ndarray): Predicted test values
        test_rmse (float): Test set RMSE
        val_rmses (list): Validation set RMSEs
        predictions_df (pl.DataFrame): DataFrame with predictions
    """
    # Ensure y_test and y_test_pred are NumPy arrays
    if isinstance(y_test_pred, torch.Tensor):
        y_test_pred = y_test_pred.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    # Flatten y_test_pred and y_test if they are not 1-dimensional
    y_test_pred = y_test_pred.flatten()
    y_test = y_test.flatten()

    residuals = y_test - y_test_pred

    # Convert Polars DataFrame to Pandas DataFrame for compatibility with plotting functions
    predictions_df = predictions_df.to_pandas()

    # Sort predictions by timestamp for chronological plotting
    predictions_df = predictions_df.sort_values(by="timestamp")

    # --- Create Figure and Subplots ---
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))  # 3 rows, 3 columns

    # --- Residual Plots ---

    # 1. Overall Residual Plot
    axes[0, 0].scatter(range(len(residuals)), residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title(f'Overall Residuals (Test RMSE: {test_rmse:.4f})')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Residual')

    # 2. Residuals vs. Actual Values
    axes[0, 1].scatter(y_test, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_title('Residuals vs. Actual Values')
    axes[0, 1].set_xlabel('Actual Value')
    axes[0, 1].set_ylabel('Residual')

    # 3. Residuals Over Time
    min_len = min(len(predictions_df["timestamp"]), len(residuals))
    axes[0, 2].scatter(predictions_df["timestamp"][:min_len], residuals[:min_len], alpha=0.7)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title('Residuals Over Time')
    axes[0, 2].set_xlabel('Timestamp')
    axes[0, 2].set_ylabel('Residual')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # --- Distribution of Predictions ---
    sns.histplot(y_test_pred, ax=axes[1, 0], kde=True, color='skyblue', label='Predicted')
    sns.histplot(y_test, ax=axes[1, 0], kde=True, color='orange', label='Actual')
    axes[1, 0].set_title('Distribution of Predictions and Actual Values')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # --- RMSE Distribution ---
    sns.histplot(val_rmses, ax=axes[1, 1], kde=True, color='green')
    axes[1, 1].axvline(x=test_rmse, color='r', linestyle='--', label=f'Test RMSE: {test_rmse:.4f}')
    axes[1, 1].set_title('Distribution of Validation RMSEs')
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    # --- Actual vs Predicted Plot ---
    min_val = min(np.min(y_test), np.min(y_test_pred))
    max_val = max(np.max(y_test), np.max(y_test_pred))
    axes[1, 2].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # Diagonal line
    axes[1, 2].scatter(y_test, y_test_pred, alpha=0.7)
    axes[1, 2].set_title('Actual vs Predicted Values')
    axes[1, 2].set_xlabel('Actual Value')
    axes[1, 2].set_ylabel('Predicted Value')
    axes[1, 2].set_xlim([min_val, max_val])
    axes[1, 2].set_ylim([min_val, max_val])

    # --- Distribution of Residuals ---
    sns.histplot(residuals, ax=axes[2, 0], kde=True, color='purple')
    axes[2, 0].set_title('Distribution of Residuals')
    axes[2, 0].set_xlabel('Residual')
    axes[2, 0].set_ylabel('Frequency')

    # --- QQ Plot for Residuals ---
    stats.probplot(residuals, dist="norm", plot=axes[2, 1])
    axes[2, 1].set_title('QQ Plot of Residuals')
    axes[2, 1].set_xlabel('Theoretical Quantiles')
    axes[2, 1].set_ylabel('Sample Quantiles')

    # --- Autocorrelation of Residuals ---
    plot_acf(residuals, lags=40, ax=axes[2, 2], title='Autocorrelation of Residuals')
    axes[2, 2].set_xlabel('Lag')
    axes[2, 2].set_ylabel('Autocorrelation')

    plt.tight_layout()
    plt.savefig("combined_diagnostics.png")
    plt.close() 