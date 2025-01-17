import logging
import multiprocessing
import os
import random
import time
from datetime import datetime
from itertools import product

import numpy as np
import polars as pl
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from preprocessing import select_features_vif_polars
from data import load_data

import matplotlib.pyplot as plt
import seaborn as sns  

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

def plot_residuals_predictions_and_rmse_distribution(y_test, y_test_pred, test_rmse, val_rmses, predictions_df):
    """
    Plots the residuals of the test set predictions, the distribution of predictions,
    and a histogram of the validation RMSEs, all in one image with subplots.

    Args:
        y_test: True values of the test set (NumPy array).
        y_test_pred: Predicted values of the test set (NumPy array).
        test_rmse: RMSE of the test set.
        val_rmses: List of RMSEs from cross-validation.
        predictions_df: DataFrame containing test set predictions, including timestamps.
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

pl.Config.set_tbl_rows(100)
pl.Config(tbl_cols=10)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def preprocess_data(X, selected_features=None, poly_transform=None, scaler=None, for_training=True):
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

def load_and_preprocess_data(lags):
    """Load and preprocess the data."""
    df = load_data(lags=lags)

    cols = ["timestamp", "day_of_week", "day_of_month", "month_of_year",
            "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
            # "ewma_3d", "ewma_3d_dist", "ewma_5d", "ewma_5d_dist", "ewma_7d", "ewma_7d_dist", "ewma_14d", 
            # "ewma_14d_dist", "ewma_30d", "ewma_30d_dist",
            "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"]
    
    features = [
        "day_of_week", "day_of_month", "month_of_year",
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"
    ]

    lag_features = [f"{feature}_lag_{lag}" for feature in features for lag in range(1, lags + 1)]

    cols = cols + lag_features

    if 'buy_coin_volume' in df.columns:
        cols = cols + ["coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"]
    else:
        cols = cols + ["volume"]

    X = df.select(cols)
    
    y = df.select("future_return_std_7d")

    # Preprocess the data
    X_processed, scaler, poly_transform, selected_features = preprocess_data(X)

    # Combine features and target
    data = pl.concat([pl.DataFrame(X_processed), y], how='horizontal')
    
    return data.to_numpy(), scaler, poly_transform, selected_features

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

def time_series_walk_forward_cv_xgboost_parallel(
    features, target, model_params, hyperparameter_grid, selected_features, lags,
    mode='dynamic', window_type='expanding', n_folds=5, validation_days=30, 
    min_training_days=90, test_size=0.1, gap_days=7
):
    """Perform time series walk-forward cross-validation with XGBoost."""
    logging.info("Starting time series walk-forward cross-validation")
    logging.info(f"Data shape: Features {features.shape}, Target {target.shape}")
    
    # Keep data on CPU initially
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
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
        
        # Convert to float32 to save memory
        X_train_val_scaled = X_train_val_scaled.astype(np.float32)
        y_train_val = y_train_val.astype(np.float32)

        # Split the training data into training and validation sets
        split_idx = int(0.8 * len(X_train_val_scaled))
        X_train_final = X_train_val_scaled[:split_idx]
        y_train_final = y_train_val[:split_idx]
        X_val_final = X_train_val_scaled[split_idx:]
        y_val_final = y_train_val[split_idx:]

        # Process in batches for the top 5 models
        BATCH_SIZE = 10000  # Adjust based on your GPU memory
        top_5_models = []

        for hyperparams_tuple in top_5_hyperparams:
            # Create a copy of the base model parameters
            model_params_copy = model_params.copy()
            model_params_copy.update(dict(hyperparams_tuple))
            model_params_copy['early_stopping_rounds'] = 10
            
            # Train in batches if data is large
            if len(X_train_final) > BATCH_SIZE:
                model = xgb.XGBRegressor(**model_params_copy)
                for i in range(0, len(X_train_final), BATCH_SIZE):
                    end_idx = min(i + BATCH_SIZE, len(X_train_final))
                    batch_X = X_train_final[i:end_idx]
                    batch_y = y_train_final[i:end_idx]
                    
                    # Move batch to GPU, train, then free memory
                    with torch.cuda.device('cuda'):
                        batch_X_gpu = torch.tensor(batch_X, device='cuda')
                        batch_y_gpu = torch.tensor(batch_y, device='cuda')
                        
                        model.fit(
                            batch_X_gpu.cpu().numpy(), batch_y_gpu.cpu().numpy(),
                            eval_set=[(X_val_final, y_val_final)],
                            verbose=False,
                            xgb_model=model if i > 0 else None  # Continue training from previous batch
                        )
                        
                        # Explicitly free GPU memory
                        del batch_X_gpu, batch_y_gpu
                        torch.cuda.empty_cache()
            else:
                # For smaller datasets, train normally
                model = xgb.XGBRegressor(**model_params_copy)
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_val_final, y_val_final)],
                    verbose=False
                )
            
            top_5_models.append(model)

        # Save models and free memory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for i, model in enumerate(top_5_models):
            model_path = os.path.join(model_dir, f"xgboost_model_{timestamp}_top_{i+1}.json")
            model.save_model(model_path)
            logging.info(f"Saved top {i+1} model to {model_path}")

        # Process test data in batches
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = X_test_scaled.astype(np.float32)
        test_predictions = []

        for model in top_5_models:
            if len(X_test_scaled) > BATCH_SIZE:
                batch_predictions = []
                for i in range(0, len(X_test_scaled), BATCH_SIZE):
                    end_idx = min(i + BATCH_SIZE, len(X_test_scaled))
                    batch_X = X_test_scaled[i:end_idx]
                    
                    # Move batch to GPU, predict, then free memory
                    with torch.cuda.device('cuda'):
                        batch_X_gpu = torch.tensor(batch_X, device='cuda')
                        batch_pred = model.predict(batch_X_gpu.cpu().numpy())
                        batch_predictions.append(batch_pred)
                        
                        del batch_X_gpu
                        torch.cuda.empty_cache()
                
                y_test_pred = np.concatenate(batch_predictions)
            else:
                y_test_pred = model.predict(X_test_scaled)
                
            test_predictions.append(y_test_pred)

        # Average the predictions
        y_test_pred_avg = np.mean(test_predictions, axis=0)

        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_avg))
        test_r2 = r2_score(y_test, y_test_pred_avg)

        logging.info(f"Final Test Set RMSE (Averaged): {test_rmse:.4f}")
        logging.info(f"Final Test Set R² (Averaged): {test_r2:.4f}")

        # Load the current data for final predictions
        from data import load_data
        current_data = load_data(lags=lags, for_training=False)

        cols = ["timestamp", "day_of_week", "day_of_month", "month_of_year",
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        # "ewma_3d", "ewma_3d_dist", "ewma_5d", "ewma_5d_dist", "ewma_7d", "ewma_7d_dist", "ewma_14d", 
        # "ewma_14d_dist", "ewma_30d", "ewma_30d_dist",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"]
    
        features = [
        "day_of_week", "day_of_month", "month_of_year",
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d"
        ]

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

    lags=14

    # Load and preprocess the data
    data_np, scaler, poly_transform, selected_features = load_and_preprocess_data(lags=lags)

    # Split into features (X) and target (y)
    X_np = data_np[:, :-1]  
    y_np = data_np[:, -1]   

    # Define model parameters and hyperparameter grid
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'nthread': -1,
        'seed': 42,
        'device': 'cuda'
    }

    hyperparameter_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 7],
        'learning_rate': [0.03, 0.05],
        'gamma': [0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9, 1.0],
        'min_child_weight': [3],
        'lambda': [1],
        'alpha': [0]
    }       

    hyperparameter_grid = {
        'n_estimators': [100, 200, 300, 400],  
        'max_depth': [3, 5, 7, 9],  
        'learning_rate': [0.01, 0.03, 0.05, 0.1],  
        'gamma': [0, 0.1, 0.2],  
        'subsample': [0.8, 0.9, 1.0],  
        'colsample_bytree': [0.8, 0.9, 1.0],  
        'min_child_weight': [1, 3, 5],  
        'lambda': [0, 1, 2],  
        'alpha': [0, 0.5, 1]
    }

    # Run the cross-validation
    test_rmse, best_model_params, predictions_df, current_predictions_df = time_series_walk_forward_cv_xgboost_parallel(
        features=X_np,
        target=y_np,
        model_params=model_params,
        hyperparameter_grid=hyperparameter_grid,
        selected_features=selected_features,  
        lags=lags,
        mode='fixed',  
        window_type='expanding',
        n_folds=10,     
        validation_days=30,  
        min_training_days=90,  
        test_size=0.1,  
        gap_days=7      
    )   