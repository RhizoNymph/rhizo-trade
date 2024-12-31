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

def load_and_preprocess_data():
    """Load and preprocess the data."""
    df = load_data()
    
    X = df.select([
                    "timestamp", 
                    "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
                    "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d",
                    #'vwma_3d', 'vwma_3d_dist', 'vwma_5d', 'vwma_5d_dist', 'vwma_7d', 'vwma_7d_dist', 'vwma_14d', 'vwma_14d_dist', 'vwma_30d', 'vwma_30d_dist',
                    "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"
                ])
    y = df.select("future_return_14d")

    # Preprocess the data
    X_processed, scaler, poly_transform, selected_features = preprocess_data(X)

    # Combine features and target
    data = pl.concat([pl.DataFrame(X_processed), y], how='horizontal')
    
    return data.to_numpy(), scaler, poly_transform, selected_features

def train_and_evaluate(args):
    """Train and evaluate the model for a    given fold and hyperparameters."""
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
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    
    # Store predictions
    predictions = [{
        'timestamp': X_val_fold[i, -1],
        'actual': y_val_fold[i],
        'predicted': y_val_pred[i],
        'rmse': val_rmse
    } for i in range(len(y_val_fold))]
    
    return val_rmse, hyperparams, fold_idx, time.time() - start_time_fold, predictions

def time_series_walk_forward_cv_xgboost_parallel(features, target, model_params, hyperparameter_grid, selected_features, mode='dynamic', n_folds=5, validation_days=30, min_training_days=90, test_size=0.1, gap_days=7):
    """Perform time series walk-forward cross-validation with XGBoost."""
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
            start_train = 0  
            end_train = start_val - gap_days
            
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

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_params = hyperparams
                logging.info(f"New best model found! RMSE: {best_val_rmse:.4f}")
                logging.info(f"Best hyperparameters: {best_model_params}")

            print(f"Fold {fold_idx+1}, Hyperparams: {hyperparams}, Val RMSE: {val_rmse:.4f}, Time: {duration:.2f}s")

        # Create predictions DataFrame and sort by timestamp
        predictions_df = pl.DataFrame(all_predictions)
        predictions_df = predictions_df.sort("timestamp")
        
        print("\nLatest Validation Predictions:")
        print(predictions_df.tail(10))

        # Train final model on all data except test set
        logging.info("Training final model with best hyperparameters")
        final_model_params = model_params.copy()
        final_model_params.update(best_model_params)
        final_model_params['early_stopping_rounds'] = 10

        final_model = xgb.XGBRegressor(**final_model_params)

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

        # Train the final model with early stopping
        final_model.fit(
            X_train_final.cpu().numpy(), y_train_final.cpu().numpy(),
            eval_set=[(X_val_final.cpu().numpy(), y_val_final.cpu().numpy())],
            verbose=False
        )
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_path = os.path.join(model_dir, f"xgboost_model_{timestamp}.json")
        final_model.save_model(model_path)
        logging.info(f"Saved model to {model_path}")
        
        # Make predictions on test set
        logging.info("Making predictions on test set")
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device='cuda')
        y_test_pred = final_model.predict(X_test_scaled.cpu().numpy())
        y_test_pred = torch.tensor(y_test_pred, dtype=torch.float32, device='cuda')
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred.cpu().numpy()))
        test_r2 = r2_score(y_test, y_test_pred.cpu().numpy())
        
        logging.info(f"Final Test Set RMSE: {test_rmse:.4f}")
        logging.info(f"Final Test Set R²: {test_r2:.4f}")
        
        # Load the current data for final predictions
        from data import load_data
        current_data = load_data(for_training=False) 
  

        # Define X_current by selecting relevant features from current_data
        X_current = current_data.select([
                                            "timestamp", "coin",
                                            "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
                                            "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d",
                                            #'vwma_3d', 'vwma_3d_dist', 'vwma_5d', 'vwma_5d_dist', 'vwma_7d', 'vwma_7d_dist', 'vwma_14d', 'vwma_14d_dist', 'vwma_30d', 'vwma_30d_dist',
                                            "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"
                                        ])
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

        # Make predictions on the current data (drop 'timestamp' before prediction)
        X_current_final = X_current_processed[:, :-1] 
        y_current_pred = final_model.predict(X_current_final)
        
        print(y_current_pred.shape)
        # Create current predictions DataFrame
        current_predictions = {
            'timestamp': X_current_timestamp.flatten(),
            'coin': current_coins,  
            'predicted_future_return_14d': y_current_pred  
        }

        # Convert to Polars DataFrame
        current_predictions_df = pl.DataFrame(current_predictions)

        # Get only most current predictions of most recent timestamp
        current_predictions_df = current_predictions_df.filter(pl.col("timestamp") == current_predictions_df["timestamp"].max())

        # Sort by predicted_future_return_14d (descending order)
        current_predictions_df = current_predictions_df.sort("predicted_future_return_14d", descending=True)

        print("\nCurrent Predictions Sorted by Predicted (Descending):")
        print(current_predictions_df.head(20))
        
        logging.info("Validation complete!")
        return final_model, test_rmse, best_model_params, predictions_df, current_predictions_df
        
if __name__ == '__main__':
    # Load and preprocess the data
    data_np, scaler, poly_transform, selected_features = load_and_preprocess_data()

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

    # hyperparameter_grid = {
    #     'n_estimators': [100, 200, 300, 400],  
    #     'max_depth': [3, 5, 7, 9],  
    #     'learning_rate': [0.01, 0.03, 0.05, 0.1],  
    #     'gamma': [0, 0.1, 0.2],  
    #     'subsample': [0.8, 0.9, 1.0],  
    #     'colsample_bytree': [0.8, 0.9, 1.0],  
    #     'min_child_weight': [1, 3, 5],  
    #     'lambda': [0, 1, 2],  
    #     'alpha': [0, 0.5, 1]  
    # }

    # Run the cross-validation
    final_model, test_rmse, best_model_params, predictions_df, current_predictions_df = time_series_walk_forward_cv_xgboost_parallel(
        features=X_np,
        target=y_np,
        model_params=model_params,
        hyperparameter_grid=hyperparameter_grid,
        selected_features=selected_features,  
        mode='fixed',  
        n_folds=10,     
        validation_days=30,  
        min_training_days=90,  
        test_size=0.1,  
        gap_days=7      
    )