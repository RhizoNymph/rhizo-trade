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
import multiprocessing
from itertools import product
import time
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from preprocessing import *
from data import load_data

df = load_data()

X = df.select(["timestamp", "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d", "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"])
y = df.select("future_return_14d")

selected_features = select_features_vif_polars(X.drop("timestamp"), threshold=5)
selected_features = ["timestamp"] + selected_features  
X_selected = X.select(selected_features)

features_for_scaling = [col for col in selected_features if col != 'timestamp']

interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_interaction = interaction.fit_transform(X_selected.select(features_for_scaling))
feature_names = interaction.get_feature_names_out(features_for_scaling)

X_interaction = pl.DataFrame(X_interaction).rename(
    {f"column_{i}": name for i, name in enumerate(feature_names)}
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_interaction)

X_scaled = pl.DataFrame(X_scaled, schema=X_interaction.columns)

X_scaled = X_scaled.with_columns(X_selected.select("timestamp"))

data = pl.concat([X_scaled, y], how='horizontal')

data_np = data.to_numpy()

X_np = data_np[:, :-1]  
y_np = data_np[:, -1]   

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

def train_and_evaluate(args):
    hyperparams, fold_idx, X_train_fold_cpu, y_train_fold_cpu, X_val_fold_cpu, y_val_fold_cpu, model_params, start_time_fold = args
    
    logging.info(f"Starting fold {fold_idx+1} evaluation with hyperparams: {hyperparams}")
    
    # Clear GPU cache at the start of processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process in batches to manage memory
    batch_size = 1024  # Adjust this based on your GPU memory
    
    def process_in_batches(X, y, scaler=None):
        if scaler:
            X = scaler.transform(X.cpu().numpy())
            X = torch.tensor(X, dtype=torch.float32)
        
        results = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size].cuda()
            batch_y = y[i:i + batch_size].cuda() if y is not None else None
            results.append((batch_X, batch_y))
            # Move back to CPU to free GPU memory
            if i + batch_size < len(X):
                torch.cuda.empty_cache()
        return results

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on CPU
    X_train_np = X_train_fold_cpu.numpy()
    y_train_np = y_train_fold_cpu.numpy().reshape(-1, 1)
    
    scaler_X.fit(X_train_np)
    scaler_y.fit(y_train_np)
    
    # Process validation data
    X_val_batches = process_in_batches(X_val_fold_cpu, None, scaler_X)
    
    # Train model with early stopping
    current_model_params = model_params.copy()
    current_model_params.update(hyperparams)
    current_model_params['early_stopping_rounds'] = 5
    
    model = xgb.XGBRegressor(**current_model_params)
    
    # Train on CPU since XGBoost handles its own GPU memory
    X_train_scaled = scaler_X.transform(X_train_np)
    y_train_scaled = scaler_y.transform(y_train_np)
    
    model.fit(
        X_train_scaled,
        y_train_scaled.ravel(),
        eval_set=[(X_train_scaled, y_train_scaled.ravel())],
        verbose=False
    )
    
    # Make predictions in batches
    all_preds = []
    for X_val_batch, _ in X_val_batches:
        batch_pred = model.predict(X_val_batch.cpu().numpy())
        batch_pred = scaler_y.inverse_transform(batch_pred.reshape(-1, 1)).flatten()
        all_preds.append(torch.tensor(batch_pred, dtype=torch.float32).cuda())
        
    y_val_pred = torch.cat(all_preds)
    
    # Calculate RMSE
    val_rmse = np.sqrt(mean_squared_error(y_val_fold_cpu.cpu(), y_val_pred.cpu()))
    
    # Store predictions
    predictions = [{
        'timestamp': X_val_fold_cpu[i, -1].cpu().item(),
        'actual': y_val_fold_cpu[i].cpu().item(),
        'predicted': y_val_pred[i].cpu().item(),
        'rmse': val_rmse
    } for i in range(len(y_val_fold_cpu))]

    return val_rmse, hyperparams, fold_idx, time.time() - start_time_fold, predictions

def time_series_walk_forward_cv_xgboost_parallel(features, target, model_params, hyperparameter_grid, validation_days=30, min_training_days=90, test_size=0.1):
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
    total_train_val_days = len(X_train_val)
    n_splits = (total_train_val_days - min_training_days) // validation_days
    
    logging.info(f"Using {n_splits} validation windows of {validation_days} days each")
    logging.info(f"Minimum training size: {min_training_days} days")
    
    val_rmses = []
    best_val_rmse = float('inf')
    best_model_params = None
    all_predictions = []
    
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
    logging.info(f"Testing {len(hyperparameter_combinations)} hyperparameter combinations")
    
    args_list = []
    for fold_idx in range(n_splits):
        # Calculate window boundaries
        end_val = total_train_val_days - (n_splits - fold_idx - 1) * validation_days
        start_val = end_val - validation_days
        start_train = 0  # Always start from beginning
        end_train = start_val
        
        if end_train - start_train < min_training_days:
            continue
            
        X_train_fold = torch.tensor(X_train_val[start_train:end_train], dtype=torch.float32, device='cpu')
        y_train_fold = torch.tensor(y_train_val[start_train:end_train], dtype=torch.float32, device='cpu')
        X_val_fold = torch.tensor(X_train_val[start_val:end_val], dtype=torch.float32, device='cpu')
        y_val_fold = torch.tensor(y_train_val[start_val:end_val], dtype=torch.float32, device='cpu')

        for hyperparams in hyperparameter_combinations:
            start_time_fold = time.time()
            args_list.append((hyperparams, fold_idx, X_train_fold, y_train_fold, 
                            X_val_fold, y_val_fold, model_params, start_time_fold))
    
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
        X_train_val_tensor = torch.tensor(X_train_val, dtype=torch.float32, device='cuda')
        y_train_val_tensor = torch.tensor(y_train_val, dtype=torch.float32, device='cuda')
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device='cuda')
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device='cuda')

        # Scale all data
        scaler_X_final = StandardScaler()
        X_train_val_scaled = scaler_X_final.fit_transform(X_train_val_tensor.cpu())
        X_test_scaled = scaler_X_final.transform(X_test_tensor.cpu())

        scaler_y_final = StandardScaler()
        y_train_val_scaled = scaler_y_final.fit_transform(y_train_val_tensor.cpu())

        # Convert back to GPU tensors
        X_train_val_scaled = torch.tensor(X_train_val_scaled, dtype=torch.float32, device='cuda')
        X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device='cuda')
        y_train_val_scaled = torch.tensor(y_train_val_scaled, dtype=torch.float32, device='cuda')

        # Train final model
        final_model_params = model_params.copy()
        final_model_params.update(best_model_params)
        final_model_params['early_stopping_rounds'] = 10
        
        final_model = xgb.XGBRegressor(**final_model_params)
        
        # Split for early stopping in final model
        split_idx = int(len(X_train_val_scaled) * 0.9)
        X_train_final = X_train_val_scaled[:split_idx]
        X_early_stop_final = X_train_val_scaled[split_idx:]
        y_train_final = y_train_val_scaled[:split_idx]
        y_early_stop_final = y_train_val_scaled[split_idx:]
        
        final_model.fit(
            X_train_final, y_train_final.ravel(),
            eval_set=[(X_early_stop_final, y_early_stop_final.ravel())],
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
        y_test_pred = final_model.predict(X_test_scaled)
        y_test_pred = scaler_y_final.inverse_transform(y_test_pred.reshape(-1, 1))
        test_rmse = np.sqrt(mean_squared_error(y_test_tensor.cpu(), y_test_pred))
        
        logging.info(f"Final Test Set RMSE: {test_rmse:.4f}")
        
        # Create test set predictions DataFrame
        test_predictions = {
            'timestamp': X_test_tensor[:, -1].cpu().numpy(),
            'actual': y_test_tensor.cpu().numpy().flatten(),
            'predicted': y_test_pred.flatten(),
            'rmse': [test_rmse] * len(y_test)
        }
        test_predictions_df = pl.DataFrame(test_predictions)
        test_predictions_df = test_predictions_df.sort("timestamp")
        
        print("\nLatest Test Set Predictions:")
        print(test_predictions_df.tail(10))
        
        # Save predictions
        predictions_path = os.path.join(model_dir, f"predictions_{timestamp}.csv")
        test_predictions_df.write_csv(predictions_path)
        logging.info(f"Saved predictions to {predictions_path}")
        
        logging.info("Validation complete!")
        return final_model, test_rmse, best_model_params, predictions_df, test_predictions_df

time_series_walk_forward_cv_xgboost_parallel(
    features=X_np,
    target=y_np,
    model_params=model_params,
    hyperparameter_grid=hyperparameter_grid,
    validation_days=30,
    min_training_days=90,
    test_size=0.1
)