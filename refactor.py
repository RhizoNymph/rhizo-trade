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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataPreprocessor:
    """Handles data preprocessing including feature selection, polynomial transformation, and scaling."""
    def __init__(self, poly_degree=2, vif_threshold=5):
        self.poly_degree = poly_degree
        self.vif_threshold = vif_threshold
        self.scaler = None
        self.poly_transform = None
        self.selected_features = None

    def preprocess(self, X, for_training=True):
        """Preprocess the data."""
        # Feature selection
        if self.selected_features is None:
            self.selected_features = select_features_vif_polars(X.drop("timestamp"), threshold=self.vif_threshold)
            self.selected_features = ["timestamp"] + self.selected_features
        X_selected = X.select(self.selected_features)

        # Separate 'timestamp' from features to be transformed
        features_for_transform = [col for col in self.selected_features if col != 'timestamp']
        X_features = X_selected.select(features_for_transform)
        X_timestamp = X_selected.select("timestamp")

        # Convert to numpy for sklearn transformers
        X_features_np = X_features.to_numpy()
        X_timestamp_np = X_timestamp.to_numpy()

        # Apply polynomial features
        if self.poly_transform is None:
            self.poly_transform = PolynomialFeatures(degree=self.poly_degree, include_bias=False, interaction_only=False)
            X_poly = self.poly_transform.fit_transform(X_features_np)
        else:
            X_poly = self.poly_transform.transform(X_features_np)

        # Scale the features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_poly)
        else:
            X_scaled = self.scaler.transform(X_poly)

        # Combine scaled features with timestamp
        X_scaled_df = pl.DataFrame(X_scaled)
        timestamp_series = pl.Series("timestamp", X_timestamp_np.flatten().tolist())

        # Ensure the number of rows matches
        assert X_scaled.shape[0] == len(X_timestamp_np), "Mismatch in number of rows."

        # Get feature names after polynomial transformation
        poly_feature_names = self.poly_transform.get_feature_names_out(features_for_transform)

        # Assign the correct column names
        X_scaled_df.columns = list(poly_feature_names)

        return X_scaled_df

class ModelTrainer:
    """Handles model training, evaluation, and hyperparameter tuning."""
    def __init__(self, model_params, hyperparameter_grid):
        self.model_params = model_params
        self.hyperparameter_grid = hyperparameter_grid

    def train_and_evaluate(self, args):
        """Train and evaluate the model for a given fold and hyperparameters."""
        hyperparams, fold_idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold, start_time_fold, scaler = args

        logging.info(f"Starting fold {fold_idx + 1} evaluation with hyperparams: {hyperparams}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Scale the data using the provided scaler
        X_train_scaled = scaler.transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Update model parameters with hyperparameters
        current_model_params = self.model_params.copy()
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

class TimeSeriesCV:
    """Handles time series cross-validation and parallel processing."""
    def __init__(self, preprocessor, trainer, mode='dynamic', n_folds=5, validation_days=30, min_training_days=90, test_size=0.1, gap_days=7):
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.mode = mode
        self.n_folds = n_folds
        self.validation_days = validation_days
        self.min_training_days = min_training_days
        self.test_size = test_size
        self.gap_days = gap_days

    def run(self, X, y):
        """Run time series cross-validation."""
        logging.info("Starting time series walk-forward cross-validation")

        # Split into train-val and final test
        total_days = len(X)
        test_days = int(total_days * self.test_size)
        X_train_val = X[:-test_days]
        y_train_val = y[:-test_days]
        X_test = X[-test_days:]
        y_test = y[-test_days:]

        logging.info(f"Train-val size: {len(X_train_val)}, Test size: {len(X_test)}")

        # Calculate number of validation windows
        if self.mode == 'dynamic':
            total_train_val_days = len(X_train_val)
            n_splits = (total_train_val_days - self.min_training_days - self.gap_days) // self.validation_days
            logging.info(f"Using dynamic mode with {n_splits} validation windows of {self.validation_days} days each")
        elif self.mode == 'fixed':
            n_splits = self.n_folds
            logging.info(f"Using fixed mode with {n_splits} folds")
        else:
            raise ValueError("Invalid mode. Choose 'dynamic' or 'fixed'.")

        # Fit the scaler on the entire training data
        scaler = StandardScaler()
        scaler.fit(X_train_val)

        # Generate arguments for parallel processing
        args_list = self._generate_args_list(X_train_val, y_train_val, n_splits, scaler)

        # Run parallel processing
        with multiprocessing.Pool() as pool:
            logging.info("Starting parallel processing of folds")
            results = pool.map(self.trainer.train_and_evaluate, args_list)

        # Process results
        val_rmses = []
        best_val_rmse = float('inf')
        best_model_params = None
        all_predictions = []

        for val_rmse, hyperparams, fold_idx, duration, predictions in results:
            val_rmses.append(val_rmse)
            all_predictions.extend(predictions)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_params = hyperparams
                logging.info(f"New best model found! RMSE: {best_val_rmse:.4f}")
                logging.info(f"Best hyperparameters: {best_model_params}")

            print(f"Fold {fold_idx + 1}, Hyperparams: {hyperparams}, Val RMSE: {val_rmse:.4f}, Time: {duration:.2f}s")

        # Create predictions DataFrame and sort by timestamp
        predictions_df = pl.DataFrame(all_predictions)
        predictions_df = predictions_df.sort("timestamp")

        print("\nLatest Validation Predictions:")
        print(predictions_df.tail(10))

        # Train final model on all data except test set with early stopping
        logging.info("Training final model with best hyperparameters and early stopping")
        final_model_params = self.trainer.model_params.copy()
        final_model_params.update(best_model_params)
        final_model_params['early_stopping_rounds'] = 10  # Set early stopping rounds

        # Split the training data into training and validation sets for early stopping
        split_idx = int(0.8 * len(X_train_val))  # 80% training, 20% validation
        X_train_final = X_train_val[:split_idx]
        y_train_final = y_train_val[:split_idx]
        X_val_final = X_train_val[split_idx:]
        y_val_final = y_train_val[split_idx:]

        # Scale the data
        X_train_final_scaled = scaler.transform(X_train_final)
        X_val_final_scaled = scaler.transform(X_val_final)

        # Train the final model with early stopping
        final_model = xgb.XGBRegressor(**final_model_params)
        final_model.fit(
            X_train_final_scaled, y_train_final,
            eval_set=[(X_val_final_scaled, y_val_final)],
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
        y_test_pred = final_model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        logging.info(f"Final Test Set RMSE: {test_rmse:.4f}")
        logging.info(f"Final Test Set R²: {test_r2:.4f}")

        return final_model, test_rmse, best_model_params, predictions_df
if __name__ == '__main__':
    # Load and preprocess the data
    df = load_data()
    preprocessor = DataPreprocessor()
    X = df.select([
        "timestamp", 
        "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
        "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d",
        "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"
    ])
    y = df.select("future_return_14d").to_numpy().flatten()

    X_processed = preprocessor.preprocess(X)
    data = pl.concat([X_processed, pl.DataFrame(y, schema=["future_return_14d"])], how='horizontal')
    data_np = data.to_numpy()

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

    # Initialize trainer and cross-validator
    trainer = ModelTrainer(model_params, hyperparameter_grid)
    cv = TimeSeriesCV(preprocessor, trainer, mode='fixed', n_folds=10, validation_days=30, min_training_days=90, test_size=0.1, gap_days=7)

    # Run the cross-validation
    final_model, test_rmse, best_model_params, predictions_df = cv.run(X_np, y_np)