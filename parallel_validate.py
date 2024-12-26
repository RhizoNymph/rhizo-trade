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
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'gamma': [0, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

def train_and_evaluate(args):
    hyperparams, fold_idx, X_train_fold_cpu, y_train_fold_cpu, X_train_val, y_train_val, start_val, end_val, model_params, start_time_fold = args
    fold_train_rmses = []
    fold_val_rmses = []

    
    X_train_fold = X_train_fold_cpu.clone().detach().to('cuda')
    y_train_fold = y_train_fold_cpu.clone().detach().to('cuda')
    for j in range(len(X_train_val[start_val:end_val])):
        
        X_val_point = X_train_val[start_val + j:start_val + j + 1].to('cuda')  
        y_val_point = y_train_val[start_val + j:start_val + j + 1].to('cuda')

        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold.cpu())
        X_val_scaled = scaler_X.transform(X_val_point.cpu())

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_fold.cpu())

        
        X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device='cuda')
        X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32, device='cuda')
        y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32, device='cuda')

        
        current_model_params = model_params.copy()
        current_model_params.update(hyperparams)

        model = xgb.XGBRegressor(**current_model_params)
        model.fit(X_train_scaled, y_train_scaled.ravel()) 

        
        y_train_pred = model.predict(X_train_scaled) 
        y_val_pred = model.predict(X_val_scaled) 

        
        y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_val_pred = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        y_train_fold_rescaled = scaler_y.inverse_transform(y_train_scaled.cpu()).flatten()

        
        y_train_pred = torch.tensor(y_train_pred, dtype=torch.float32, device='cuda')
        y_val_pred = torch.tensor(y_val_pred, dtype=torch.float32, device='cuda')
        y_train_fold_rescaled = torch.tensor(y_train_fold_rescaled, dtype=torch.float32, device='cuda')

        
        train_rmse = np.sqrt(mean_squared_error(y_train_fold_rescaled.cpu(), y_train_pred.cpu()))
        val_rmse = np.sqrt(mean_squared_error(y_val_point.cpu(), y_val_pred.cpu())) 

        
        X_train_fold = torch.cat([X_train_fold, X_val_point])  
        y_train_fold = torch.cat([y_train_fold, y_val_point])

        fold_train_rmses.append(train_rmse)
        fold_val_rmses.append(val_rmse)

    
    avg_fold_train_rmse = np.mean(fold_train_rmses)
    avg_fold_val_rmse = np.mean(fold_val_rmses)

    return avg_fold_train_rmse, avg_fold_val_rmse, hyperparams, fold_idx, time.time() - start_time_fold

def time_series_walk_forward_cv_xgboost_parallel(features, target, model_params, hyperparameter_grid, n_splits, test_size):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=test_size, shuffle=False
    )

    X_train_val = torch.tensor(X_train_val, dtype=torch.float32, device='cpu')
    y_train_val = torch.tensor(y_train_val, dtype=torch.float32, device='cpu').view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32, device='cpu')
    y_test = torch.tensor(y_test, dtype=torch.float32, device='cpu').view(-1, 1)
 
    n_train_val = len(X_train_val)
    fold_size = n_train_val // n_splits
    
    if fold_size == 0:
        raise ValueError("Fold size cannot be zero. Increase n_train_val or decrease n_splits.")
  
    train_rmses = []
    val_rmses = []
    best_avg_val_rmse = float('inf')
    best_model_params = None
    
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    args_list = []
    for fold_idx in range(n_splits):
        start_train = 0
        end_train = max(1, fold_idx * fold_size)
        start_val = end_train
        end_val = start_val + fold_size
        X_train_fold_cpu = X_train_val[start_train:end_train].clone().detach()
        y_train_fold_cpu = y_train_val[start_train:end_train].clone().detach()

        for hyperparam_idx, hyperparams in enumerate(hyperparameter_combinations):
            start_time_fold = time.time()
            args_list.append((hyperparams, fold_idx, X_train_fold_cpu, y_train_fold_cpu, X_train_val, y_train_val, start_val, end_val, model_params, start_time_fold))
    
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn')

        with multiprocessing.Pool() as pool:
            results = pool.map(train_and_evaluate, args_list)

        
        for avg_fold_train_rmse, avg_fold_val_rmse, hyperparams, fold_idx, duration in results:
            train_rmses.append(avg_fold_train_rmse)
            val_rmses.append(avg_fold_val_rmse)

            if avg_fold_val_rmse < best_avg_val_rmse:
                best_avg_val_rmse = avg_fold_val_rmse
                best_model_params = hyperparams

            print(f"Fold {fold_idx+1}, Hyperparams: {hyperparams}, Avg Train RMSE: {avg_fold_train_rmse:.4f}, Avg Val RMSE: {avg_fold_val_rmse:.4f}, Time: {duration:.2f}s")

        
        scaler_X_final = StandardScaler()
        X_train_val_scaled = scaler_X_final.fit_transform(X_train_val.cpu())
        X_test_scaled = scaler_X_final.transform(X_test.cpu())

        scaler_y_final = StandardScaler()
        y_train_val_scaled = scaler_y_final.fit_transform(y_train_val.cpu())

        
        X_train_val_scaled = torch.tensor(X_train_val_scaled, dtype=torch.float32, device='cuda')
        X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device='cuda')
        y_train_val_scaled = torch.tensor(y_train_val_scaled, dtype=torch.float32, device='cuda')

        best_model = xgb.XGBRegressor(**best_model_params)
        best_model.fit(X_train_val_scaled, y_train_val_scaled.ravel())

        y_test_pred = best_model.predict(X_test_scaled)

        y_test_pred = scaler_y_final.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

        
        y_test_pred = torch.tensor(y_test_pred, dtype=torch.float32, device='cuda')

        test_rmse = np.sqrt(mean_squared_error(y_test.cpu(), y_test_pred.cpu()))

        
        avg_train_rmse = np.mean(train_rmses)
        avg_val_rmse = np.mean(val_rmses)

        print("\nXGBoost Cross-Validation Results:")
        print(f"  Average Training RMSE: {avg_train_rmse:.4f}")
        print(f"  Average Validation RMSE: {avg_val_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Best Hyperparameters: {best_model_params}")
        print(f"  Best Average Validation RMSE: {best_avg_val_rmse:.4f}")

time_series_walk_forward_cv_xgboost_parallel(
    features=X_np,
    target=y_np,
    model_params=model_params,
    hyperparameter_grid=hyperparameter_grid,
    n_splits=5,
    test_size=0.2
)