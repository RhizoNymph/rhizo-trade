import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import xgboost as xgb
import numpy as np

# Create a simple dummy dataset
X = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
y = pd.Series(np.random.rand(100))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define a simple hyperparameter grid
param_grid = {
    'max_depth': [3],
    'learning_rate': [0.1],
    'n_estimators': [50]
}

# Define custom scorer (simplified)
def custom_scorer(y_true, y_pred):
    return -np.mean(np.abs(y_true - y_pred))

# Create a scorer using make_scorer
custom_scorer = make_scorer(custom_scorer)

# Set up TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring=custom_scorer, cv=tscv, verbose=2, n_jobs=-1)

# Fit the model (this is where the error was occurring)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)