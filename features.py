import polars as pl
import numpy as np
import statsmodels.api as sm

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