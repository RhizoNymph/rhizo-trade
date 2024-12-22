from statsmodels.regression.linear_model import WLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

import statsmodels.api as sm
import pandas as pd
import numpy as np

def calculate_vif(X):
    """
    Calculates Variance Inflation Factor for each feature
    """
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i)
                       for i in range(X_const.shape[1])]
    return vif_data

def select_features_vif(X, threshold=5):
    """
    Iteratively remove features with highest VIF until all features are below threshold
    """
    features = X.columns
    excluded_features = []

    while True:
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                          for i in range(X_with_const.shape[1])]

        max_vif = vif_data.loc[vif_data['Feature'] != 'const', 'VIF'].max()

        if max_vif < threshold:
            break

        feature_to_remove = vif_data.loc[
            (vif_data['Feature'] != 'const') & (vif_data['VIF'] == max_vif),
            'Feature'
        ].iloc[0]

        X = X.drop(columns=[feature_to_remove])
        excluded_features.append(feature_to_remove)

    return X.columns.tolist(), excluded_features

def perform_wls_regression(X, y):
    """
    Perform Weighted Least Squares regression with error handling
    """
    try:
        X_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_const).fit()
        weights = 1 / (np.abs(ols_model.resid) + 1e-6)  # Added small constant to avoid division by zero
        wls_model = WLS(y, X_const, weights=weights).fit()
        return wls_model
    except Exception as e:
        print(f"WLS Regression Error: {str(e)}")
        return None

def perform_robust_regression(X, y):
    """
    Perform robust regression with HC3 covariance type
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type='HC3')
    return model

def perform_heteroskedasticity_test(X, y):
    """
    Performs Breusch-Pagan test for heteroskedasticity with error handling
    """
    try:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        bp_test = het_breuschpagan(model.resid, X_const)

        return {
            'lm_stat': f"{bp_test[0]:.4f}",
            'p_value': f"{bp_test[1]:.4f}",
            'f_stat': f"{bp_test[2]:.4f}",
            'f_p_value': f"{bp_test[3]:.4f}"
        }
    except Exception as e:
        print(f"Heteroskedasticity Test Error: {str(e)}")
        return {
            'lm_stat': 'N/A',
            'p_value': 'N/A',
            'f_stat': 'N/A',
            'f_p_value': 'N/A'
        }
