from statsmodels.tsa.stattools import adfuller, kpss

def calculate_returns(group):
    # Future returns
    group['future_1d_return'] = group['close_price'].shift(-1) / group['close_price'] - 1
    group['future_3d_return'] = group['close_price'].shift(-3) / group['close_price'] - 1
    group['future_5d_return'] = group['close_price'].shift(-5) / group['close_price'] - 1
    group['future_7d_return'] = group['close_price'].shift(-7) / group['close_price'] - 1
    # Past returns
    group['return_3d'] = group['close_price'] / group['close_price'].shift(3) - 1
    group['return_5d'] = group['close_price'] / group['close_price'].shift(5) - 1
    group['return_7d'] = group['close_price'] / group['close_price'].shift(7) - 1
    group['return_14d'] = group['close_price'] / group['close_price'].shift(14) - 1
    group['return_30d'] = group['close_price'] / group['close_price'].shift(30) - 1
    group['return_60d'] = group['close_price'] / group['close_price'].shift(60) - 1
    return group

def select_features_vif(X, threshold=5):
    """
    Iteratively remove features with highest VIF until all features are below threshold
    """
    X = X.copy()
    features = X.columns.tolist()
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

def perform_stationarity_tests(data):
    """
    Perform ADF and KPSS tests with improved error handling
    """
    # Default error outputs
    error_output = {
        'Test Statistic': 'N/A',
        'p-value': 'N/A',
        'Critical Values': {'1%': 'N/A', '5%': 'N/A', '10%': 'N/A'}
    }

    # Check if data is valid
    if data is None or len(data.dropna()) < 2:
        return error_output, error_output

    try:
        adf_result = adfuller(data.dropna())
        adf_output = {
            'Test Statistic': f"{adf_result[0]:.4f}",
            'p-value': f"{adf_result[1]:.4f}",
            'Critical Values': {k: f"{v:.4f}" for k,v in adf_result[4].items()}
        }
    except Exception as e:
        print(f"ADF Test Error: {str(e)}")
        adf_output = error_output

    try:
        kpss_result = kpss(data.dropna(), regression='c', nlags='auto')
        p_value = kpss_result[1]
        p_value = "> 0.1" if p_value == 0.1 else f"{p_value:.4f}"

        kpss_output = {
            'Test Statistic': f"{kpss_result[0]:.4f}",
            'p-value': p_value,
            'Critical Values': {k: f"{v:.4f}" for k,v in kpss_result[3].items()}
        }
    except Exception as e:
        print(f"KPSS Test Error: {str(e)}")
        kpss_output = error_output

    return adf_output, kpss_output

def calculate_vwmas(df):
    """Calculate volume-weighted moving averages and normalized distances"""
    lookback_periods = [3, 5, 7, 14, 30, 60]

    for period in lookback_periods:
        # Calculate VWMA
        vwma = (df['close'] * df['dollar_volume']).rolling(period).sum() / \
               df['dollar_volume'].rolling(period).sum()

        # Calculate normalized distance from VWMA
        dist = (df['close'] - vwma) / vwma
        df[f'vwma_{period}d_dist'] = dist

    return df
