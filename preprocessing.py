import polars as pl

import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.filters import hp_filter

def calculate_returns(group, future_periods=[1, 3, 5, 7, 14, 30], past_periods=[1, 3, 5, 7, 14, 30], frequency='d'):
    # Calculate future returns
    for period in future_periods:
        group = group.with_columns(
            (pl.col('close_price').shift(-period) / pl.col('close_price') - 1).alias(f'future_return_{period}{frequency}')
        )

    # Calculate past returns
    for period in past_periods:
        group = group.with_columns(
            (pl.col('close_price') / pl.col('close_price').shift(period) - 1).alias(f'return_{period}{frequency}')

        )   

    # Calculate standard deviation of past returns
    for period in past_periods[1:]:
        group = group.with_columns(
            pl.col(f'return_{period}{frequency}').rolling_std(period).alias(f'return_std_{period}{frequency}')
        )

    return group

def calculate_trading_features(group):
    group = group.with_columns(
        (pl.col('buy_coin_volume') / pl.col('sell_coin_volume')).alias('coin_volume_bs_ratio'),
        (pl.col('buy_trades') / pl.col('sell_trades')).alias('trades_bs_ratio'),
        (pl.col('buy_coin_volume') + pl.col('sell_coin_volume')).alias('total_coin_volume'),
        (pl.col('buy_trades') + pl.col('sell_trades')).alias('total_trades'),
    )

    return group

def kpss_test(series):
    kpsstest = kpss(series, regression="c", nlags="auto")
    kpss_output = {
        "Test Statistic": kpsstest[0],
        "p-value": kpsstest[1]
    }
    for key, value in kpsstest[3].items():
        if key == "5%":
            kpss_output[f"Critical Value ({key})"] = value
    return kpss_output

def adf_test(series):
    dftest = adfuller(series, autolag="AIC")
    dfoutput = {
        "Test Statistic": dftest[0],
        "p-value": dftest[1]
    }
    for key, value in dftest[4].items():
        if key == "5%":
            dfoutput[f"Critical Value ({key})"] = value
    return dfoutput

def run_stationarity_tests(df: pl.DataFrame) -> pl.DataFrame:
    results = []

    for column in df.columns:
        series = df[column].to_numpy()  # Convert Polars Series to numpy array
        kpss_result = kpss_test(series)
        adf_result = adf_test(series)

        # Combine results into a single dictionary
        combined_result = {
            "Feature": column,
            **{f"KPSS {key}": value for key, value in kpss_result.items()},
            **{f"ADF {key}": value for key, value in adf_result.items()},
        }

        # Add interpretation based on p-values
        kpss_p_value = kpss_result["p-value"]
        adf_p_value = adf_result["p-value"]

        # Interpretation logic
        if kpss_p_value < 0.05 and adf_p_value < 0.05:
            interpretation = "Non-Stationary"
        elif kpss_p_value >= 0.05 and adf_p_value >= 0.05:
            interpretation = "Stationary"
        elif kpss_p_value >= 0.05 and adf_p_value < 0.05:
            interpretation = "Trend Stationary"
        elif kpss_p_value < 0.05 and adf_p_value >= 0.05:
            interpretation = "Difference Stationary"
        else:
            interpretation = "Unknown"

        combined_result["Interpretation"] = interpretation

        results.append(combined_result)

    # Convert results to a Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df

def difference_dataframe(df: pl.DataFrame, periods: int = 1) -> pl.DataFrame:
    differenced_df = df.select(
        [
            (pl.col(col) - pl.col(col).shift(periods)).alias(col)
            for col in df.columns
        ]
    )

    # Drop rows with null values introduced by shifting
    differenced_df = differenced_df.drop_nulls()

    return differenced_df

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