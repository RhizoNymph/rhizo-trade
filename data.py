import glob
import os
import polars as pl
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
import numpy as np
from datetime import datetime

# def load_and_preprocess_data():
#     """Load and preprocess the data."""
#     df = load_data()
    
#     X = df.select([
#                     "timestamp", 
#                     "return_1d", "return_3d", "return_5d", "return_7d", "return_14d", "return_30d",
#                     "return_std_3d", "return_std_5d", "return_std_7d", "return_std_14d", "return_std_30d",
#                     #'vwma_3d', 'vwma_3d_dist', 'vwma_5d', 'vwma_5d_dist', 'vwma_7d', 'vwma_7d_dist', 'vwma_14d', 'vwma_14d_dist', 'vwma_30d', 'vwma_30d_dist',
#                     "coin_volume_bs_ratio", "trades_bs_ratio", "total_coin_volume", "total_trades"
#                 ])
#     y = df.select("future_return_7d")

#     # Preprocess the data
#     X_processed, scaler, poly_transform, selected_features = preprocess_data(X)

#     # Combine features and target
#     data = pl.concat([pl.DataFrame(X_processed), y], how='horizontal')
    
#     return data.to_numpy(), scaler, poly_transform, selected_features

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

def load_data(data_dir='data/velo/spot/binance/1d', for_training=True):
    files = glob.glob(f'{data_dir}/*.parquet')
    df = pl.concat([enforce_schema(pl.read_parquet(f)) for f in files])
    df = df.filter(~pl.col('coin').is_in(['USDT', 'USDC']))
    
    # Sort by coin and timestamp
    df = df.sort(['coin', 'timestamp'])

    # Calculate returns and trading features
    df = df.group_by('coin').map_groups(lambda group: calculate_returns(group, frequency=data_dir[-1]))
    df = df.group_by('coin').map_groups(lambda group: calculate_trading_features(group))    
    df = df.group_by('coin').map_groups(lambda group: calculate_time_features(group))

    # For training, calculate future_return_14d and drop nulls
    if for_training:
        df = df.with_columns(
            (pl.col('close_price').shift(-7) / pl.col('close_price') - 1).alias('future_return_7d')
        )
        df = df.drop_nulls()

    return df

expected_schema = {
    "exchange": pl.Utf8,
    "coin": pl.Utf8,
    "product": pl.Utf8,
    "timestamp": pl.Int64,
    "open_price": pl.Float64,
    "high_price": pl.Float64,
    "low_price": pl.Float64,
    "close_price": pl.Float64,
    "coin_volume": pl.Float64,
    "dollar_volume": pl.Float64,
    "buy_trades": pl.Int64,
    "sell_trades": pl.Int64,
    "total_trades": pl.Int64,
    "buy_coin_volume": pl.Float64,
    "sell_coin_volume": pl.Float64,
    "buy_dollar_volume": pl.Float64,
    "sell_dollar_volume": pl.Float64,
}

# Function to enforce schema
def enforce_schema(df):
    return df.select(
        [pl.col(col_name).cast(col_type) for col_name, col_type in expected_schema.items()]
    )

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

    # for period in past_periods[1:]:
    #     group = group.tail(period).with_columns(
    #         (pl.col(f'return_{period}{frequency}').mean().alias(f'avg_return_{period}{frequency}'))
    #     )

    # Calculate standard deviation of past returns
    for period in past_periods[1:]:
        group = group.with_columns(
            pl.col(f'return_{period}{frequency}').rolling_std(period).alias(f'return_std_{period}{frequency}')
        )

    return group

def calculate_trading_features(group):
    if 'buy_coin_volume' in group.columns:
        group = group.with_columns(
            (pl.col('buy_coin_volume') / pl.col('sell_coin_volume')).alias('coin_volume_bs_ratio'),
            (pl.col('buy_trades') / pl.col('sell_trades')).alias('trades_bs_ratio'),
            (pl.col('buy_coin_volume') + pl.col('sell_coin_volume')).alias('total_coin_volume'),
            (pl.col('buy_trades') + pl.col('sell_trades')).alias('total_trades'),
        )
    
    return group

def calculate_time_features(group):
    # Convert Unix timestamp (in milliseconds) to datetime
    group = group.with_columns(
        pl.col('timestamp').map_elements(
            lambda x: datetime.utcfromtimestamp(x / 1000), return_dtype=pl.Datetime
        ).alias('datetime')
    )

    # Extract time features
    group = group.with_columns([
        pl.col("datetime").dt.weekday().alias("day_of_week"),
        pl.col("datetime").dt.day().alias("day_of_month"),
        pl.col("datetime").dt.month().alias("month_of_year")
    ])

    return group


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