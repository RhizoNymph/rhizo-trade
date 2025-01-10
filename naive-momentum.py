import glob
import polars as pl
import vectorbt as vbt
import numpy as np

pl.Config.set_tbl_rows(100)
pl.Config(tbl_cols=10)


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

def enforce_schema(df):
    return df.select(
        [pl.col(col_name).cast(col_type) for col_name, col_type in expected_schema.items()]
    )

data_dir='data/velo/spot/binance/1d'
files = glob.glob(f'{data_dir}/*.parquet')
df = pl.concat([enforce_schema(pl.read_parquet(f)) for f in files])
df = df.filter(~pl.col('coin').is_in(['USDT', 'USDC']))

# Sort by coin and timestamp
df = df.sort(['coin', 'timestamp'])
df = df.group_by('coin').map_groups(lambda group: calculate_returns(group, frequency=data_dir[-1]))
df = df.filter(pl.col('product').str.tail(3)!='USDT')

# Create a bi-weekly timestamp
df = df.with_columns(
    (pl.col("timestamp") - (pl.col("timestamp") % (7 * 24 * 60 * 60 * 1000))).alias("biweekly_timestamp")
)

# Rank stocks by 30d_return within each bi-weekly timestamp
ranked_df = df.with_columns(
    pl.col("return_30d")
    .rank(descending=True, method="ordinal")
    .over("biweekly_timestamp")
    .alias("rank")
)

def generate_signals(df):
    # Calculate the decile threshold
    decile_threshold = pl.col('rank').max() // 10

    # Generate entry and exit signals only at the start of each bi-weekly period
    return pl.when(pl.col('timestamp') == pl.col('biweekly_timestamp')) \
             .then(pl.when(pl.col('rank') <= decile_threshold) \
                   .then(1)  # Buy signal when entering top decile
                   .when(pl.col('rank') > decile_threshold) \
                   .then(-1)  # Sell signal when exiting top decile
                   .otherwise(0)) \
             .otherwise(0)

ranked_df = ranked_df.with_columns(
    generate_signals(ranked_df).alias('signal')
)
# Prepare data for vectorbt
close_prices = ranked_df.pivot(
    index="timestamp", on="coin", values="close_price"
)

entries = ranked_df.pivot(
    index="timestamp", on="coin", values="signal"
).select(
    [
        pl.when(pl.col(col) == 1)
        .then(True)
        .otherwise(False)
        .alias(f"{col}_entry")
        for col in ranked_df.select("coin").unique().sort("coin").to_series()
    ]
)

exits = ranked_df.pivot(
    index="timestamp", on="coin", values="signal"
).select(
    [
        pl.when(pl.col(col) == -1)
        .then(True)
        .otherwise(False)
        .alias(f"{col}_exit")
        for col in ranked_df.select("coin").unique().sort("coin").to_series()
    ]
)

# Convert Polars DataFrames to Pandas DataFrames
close_prices_pd = close_prices.to_pandas().set_index("timestamp")
entries_pd = entries.to_pandas().set_index(close_prices_pd.index)
exits_pd = exits.to_pandas().set_index(close_prices_pd.index)

# Create the vectorbt portfolio
pf = vbt.Portfolio.from_signals(
    close_prices_pd,
    entries_pd,
    exits_pd,
    freq="1W",
    fees=0.001,
    slippage=0.001
)

# Print the portfolio statistics
print(pf.stats())