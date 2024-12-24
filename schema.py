import polars as pl

# Expected schema definition
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