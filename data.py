import glob

from schema import expected_schema, enforce_schema
from preprocessing import *

def load_data(data_dir='data/velo/spot/binance/1d', for_training=True):
    files = glob.glob(f'{data_dir}/*.parquet')
    df = pl.concat([enforce_schema(pl.read_parquet(f)) for f in files])
    df = df.filter(~pl.col('coin').is_in(['BTC', 'USDT', 'USDC']))
    
    # Sort by coin and timestamp
    df = df.sort(['coin', 'timestamp'])

    # Calculate returns and trading features
    df = df.group_by('coin').map_groups(lambda group: calculate_returns(group, frequency=data_dir[-1]))
    df = df.group_by('coin').map_groups(lambda group: calculate_trading_features(group))    

    # For training, calculate future_return_14d and drop nulls
    if for_training:
        df = df.with_columns(
            (pl.col('close_price').shift(-14) / pl.col('close_price') - 1).alias('future_return_14d')
        )
        df = df.drop_nulls()

    return df