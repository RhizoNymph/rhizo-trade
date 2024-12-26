import glob

from schema import expected_schema, enforce_schema
from preprocessing import *

def load_data(data_dir='data/velo/spot/binance/1d'):
    files = glob.glob(f'{data_dir}/*.parquet')
    df = pl.concat([
        enforce_schema(pl.read_parquet(f)) for f in files
    ])

    # Sort by coin and timestamp
    df = df.sort(['coin', 'timestamp'])

    df = df.group_by('coin').map_groups(lambda group: calculate_returns(group, frequency=data_dir[-1]))
    df = df.group_by('coin').map_groups(lambda group: calculate_trading_features(group))

    df = df.drop_nulls()

    return df