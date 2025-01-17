import os
from velodata import lib as velo
import polars as pl
from datetime import datetime, timedelta

API_KEY = "29fff919f0ae45b3bb30e6dc0663bdc0"
client = velo.client(API_KEY)

def handle_velo_spot(args):
    if args.exchange == 'all':
        data = get_all_spot_data(resolution=args.resolution)
    else:
        spots = client.get_spot()
        columns = client.get_spot_columns()
        spot_markets = [s for s in spots if s['exchange'] == args.exchange]

        data = {}
        for spot in spot_markets:
            try:
                df = get_market_data(
                    "spot",
                    columns,
                    [args.exchange],
                    [spot['product']],
                    args.lookback,
                    args.resolution
                )

                if not df.is_empty():
                    data[f"{args.exchange}:{spot['product']}"] = df
            except Exception as e:
                print(f"Error processing market {spot['product']}: {str(e)}")
                print(f"Full error: {repr(e)}")

        # Save/update data
        output_dir = f"data/velo/spot/{args.exchange}/{args.resolution}"
        os.makedirs(output_dir, exist_ok=True)

        for market, df in data.items():
            file_path = f"{output_dir}/{market}.parquet"
            if os.path.exists(file_path):
                # Read existing data
                existing_df = pl.read_parquet(file_path)
                # Combine with new data and remove duplicates
                combined_df = pl.concat([existing_df, df]).unique(subset=['timestamp'])
                combined_df.write_parquet(file_path)
            else:
                df.write_parquet(file_path)

def handle_velo_futures(args):
    if args.exchange == 'all':
        data = get_all_futures_data(resolution=args.resolution)
    else:
        futures = client.get_futures()
        columns = client.get_futures_columns()
        future_markets = [f for f in futures if f['exchange'] == args.exchange]

        data = {}
        for future in future_markets:
            df = get_market_data(
                "futures",
                columns,
                [args.exchange],
                [future['product']],
                args.lookback,
                args.resolution
            )
            if not df.is_empty():
                data[f"{args.exchange}:{future['product']}"] = df

    # Save/update data
    output_dir = f"data/velo/futures/{args.exchange}/{args.resolution}"
    os.makedirs(output_dir, exist_ok=True)

    for market, df in data.items():
        file_path = f"{output_dir}/{market}.parquet"
        if os.path.exists(file_path):
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df]).unique(subset=['timestamp'])
            combined_df.write_parquet(file_path)
        else:
            df.write_parquet(file_path)

def handle_velo_options(args):
    if args.exchange == 'all':
        data = get_all_options_data(resolution=args.resolution)
    else:
        options = client.get_options()
        columns = client.get_options_columns()
        option_markets = [o for o in options if o['exchange'] == args.exchange]

        data = {}
        for option in option_markets:
            df = get_market_data(
                "options",
                columns,
                [args.exchange],
                [option['product']],
                args.lookback,
                args.resolution
            )
            if not df.is_empty():
                data[f"{args.exchange}:{option['product']}"] = df

    # Save/update data
    output_dir = f"data/velo/options/{args.exchange}/{args.resolution}"
    os.makedirs(output_dir, exist_ok=True)

    for market, df in data.items():
        file_path = f"{output_dir}/{market}.parquet"
        if os.path.exists(file_path):
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df]).unique(subset=['timestamp'])
            combined_df.write_parquet(file_path)
        else:
            df.write_parquet(file_path)

def get_market_data(type, columns, exchanges, products, lookback_hours, resolution):
    now = client.timestamp()
    start = now - (lookback_hours * 60 * 60 * 1000)

    params = {
        "type": type,
        "columns": columns,
        "exchanges": exchanges,
        "products": products,
        "begin": start,
        "end": now,
        "resolution": resolution
    }

    try:
        df = pl.DataFrame(client.get_rows(params))
        if not df.is_empty():
            df = df.rename({"time": "timestamp"})
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Int64)
            )
        return df
    except Exception as e:
        print(f"Error fetching {type} data for {exchanges[0]}:{products[0]}: {e}")
        return pl.DataFrame()

def get_all_futures_data(resolution="1h"):
    futures = client.get_futures()
    columns = client.get_futures_columns()

    all_data = {}
    for future in futures:
        df = get_market_data(
            "futures",
            columns,
            [future["exchange"]],
            [future["product"]],
            args.lookback,
            resolution
        )
        if not df.is_empty():
            all_data[f"{future['exchange']}:{future['product']}"] = df

    return all_data

def get_all_spot_data(resolution="1h"):
    spots = client.get_spot()
    columns = client.get_spot_columns()

    all_data = {}
    for spot in spots:
        df = get_market_data(
            "spot",
            columns,
            [spot["exchange"]],
            [spot["product"]],
            args.lookback,
            resolution
        )
        if not df.is_empty():
            all_data[f"{spot['exchange']}:{spot['product']}"] = df

    return all_data

def get_all_options_data(resolution="1h"):
    options = client.get_options()
    columns = client.get_options_columns()

    all_data = {}
    for option in options:
        df = get_market_data(
            "options",
            columns,
            [option["exchange"]],
            [option["product"]],
            args.lookback,
            resolution
        )
        if not df.is_empty():
            all_data[f"{option['exchange']}:{option['product']}"] = df

    return all_data
