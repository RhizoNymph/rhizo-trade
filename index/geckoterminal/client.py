import os
import aiohttp
import asyncio
import polars as pl
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from datetime import datetime, timedelta

class GeckoTerminalClient:
    BASE_URL = "https://api.geckoterminal.com/api/v2"

    def __init__(self):
        self.rate_limiter = AsyncLimiter(1, 2)
        self.session = None

    async def __aenter__(self):
        self.session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_pools(self, network, pool_type="trending", page=1):
        """Get pools by type (top, trending, or new)"""
        async with self.rate_limiter:
            if pool_type == "new":
                endpoint = f"/networks/{network}/new_pools"
            elif pool_type == "trending":
                endpoint = f"/networks/{network}/trending_pools"
            else:  # top
                endpoint = f"/networks/{network}/pools"

            url = f"{self.BASE_URL}{endpoint}?page={page}"
            #print(f"Fetching pools: {url}")

            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'data' in data:
                            return data
                        else:
                            print(f"No data in response for page {page}")
                            return None
                    elif response.status == 429:  # Rate limit
                        print(f"Rate limit hit, waiting...")
                        await asyncio.sleep(2)
                        return await self.get_pools(network, pool_type, page)
                    else:
                        print(f"Error {response.status} fetching pools for {network} page {page}")
                        return None
            except Exception as e:
                print(f"Exception fetching pools: {e}")
                return None

    async def get_ohlcv(self, pool_address, network, timeframe="hour", before=None):
        """Get OHLCV data for a specific pool"""
        async with self.rate_limiter:
            clean_address = pool_address.replace(f"{network}_", "")

            if before is None:
                before = int(datetime.now().timestamp())
            elif isinstance(before, datetime):
                before = int(before.timestamp())

            endpoint = f"/networks/{network}/pools/{clean_address}/ohlcv/{timeframe}"
            url = f"{self.BASE_URL}{endpoint}?limit=1000&before_timestamp={before}"

            #print(f"Fetching OHLCV: {url}")

            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        print(f"Rate limit hit, waiting...")
                        await asyncio.sleep(2)
                        return await self.get_ohlcv(pool_address, network, timeframe, before)
                    else:
                        print(f"Error {response.status} fetching OHLCV for {clean_address}")
                        return None
            except Exception as e:
                print(f"Exception fetching OHLCV: {e}")
                return None

def process_pool_list(pools_data, fetch_timestamp):
    """Convert pool list data to Polars DataFrame with timestamp"""
    records = []
    for pool in pools_data:
        try:
            attrs = pool['attributes']
            record = {
                'timestamp': fetch_timestamp,
                'pool_address': pool['id'],
                'name': attrs.get('name', ''),
                'base_token_name': attrs.get('base_token', {}).get('name', ''),
                'quote_token_name': attrs.get('quote_token', {}).get('name', ''),
                'base_token_symbol': attrs.get('base_token', {}).get('symbol', ''),
                'quote_token_symbol': attrs.get('quote_token', {}).get('symbol', ''),
                'base_token_price_usd': float(attrs.get('base_token_price_usd', 0)),
                'quote_token_price_usd': float(attrs.get('quote_token_price_usd', 0)),
                'volume_usd_24h': float(attrs.get('volume_usd_24h', 0)),
                'reserve_in_usd': float(attrs.get('reserve_in_usd', 0))
            }
            records.append(record)
        except Exception as e:
            print(f"Error processing pool data: {e}")
            continue

    return pl.DataFrame(records)

def process_ohlcv_data(ohlcv_data):
    """Convert OHLCV data to Polars DataFrame"""
    if not ohlcv_data.get('data', {}).get('attributes', {}).get('ohlcv_list'):
        return pl.DataFrame()

    records = []
    for entry in ohlcv_data['data']['attributes']['ohlcv_list']:
        try:
            record = {
                'timestamp': int(entry[0]),
                'open': float(entry[1]),
                'high': float(entry[2]),
                'low': float(entry[3]),
                'close': float(entry[4]),
                'volume': float(entry[5])
            }
            records.append(record)
        except Exception as e:
            print(f"Error processing OHLCV entry: {e}")
            continue

    return pl.DataFrame(records)

async def fetch_and_store_pools(network, resolution="1h", pool_type="trending"):
    base_dir = f"data/geckoterminal/{network}"

    # Convert resolution to timeframe format expected by the API
    resolution_map = {
        "1m": "minute",
        "5m": "minute5",
        "15m": "minute15",
        "30m": "minute30",
        "1h": "hour",
        "4h": "hour4",
        "12h": "hour12",
        "1d": "day"
    }
    timeframe = resolution_map.get(resolution, "hour")

    # Use the timeframe in the directory structure
    ohlcv_dir = f"{base_dir}/ohlcv/{timeframe}"
    pools_dir = f"{base_dir}/pools"

    os.makedirs(ohlcv_dir, exist_ok=True)
    os.makedirs(pools_dir, exist_ok=True)

    fetch_timestamp = datetime.utcnow().isoformat()
    current_timestamp = int(datetime.now().timestamp())
    oldest_needed = current_timestamp - (180 * 24 * 3600)

    async with GeckoTerminalClient() as client:
        # Fetch all pages of pools
        all_pools = []
        for page in range(1, 11):
            try:
                print(f"Fetching page {page}...")
                pools_data = await client.get_pools(network, pool_type, page)
                if not pools_data or 'data' not in pools_data:
                    print(f"No more data after page {page}")
                    break

                all_pools.extend(pools_data['data'])
                print(f"Fetched page {page} with {len(pools_data['data'])} pools. Total pools: {len(all_pools)}")

                # Check if we've reached the last page
                if len(pools_data['data']) < 20:  # Less than full page
                    print("Reached last page")
                    break

                # Add delay between pages
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error fetching page {page} of {pool_type} pools: {e}")
                break

        print(f"Total pools fetched: {len(all_pools)}")

        # Save pool list with timestamp
        if all_pools:
            pools_df = process_pool_list(all_pools, fetch_timestamp)
            pools_list_file = f"{pools_dir}/{pool_type}_pools.parquet"

            if os.path.exists(pools_list_file):
                existing_df = pl.read_parquet(pools_list_file)
                combined_df = pl.concat([existing_df, pools_df])
                combined_df = combined_df.unique(subset=['timestamp'])
                combined_df.write_parquet(pools_list_file)
            else:
                pools_df.write_parquet(pools_list_file)

        # Fetch OHLCV data for each pool
        for pool in all_pools:
            pool_address = pool['id']
            file_path = f"{ohlcv_dir}/{pool_address}.parquet"

            try:
                if os.path.exists(file_path):
                    existing_df = pl.read_parquet(file_path)
                    if not existing_df.is_empty():
                        existing_df = existing_df.sort('timestamp')
                        existing_timestamps = set(existing_df['timestamp'].to_list())

                        # Calculate missing periods
                        all_hours = range(
                            max(oldest_needed, int(existing_df['timestamp'].min())),
                            current_timestamp,
                            3600
                        )
                        missing_periods = []
                        current_gap_start = None

                        for hour in all_hours:
                            if hour not in existing_timestamps:
                                if current_gap_start is None:
                                    current_gap_start = hour
                            else:
                                if current_gap_start is not None:
                                    missing_periods.append((current_gap_start, hour))
                                    current_gap_start = None

                        if current_gap_start is not None:
                            missing_periods.append((current_gap_start, current_timestamp))

                        if not missing_periods:
                            print(f"Data is up to date for {pool_address}")
                            continue
                else:
                    missing_periods = [(oldest_needed, current_timestamp)]
                    existing_df = None

                # Fetch data for missing periods
                all_new_data = []
                for start, end in missing_periods:
                    before_timestamp = end
                    while before_timestamp > start:
                        ohlcv_data = await client.get_ohlcv(
                            pool_address,
                            network,
                            timeframe=timeframe,  # Use the mapped timeframe
                            before=before_timestamp
                        )

                        if not ohlcv_data or 'data' not in ohlcv_data:
                            break

                        df = process_ohlcv_data(ohlcv_data)
                        if df.is_empty():
                            break

                        all_new_data.append(df)

                        # Update timestamp for next iteration
                        before_timestamp = df['timestamp'].min()

                        await asyncio.sleep(0.5)  # Rate limiting

                if all_new_data:
                    new_df = pl.concat(all_new_data)
                    if existing_df is not None:
                        combined_df = pl.concat([existing_df, new_df])
                        combined_df = combined_df.unique(subset=['timestamp'])
                        combined_df = combined_df.sort('timestamp')
                        combined_df.write_parquet(file_path)
                    else:
                        new_df = new_df.sort('timestamp')
                        new_df.write_parquet(file_path)

            except Exception as e:
                print(f"Error processing pool {pool_address}: {e}")
                continue

            # Rate limiting
            await asyncio.sleep(0.5)

def handle_geckoterminal(args):
    asyncio.run(fetch_and_store_pools(
        network=args.network,
        resolution=args.resolution,
        pool_type=args.pool_type
    ))
