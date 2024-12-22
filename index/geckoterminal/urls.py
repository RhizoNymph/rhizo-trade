import datetime
import sys

free_url = "https://api.geckoterminal.com/api/v2"
base_url = "https://pro-api.coingecko.com/api/v3/onchain"

def get_base_url(filling_gaps=False):
    if filling_gaps:
        return free_url
    return free_url if (len(sys.argv) > 1 and sys.argv[1] == "free") else base_url

def trending_pools_url(network, page=1):
    return get_base_url() + f'/networks/{network}/trending_pools?page={page}'

def new_pools_url(network, page=1):
    return get_base_url() + f'/networks/{network}/trending_pools?page={page}'

def top_pools_url(network, page):
    return get_base_url() + f'/networks/{network}/pools?page={page}'

def pool_url(network, pool):
    return get_base_url() + f'/networks/{network}/pools/{pool}'

def multi_pool_url(network, pools):
    return get_base_url() + f'/networks/{network}/pools/multi/{pools}'

def ohlcv_url(network, pool, timeframe = "hour", before = int(datetime.datetime.now().timestamp()), filling_gaps=False):
    if before is None:
        before = int(datetime.datetime.now().timestamp())
    # Ensure before is an integer Unix timestamp
    elif isinstance(before, datetime.datetime):
        before = int(before.timestamp())

    return get_base_url(filling_gaps) + f'/networks/{network}/pools/{pool}/ohlcv/{timeframe}?limit=1000&before_timestamp={before}'

def trades_url(network, pool):
    return get_base_url() + f'/networks/{network}/pools/{pool}/trades'

def top_pools_for_token(network, token):
    return get_base_url() + f'/networks/{network}/tokens/{token}/pools'

def token_info_url(network, token):
    return get_base_url() + f'/networks/{network}/tokens/{token}/info'
