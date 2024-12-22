import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import polars as pl
import os
from datetime import datetime, timedelta
import numpy as np
import logging

AVAILABLE_RESOLUTIONS = ['hour', 'day']

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

def get_available_networks():
    """Get list of networks from data directory structure"""
    base_dir = "data/geckoterminal"
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist")
        return []
    networks = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f"Found networks: {networks}")
    return networks

def get_all_pools(network, resolution):
    """Get all OHLCV files for a network and resolution"""
    ohlcv_dir = f"data/geckoterminal/{network}/ohlcv/{resolution}"
    if not os.path.exists(ohlcv_dir):
        logger.warning(f"OHLCV directory {ohlcv_dir} does not exist")
        return []
    pools = [f.replace('.parquet', '') for f in os.listdir(ohlcv_dir)
             if f.endswith('.parquet')]
    logger.info(f"Found {len(pools)} pools for network {network} at {resolution} resolution")
    return pools

def load_pool_metadata(network):
    """Load pool metadata from any available pool list"""
    pools_dir = f"data/geckoterminal/{network}/pools"
    if not os.path.exists(pools_dir):
        logger.warning(f"Pools directory {pools_dir} does not exist")
        return None

    for file in os.listdir(pools_dir):
        if file.endswith('_pools.parquet'):
            logger.info(f"Loading pool metadata from {file}")
            df = pl.read_parquet(os.path.join(pools_dir, file))
            latest_timestamp = df['timestamp'].max()
            latest_df = df.filter(pl.col('timestamp') == latest_timestamp)
            logger.info(f"Loaded metadata for {latest_df.height} pools from {file}")
            return latest_df
    logger.warning("No pool metadata files found")
    return None

def load_ohlcv_data(network, pool_address, resolution):
    """Load OHLCV data for a specific pool"""
    ohlcv_file = f"data/geckoterminal/{network}/ohlcv/{resolution}/{pool_address}.parquet"
    if not os.path.exists(ohlcv_file):
        logger.warning(f"OHLCV file not found: {ohlcv_file}")
        return None
    df = pl.read_parquet(ohlcv_file)
    logger.debug(f"Loaded {df.height} OHLCV records for pool {pool_address}")
    return df

def calculate_cumulative_returns(ohlcv_df):
    """Calculate cumulative returns from close prices"""
    if ohlcv_df is None or ohlcv_df.is_empty():
        return None

    # Calculate returns and cumulative returns
    returns = (ohlcv_df['close'] / ohlcv_df['close'].shift(1) - 1).fill_null(0)
    cum_returns = np.cumprod(1 + returns.to_numpy()) - 1

    return pl.DataFrame({
        'timestamp': ohlcv_df['timestamp'],
        'cumulative_returns': cum_returns
    })

# Get initial networks
available_networks = get_available_networks()
default_network = available_networks[0] if available_networks else None

app.layout = html.Div([
    html.H1("GeckoTerminal Pool Performance"),

    # Controls container
    html.Div([
        # Network selector
        html.Div([
            html.Label("Select Network:"),
            dcc.Dropdown(
                id='network-dropdown',
                options=[{'label': network.capitalize(), 'value': network}
                        for network in available_networks],
                value=default_network
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),

        # Resolution selector
        html.Div([
            html.Label("Select Resolution:"),
            dcc.Dropdown(
                id='resolution-dropdown',
                options=[{'label': res.capitalize(), 'value': res}
                        for res in AVAILABLE_RESOLUTIONS],
                value='hour'
            )
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),

        # Time range selector
        html.Div([
            html.Label("Select Time Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                display_format='YYYY-MM-DD'
            )
        ], style={'width': '40%', 'display': 'inline-block'}),
    ]),

    # Graph
    dcc.Graph(id='cumulative-returns-graph'),

    # Pool selection and copy section
    html.Div([
        html.H4("Selected Pool:"),
        html.Div(id='selected-pool-info', style={
            'display': 'inline-block',
            'marginRight': '10px'
        }),
        html.Button('Copy Address', id='copy-button', n_clicks=0),
        html.Div(id='copy-status', style={
            'display': 'inline-block',
            'marginLeft': '10px'
        })
    ], style={'margin': '20px'}),

    # Store components
    dcc.Store(id='selected-pool-store'),
    dcc.Clipboard(id='clipboard', target_id='selected-pool-info'),
    dcc.Interval(id='clear-copy-status', interval=2000, disabled=True),

    # Stats display
    html.Div(id='stats-display', style={'margin': '20px'}),
])

@app.callback(
    [Output('selected-pool-store', 'data'),
     Output('selected-pool-info', 'children')],
    [Input('cumulative-returns-graph', 'clickData')],
    prevent_initial_call=True
)
def update_selected_pool(clickData):
    if clickData is None:
        return None, "No pool selected"

    try:
        # Get the first point's data
        point = clickData['points'][0]

        # Extract pool address and name from customdata
        pool_address = point['customdata'][0].split('_')[1]
        pool_name = point['customdata'][1]

        logger.info(f"Selected: {pool_name} ({pool_address})")

        return pool_address, html.Div([
            #html.Div(f"Selected Pool: {pool_name}"),
            html.Div(f"{pool_address}",)
                    #style={'wordBreak': 'break-all', 'fontSize': '0.9em', 'color': '#666'})
        ])
    except Exception as e:
        logger.error(f"Error in update_selected_pool: {str(e)}")
        logger.error(f"Received clickData: {clickData}")
        return None, "Error selecting pool"

@app.callback(
    [Output('clipboard', 'content'),
     Output('copy-status', 'children')],
    [Input('copy-button', 'n_clicks'),
     Input('clear-copy-status', 'n_intervals')],  # Add this input
    [dash.State('selected-pool-store', 'data')],
    prevent_initial_call=True
)
def handle_clipboard_actions(n_clicks, n_intervals, selected_pool):
    if not ctx.triggered:
        return None, ""

    # Check which input triggered the callback
    trigger_id = ctx.triggered[0]['prop_id']

    if trigger_id == 'clear-copy-status.n_intervals':
        return None, ""

    if n_clicks > 0 and selected_pool is not None:
        return selected_pool, html.Div([
            html.Span("Copied!", style={'color': 'green'}),
            dcc.Interval(id='clear-copy-status', interval=2000, n_intervals=0)
        ])

    return None, ""

# Update the callback to include resolution
@app.callback(
    [Output('cumulative-returns-graph', 'figure'),
     Output('stats-display', 'children')],
    [Input('network-dropdown', 'value'),
     Input('resolution-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_graph(network, resolution, start_date, end_date):
    if not network or not resolution or not start_date or not end_date:
        return go.Figure(), "Please select network, resolution, and date range"

    logger.info(f"Updating graph for network: {network} with resolution: {resolution}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Convert dates to timestamps with proper ISO format handling
    try:
        start_ts = int(datetime.fromisoformat(start_date).timestamp())
        end_ts = int(datetime.fromisoformat(end_date).timestamp())
    except ValueError:
        start_ts = int(datetime.strptime(start_date.split('T')[0], '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date.split('T')[0], '%Y-%m-%d').timestamp())

    fig = go.Figure()

    # Load pool metadata for better labeling
    pool_metadata = load_pool_metadata(network)

    # Get all pools for the selected resolution
    pools = get_all_pools(network, resolution)
    processed_pools = 0
    pools_with_data = 0
    pools_with_sufficient_data = 0

    for pool_address in pools:
        processed_pools += 1
        ohlcv_df = load_ohlcv_data(network, pool_address, resolution)
        if ohlcv_df is None:
            continue

        pools_with_data += 1

        # Filter by date range
        ohlcv_df = ohlcv_df.filter(
            (pl.col('timestamp') >= start_ts) &
            (pl.col('timestamp') <= end_ts)
        )

        if ohlcv_df.height < 2:  # Need at least 2 points for returns
            logger.debug(f"Insufficient data points for pool {pool_address}")
            continue

        pools_with_sufficient_data += 1

        cum_returns_df = calculate_cumulative_returns(ohlcv_df)
        if cum_returns_df is None:
            continue

        # Convert timestamp to datetime for better x-axis formatting
        dates = [datetime.fromtimestamp(ts) for ts in cum_returns_df['timestamp']]

        # Get pool name from metadata
        pool_name = pool_address
        if pool_metadata is not None:
            pool_info = pool_metadata.filter(pl.col('pool_address') == pool_address)
            if not pool_info.is_empty():
                pool_name = f"{pool_info['base_token_symbol'][0]}-{pool_info['quote_token_symbol'][0]}"

        fig.add_trace(go.Scatter(
            x=dates,
            y=cum_returns_df['cumulative_returns'],
            mode='lines',
            name=pool_name,
            customdata=[[pool_address, pool_name] for _ in dates],
            showlegend=True
        ))

    fig.update_layout(
        title=f'Cumulative Returns - All Pools ({resolution.capitalize()} Resolution)',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        hovermode='closest',
        clickmode='event'  # Add this line to enable clicking
    )

    # Create stats summary
    stats_html = html.Div([
        html.H3("Processing Statistics"),
        html.P(f"Total pools found: {len(pools)}"),
        html.P(f"Pools processed: {processed_pools}"),
        html.P(f"Pools with data: {pools_with_data}"),
        html.P(f"Pools with sufficient data: {pools_with_sufficient_data}"),
        html.P(f"Pools displayed on graph: {len(fig.data)}")
    ])

    logger.info(f"Graph updated with {len(fig.data)} pools")

    return fig, stats_html

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
