from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from scipy import stats
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import logging
import glob

from util.timeseries import *
from util.stats import *
from visualize.regression_divs import *

def run_regression_dashboard():
    # Setup logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)

    # Read and combine all parquet files
    files = glob.glob('data/velo/spot/binance/1d/*.parquet')
    print(f"Found {len(files)} files")

    df = pd.concat([pd.read_parquet(f) for f in files])
    print(f"Total rows in combined dataframe: {len(df)}")

    # Sort by timestamp
    df = df.sort_values(['coin', 'timestamp'])

    # Apply both returns and VWMA calculations
    df = df.groupby('coin', group_keys=False).apply(calculate_returns)
    # Remove infinite values and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Create the Dash app
    app = Dash(__name__)

    # Define the layout
    app.layout = html.Div([
        html.H1('Return Analysis Dashboard'),

        html.Div([
                # First dropdown
                html.Div([
                    html.Label('Select View:'),
                    dcc.Dropdown(
                        id='view-selector',
                        options=[
                            {'label': 'Overview', 'value': 'overview'},
                            {'label': 'Individual Plots', 'value': 'individual'}
                        ],
                        value='overview'
                    )
                ], style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}),

                # Second dropdown
                html.Div([
                    html.Label('Select Future Return Period:'),
                    dcc.Dropdown(
                        id='future-return-selector',
                        options=[
                            {'label': '1 Day', 'value': 'future_1d_return'},
                            {'label': '3 Days', 'value': 'future_3d_return'},
                            {'label': '5 Days', 'value': 'future_5d_return'},
                            {'label': '7 Days', 'value': 'future_7d_return'}
                        ],
                        value='future_3d_return'
                    )
                ], style={'width': '200px', 'display': 'inline-block'})
            ], style={'margin': '10px', 'display': 'flex', 'alignItems': 'flex-end'}),

        html.Div(id='overview-container', children=[
            html.Div([
                # Left column - Stationarity Tests
                html.Div([
                    html.H2('Stationarity Test Results'),
                    html.Div(id='stationarity-results'),
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

                # Right column - Time Series Plot
                html.Div([
                    html.H2('Return Distribution Over Time'),
                    dcc.Graph(id='time-series-plot')
                ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
            ], style={'display': 'flex', 'margin-bottom': '20px'}),

            # Model Analysis Section
            html.Div([
                html.Div([
                    # Left side - Assumption Tests
                    html.Div([
                        html.H3('Model Assumption Tests'),
                        html.Div(id='assumption-tests-results')
                    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

                    # Right side - Model Comparison
                    html.Div([
                        html.H3('Model Results'),
                        html.Div(id='model-comparison-results')
                    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
                ], style={'display': 'flex', 'width': '100%'})
            ], style={'margin-top': '20px'})
        ], style={'padding': '20px'}),

        html.Div(id='individual-plots-container', children=[
            html.Div([
                dcc.Graph(id='scatter-3d', style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='scatter-5d', style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='scatter-7d', style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='scatter-14d', style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='scatter-30d', style={'width': '33%', 'display': 'inline-block'}),
                dcc.Graph(id='scatter-60d', style={'width': '33%', 'display': 'inline-block'})
            ])
        ], style={'display': 'none'})
    ])

    @app.callback(
        [Output('overview-container', 'style'),
        Output('individual-plots-container', 'style')],
        [Input('view-selector', 'value')]
    )
    def toggle_view(selected_view):
        overview_style = {'display': 'block'} if selected_view == 'overview' else {'display': 'none'}
        individual_style = {'display': 'block'} if selected_view == 'individual' else {'display': 'none'}
        #vwma_style = {'display': 'block'} if selected_view == 'vwma' else {'display': 'none'}
        return overview_style, individual_style

    @app.callback(
        [Output('scatter-3d', 'figure'),
        Output('scatter-5d', 'figure'),
        Output('scatter-7d', 'figure'),
        Output('scatter-14d', 'figure'),
        Output('scatter-30d', 'figure'),
        Output('scatter-60d', 'figure'),
        Output('time-series-plot', 'figure'),
        Output('stationarity-results', 'children'),
        Output('assumption-tests-results', 'children'),
        Output('model-comparison-results', 'children')],
        [Input('future-return-selector', 'value')]
    )
    def update_graphs(selected_future_return):
        future_return_label = selected_future_return.replace('_', ' ').title()

        # Perform stationarity tests
        adf_results, kpss_results = perform_stationarity_tests(df[selected_future_return])
        stationarity_results = create_stationarity_results_div(adf_results, kpss_results)

        # Prepare data for regression
        past_returns = ['return_3d', 'return_5d', 'return_7d', 'return_14d', 'return_30d', 'return_60d']
        X = df[past_returns]
        y = df[selected_future_return]

        # Feature selection using VIF
        selected_features, excluded_features = select_features_vif(X, threshold=5)
        X_selected = X[selected_features]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )

        # Fit models
        wls_model = perform_wls_regression(X_scaled, y)
        robust_model = perform_robust_regression(X_scaled, y)

        # Perform assumption tests
        heteroskedasticity_results = perform_heteroskedasticity_test(X_scaled, y)
        vif_results = calculate_vif(X_selected)
        assumption_tests = create_assumption_tests_div(
            heteroskedasticity_results,
            vif_results,
            selected_features,
            excluded_features
        )

        # Create model comparison
        model_comparison = create_model_comparison_div(
            wls_model,
            robust_model,
            selected_features,
            excluded_features
        )

        # Create scatter plots
        figures = []
        for past_return, days in [('return_3d', '3-Day'), ('return_5d', '5-Day'),
                                ('return_7d', '7-Day'), ('return_14d', '14-Day'),
                                ('return_30d', '30-Day'), ('return_60d', '60-Day')]:

            x = df[past_return].values
            y = df[selected_future_return].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            line_x = np.array([min(x), max(x)])
            line_y = slope * line_x + intercept
            r_squared = r_value**2

            figures.append({
                'data': [
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        opacity=0.7,
                        marker={'size': 5},
                        name=f'{days} Return'
                    ),
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name=f'Trendline (R² = {r_squared:.3f})',
                        line=dict(color='red', width=2)
                    )
                ],
                'layout': go.Layout(
                    title=f'{days} Return vs {future_return_label}<br>R² = {r_squared:.3f}, Slope = {slope:.3f}',
                    xaxis={'title': f'{days} Return'},
                    yaxis={'title': future_return_label},
                    hovermode='closest',
                    showlegend=True
                )
            })

        # Create time series plot
        time_series_fig = {
            'data': [
                go.Scatter(
                    x=df['timestamp'],
                    y=df[selected_future_return],
                    mode='markers',
                    opacity=1,
                    marker=dict(
                        size=4,
                        color=df[selected_future_return],
                        colorscale='RdYlBu',
                        showscale=True,
                        colorbar=dict(title='Return')
                    ),
                    name='Returns'
                )
            ],
            'layout': go.Layout(
                title=f'{future_return_label} Distribution Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Return'},
                hovermode='closest',
                showlegend=True
            )
        }

        return figures + [time_series_fig, stationarity_results,
                        assumption_tests, model_comparison]

    app.run_server(debug=True, host="0.0.0.0")
