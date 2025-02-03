import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------
# Heston Model Simulation
# ---------------------------
def simulate_heston(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, n_paths):
    """
    Simulate asset paths under the Heston model.
    
    Parameters:
      S0      : initial spot price
      v0      : initial variance (volatility^2)
      r       : risk-free rate
      kappa   : mean reversion speed of the variance
      theta   : long-term variance level
      sigma_v : volatility of volatility (vol-of-vol)
      rho     : correlation between asset and volatility
      T       : time horizon in years
      steps   : number of time steps
      n_paths : number of simulation paths
      
    Returns:
      S : simulated asset price paths (n_paths x (steps+1))
      v : simulated variance paths (n_paths x (steps+1))
    """
    dt = T / steps
    S = np.zeros((n_paths, steps + 1))
    v = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    for t in range(steps):
        # Generate two independent standard normal samples
        z1 = np.random.normal(size=n_paths)
        z2 = np.random.normal(size=n_paths)
        # Correlate the second increment with the first
        dW1 = np.sqrt(dt) * z1
        dW2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        
        # Update variance process (ensuring non-negative variance)
        v[:, t+1] = v[:, t] + kappa * (theta - v[:, t]) * dt + sigma_v * np.sqrt(np.maximum(v[:, t], 0)) * dW2
        v[:, t+1] = np.maximum(v[:, t+1], 0)
        
        # Update asset price process
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + np.sqrt(np.maximum(v[:, t], 0)) * dW1)
    
    return S, v

# ---------------------------
# Black-Scholes Functions
# ---------------------------
def bs_call_price(S, K, T, r, sigma):
    """
    Compute the Black-Scholes price of a European call.
    
    Parameters:
      S     : underlying price
      K     : strike price
      T     : time to maturity
      r     : risk-free rate
      sigma : volatility
      
    Returns:
      Call price
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility_call(C_market, S, K, T, r):
    """
    Solve for the implied volatility given a market call price.
    
    Parameters:
      C_market : market (or simulated) call price
      S        : underlying price
      K        : strike price
      T        : time to maturity
      r        : risk-free rate
      
    Returns:
      Implied volatility (as a decimal)
    """
    f = lambda sigma: bs_call_price(S, K, T, r, sigma) - C_market
    try:
        imp_vol = brentq(f, 1e-6, 5.0)
    except ValueError:
        imp_vol = np.nan
    return imp_vol

# ---------------------------
# Build the Dash App Layout
# ---------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Options Scenario Simulation Dashboard"),
    
    # Left panel: parameter controls
    html.Div([
        html.Label("Initial Spot Price (S0)"),
        dcc.Slider(id="s0-slider", min=50, max=150, step=1, value=100,
                   marks={50: '50', 100: '100', 150: '150'}),
        html.Br(),
        
        html.Label("Initial Volatility (%)"),
        dcc.Slider(id="vol-slider", min=5, max=100, step=1, value=20,
                   marks={5: '5', 20: '20', 50: '50', 100: '100'}),
        html.Br(),
        
        html.Label("Risk-Free Rate (%)"),
        dcc.Slider(id="r-slider", min=0, max=10, step=0.1, value=2,
                   marks={0: '0', 2: '2', 5: '5', 10: '10'}),
        html.Br(),
        
        html.Label("Strike Price (K)"),
        dcc.Slider(id="strike-slider", min=50, max=150, step=1, value=100,
                   marks={50: '50', 100: '100', 150: '150'}),
        html.Br(),
        
        html.Label("Time to Maturity (Years)"),
        dcc.Slider(id="T-slider", min=0.1, max=3, step=0.1, value=1,
                   marks={0.1: '0.1', 1: '1', 2: '2', 3: '3'}),
        html.Br(),
        
        html.Label("Heston kappa (mean reversion speed)"),
        dcc.Slider(id="kappa-slider", min=0.1, max=10, step=0.1, value=1.5,
                   marks={0.1: '0.1', 1: '1', 5: '5', 10: '10'}),
        html.Br(),
        
        html.Label("Heston theta (long-term variance)"),
        dcc.Slider(id="theta-slider", min=0.01, max=0.5, step=0.01, value=0.04,
                   marks={0.01: '0.01', 0.1: '0.1', 0.5: '0.5'}),
        html.Br(),
        
        html.Label("Heston sigma (vol-of-vol)"),
        dcc.Slider(id="sigma-slider", min=0.1, max=2, step=0.1, value=0.5,
                   marks={0.1: '0.1', 1: '1', 2: '2'}),
        html.Br(),
        
        html.Label("Spot-Vol Correlation (rho)"),
        dcc.Slider(id="rho-slider", min=-1, max=1, step=0.05, value=-0.7,
                   marks={-1: '-1', -0.5: '-0.5', 0: '0', 0.5: '0.5', 1: '1'}),
        html.Br(),
        
        html.Label("Number of Simulation Paths"),
        dcc.Slider(id="npaths-slider", min=100, max=5000, step=100, value=1000,
                   marks={100: '100', 1000: '1000', 5000: '5000'}),
        html.Br(),
        
        html.Label("Number of Time Steps"),
        dcc.Slider(id="steps-slider", min=50, max=500, step=10, value=250,
                   marks={50: '50', 250: '250', 500: '500'}),
        html.Br(),
        
        html.Button("Run Simulation", id="run-button", n_clicks=0)
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # Right panel: simulation outputs
    html.Div([
        dcc.Graph(id="paths-graph"),
        dcc.Graph(id="histogram-graph"),
        html.Div(id="results-div", style={'fontSize': 24, 'padding': '20px'})
    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'})
])

# ---------------------------
# Callback to Update Simulation
# ---------------------------
@app.callback(
    [Output("paths-graph", "figure"),
     Output("histogram-graph", "figure"),
     Output("results-div", "children")],
    [Input("run-button", "n_clicks")],
    [State("s0-slider", "value"),
     State("vol-slider", "value"),
     State("r-slider", "value"),
     State("strike-slider", "value"),
     State("T-slider", "value"),
     State("kappa-slider", "value"),
     State("theta-slider", "value"),
     State("sigma-slider", "value"),
     State("rho-slider", "value"),
     State("npaths-slider", "value"),
     State("steps-slider", "value")]
)
def update_simulation(n_clicks, s0, vol_percent, r_percent, K, T,
                      kappa, theta, sigma, rho, n_paths, steps):
    if n_clicks is None:
        # Do not update if button has never been clicked
        return dash.no_update

    # Convert percentages to decimals:
    r = r_percent / 100.0
    init_vol = vol_percent / 100.0
    v0 = init_vol**2  # variance is the square of volatility

    # For reproducibility you may set the seed (or remove for fully random runs)
    np.random.seed(42)
    S, v = simulate_heston(S0=s0, v0=v0, r=r, kappa=kappa, theta=theta,
                           sigma_v=sigma, rho=rho, T=T, steps=steps, n_paths=n_paths)
    time_grid = np.linspace(0, T, steps + 1)
    
    # Plot a subset (e.g. 20) of simulated paths
    num_paths_to_plot = min(20, n_paths)
    paths_fig = go.Figure()
    for i in range(num_paths_to_plot):
        paths_fig.add_trace(go.Scatter(
            x=time_grid,
            y=S[i, :],
            mode='lines',
            name=f'Path {i+1}',
            line=dict(width=1)
        ))
    paths_fig.update_layout(title="Simulated Asset Price Paths",
                            xaxis_title="Time (Years)",
                            yaxis_title="Asset Price")
    
    # Create a histogram of terminal asset prices (S_T)
    terminal_prices = S[:, -1]
    hist_fig = go.Figure(data=[go.Histogram(x=terminal_prices, nbinsx=50)])
    hist_fig.update_layout(title="Histogram of Terminal Asset Prices",
                           xaxis_title="Asset Price at Maturity",
                           yaxis_title="Frequency")
    
    # Compute Monte Carlo European call price: discount the average payoff max(S_T - K, 0)
    payoffs = np.maximum(terminal_prices - K, 0)
    option_price_mc = np.exp(-r * T) * np.mean(payoffs)
    
    # Compute implied volatility from the simulated option price via the Black-Scholes formula
    imp_vol = implied_volatility_call(option_price_mc, s0, K, T, r)
    
    results_text = [
        html.P(f"Monte Carlo Call Option Price: {option_price_mc:.4f}"),
        html.P(f"Implied Volatility (from simulated price): {imp_vol*100:.2f}%")
    ]
    
    return paths_fig, hist_fig, results_text

# ---------------------------
# Run the App
# ---------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
