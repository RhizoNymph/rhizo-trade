from dash import Dash, html, dcc

import statsmodels.api as sm

def create_assumption_tests_div(heteroskedasticity_results, vif_results, selected_features, excluded_features):
    return html.Div([
        # Heteroskedasticity section
        html.H3('Heteroskedasticity Test (Breusch-Pagan)'),
        html.P([
            html.Strong('Null Hypothesis: '),
            'Homoskedasticity (constant variance)'
        ]),
        html.Table([
            html.Tr([html.Td('LM Statistic:'), html.Td(heteroskedasticity_results['lm_stat'])]),
            html.Tr([html.Td('LM p-value:'), html.Td(heteroskedasticity_results['p_value'])]),
            html.Tr([
                html.Td('Interpretation:'),
                html.Td('Heteroskedastic' if float(heteroskedasticity_results['p_value']) < 0.05
                       else 'Homoskedastic')
            ])
        ], style={'margin': '10px', 'border': '1px solid black'}),

        # Multicollinearity and Feature Selection section in same row
        html.Div([
            # VIF results
            html.Div([
                html.H3('Multicollinearity Test (VIF)'),
                html.P([
                    html.Strong('Rule of Thumb: '),
                    'VIF > 5 indicates high multicollinearity'
                ]),
                html.Table(
                    [html.Tr([html.Th('Feature'), html.Th('VIF')])] +
                    [html.Tr([html.Td(row['Feature']), html.Td(f"{row['VIF']:.4f}")])
                     for _, row in vif_results.iterrows()],
                    style={'margin': '10px', 'border': '1px solid black'}
                )
            ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Feature selection results
            html.Div([
                html.H3('Feature Selection Results'),
                html.Div([
                    html.P('Selected Features:'),
                    html.Ul([html.Li(feature) for feature in selected_features]),
                    html.P('Excluded Features (High VIF):'),
                    html.Ul([html.Li(feature) for feature in excluded_features]),
                ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'})
            ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})
        ], style={'width': '100%', 'display': 'flex'})
    ])

def create_model_comparison_div(wls_model, robust_model, selected_features, excluded_features):
    # Create base OLS model
    X_const = sm.add_constant(wls_model.model.exog)
    ols_model = sm.OLS(wls_model.model.endog, X_const).fit()

    return html.Div([
        # Model Performance section
        html.Div([
            html.H3('Model Performance Comparison'),
            html.Table([
                html.Tr([
                    html.Th('Metric'),
                    html.Th('OLS'),
                    html.Th('WLS'),
                    html.Th('Robust Regression')
                ]),
                html.Tr([
                    html.Td('R-squared'),
                    html.Td(f"{ols_model.rsquared:.4f}"),
                    html.Td(f"{wls_model.rsquared:.4f}"),
                    html.Td(f"{robust_model.rsquared:.4f}")
                ]),
                html.Tr([
                    html.Td('Adj. R-squared'),
                    html.Td(f"{ols_model.rsquared_adj:.4f}"),
                    html.Td(f"{wls_model.rsquared_adj:.4f}"),
                    html.Td(f"{robust_model.rsquared_adj:.4f}")
                ]),
                html.Tr([
                    html.Td('F-statistic p-value'),
                    html.Td(f"{ols_model.f_pvalue:.4f}"),
                    html.Td(f"{wls_model.f_pvalue:.4f}"),
                    html.Td(f"{robust_model.f_pvalue:.4f}")
                ])
            ], style={'margin': '10px', 'border': '1px solid black'})
        ], style={'width': '100%', 'margin-bottom': '20px'}),

        # Model Coefficients section
        html.Div([
            html.H3('Model Coefficients Comparison'),
            html.Table([
                html.Tr([
                    html.Th('Feature'),
                    html.Th('OLS Coef'),
                    html.Th('OLS p-value'),
                    html.Th('WLS Coef'),
                    html.Th('WLS p-value'),
                    html.Th('Robust Coef'),
                    html.Th('Robust p-value')
                ]),
                *[html.Tr([
                    html.Td(name),
                    html.Td(f"{ols_coef:.4f}"),
                    html.Td(f"{ols_pval:.4f}"),
                    html.Td(f"{wls_coef:.4f}"),
                    html.Td(f"{wls_pval:.4f}"),
                    html.Td(f"{rob_coef:.4f}"),
                    html.Td(f"{rob_pval:.4f}")
                ]) for name, ols_coef, ols_pval, wls_coef, wls_pval, rob_coef, rob_pval in zip(
                    wls_model.model.exog_names,
                    ols_model.params,
                    ols_model.pvalues,
                    wls_model.params,
                    wls_model.pvalues,
                    robust_model.params,
                    robust_model.pvalues
                )]
            ], style={'margin': '10px', 'border': '1px solid black'})
        ], style={'width': '100%'})
    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})

def create_stationarity_results_div(adf_results, kpss_results):
    return html.Div([
        html.H3('Augmented Dickey-Fuller Test'),
        html.P([
            html.Strong('Null Hypothesis: '),
            'Series has a unit root (non-stationary)'
        ]),
        html.Table([
            html.Tr([html.Td('Test Statistic:'), html.Td(adf_results['Test Statistic'])]),
            html.Tr([html.Td('p-value:'), html.Td(adf_results['p-value'])]),
            html.Tr([
                html.Td('Interpretation:'),
                html.Td('Stationary' if float(adf_results['p-value']) < 0.05 else 'Non-stationary')
            ])
        ], style={'margin': '10px', 'border': '1px solid black'}),

        html.H3('KPSS Test'),
        html.P([
            html.Strong('Null Hypothesis: '),
            'Series is stationary'
        ]),
        html.Table([
            html.Tr([html.Td('Test Statistic:'), html.Td(kpss_results['Test Statistic'])]),
            html.Tr([html.Td('p-value:'), html.Td(kpss_results['p-value'])]),
            html.Tr([
                html.Td('Interpretation:'),
                html.Td('Non-stationary' if kpss_results['p-value'] != '> 0.1' and
                       float(kpss_results['p-value'].replace('>', '')) < 0.05 else 'Stationary')
            ])
        ], style={'margin': '10px', 'border': '1px solid black'})
    ])
