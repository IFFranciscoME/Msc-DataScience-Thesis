
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizaciones.py : python script with the main functionality                               -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import chart_studio
import chart_studio.plotly as py
pio.renderers.default = "browser"

# -- ------------------------------------------------------------------------------- CREDENCIALES PLOTLY -- #
# -- --------------------------------------------------------------------------------------------------- -- #

chart_studio.tools.set_credentials_file(username='IFFranciscoME', api_key='Wv3JHvYz5h5jHGpuxvJQ')
chart_studio.tools.set_config_file(world_readable=True, sharing='public')


# -- -------------------------------------------------------- PLOT: OHLC Price Chart with Vertical Lines -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_ohlc(p_ohlc, p_theme, p_vlines):
    """
    Timeseries Candlestick with OHLC prices and figures for trades indicator

    Requirements
    ------------
    numpy
    pandas
    plotly

    Parameters
    ----------
    p_ohlc: pd.DataFrame
        that contains the following float or int columns: 'timestamp', 'open', 'high', 'low', 'close'

    p_theme: dict
        with the theme for the visualizations

    p_vlines: list
        with the dates where to visualize the vertical lines, format = pd.to_datetime('2020-01-01 22:15:00')

    Returns
    -------
    fig_g_ohlc: plotly
        objet/dictionary to .show() and plot in the browser

    References
    ----------
    https://plotly.com/python/candlestick-charts/

    """

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme['p_labels'] is not None:
        p_labels = p_theme['p_labels']
    else:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                              (max(p_ohlc['high']) - min(p_ohlc['low'])) / 10)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # Instantiate a figure object
    fig_g_ohlc = go.Figure()

    # Add layer for OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(name='ohlc', x=p_ohlc['timestamp'], open=p_ohlc['open'],
                                        high=p_ohlc['high'], low=p_ohlc['low'], close=p_ohlc['close'],
                                        opacity=0.7))

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
                             xaxis=dict(title_text=p_labels['x_title']),
                             yaxis=dict(title_text=p_labels['y_title']))

    # Color and font type for text in axes
    fig_g_ohlc.update_layout(xaxis=dict(titlefont=dict(color=p_theme['p_colors']['color_1']),
                                        tickfont=dict(color=p_theme['p_colors']['color_1'],
                                                      size=p_theme['p_fonts']['font_axis']), showgrid=False),
                             yaxis=dict(zeroline=False, automargin=True, tickformat='.4f',
                                        titlefont=dict(color=p_theme['p_colors']['color_1']),
                                        tickfont=dict(color=p_theme['p_colors']['color_1'],
                                                      size=p_theme['p_fonts']['font_axis']),
                                        showgrid=True, gridcolor='lightgrey', gridwidth=.05))

    # If parameter vlines is used
    if p_vlines is not None:
        # Dynamically add vertical lines according to the provided list of x dates.
        shapes_list = list()
        for i in p_vlines:
            shapes_list.append({'type': 'line', 'fillcolor': p_theme['p_colors']['color_1'],
                                'line': {'color': p_theme['p_colors']['color_1'],
                                         'dash': 'dashdot', 'width': 3},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Update layout for the background
    fig_g_ohlc.update_layout(yaxis=dict(tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis']),
                                        tickvals=y0_ticks_vals),
                             xaxis=dict(tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the y axis
    fig_g_ohlc.update_xaxes(rangebreaks=[dict(pattern="day of week", bounds=['sat', 'sun'])])

    # Update layout for the background
    fig_g_ohlc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text='<b> ' + p_labels['title'] + ' </b>'),
                             yaxis=dict(title=p_labels['y_title'],
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)),
                             xaxis=dict(title=p_labels['x_title'], rangeslider=dict(visible=False),
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)))

    # Final plot dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_theme['p_dims']['width']
    fig_g_ohlc.layout.height = p_theme['p_dims']['height']

    return fig_g_ohlc


# -- -------------------------------------------- PLOT: OHLC Candlesticks + Colored Classificator Result -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_ohlc_class(p_ohlc, p_theme, p_data_class, p_vlines):

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme['p_labels'] is not None:
        p_labels = p_theme['p_labels']
    else:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                              (max(p_ohlc['high']) - min(p_ohlc['low'])) / 5)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # reset the index of the input data
    p_ohlc.reset_index(inplace=True, drop=True)

    # auxiliar lists
    train_error = []
    test_error = []
    test_success = []
    train_success = []

    # error and success in train
    for row in p_data_class['train_y'].index.to_list():
        if p_data_class['train_y'][row] != p_data_class['train_y_pred'][row]:
            train_error.append(row)
        else:
            train_success.append(row)

    # error and success in test
    for row in p_data_class['test_y'].index.to_list():
        if p_data_class['test_y'][row] != p_data_class['test_y_pred'][row]:
            test_error.append(row)
        else:
            test_success.append(row)

    # train and test errors in a list
    train_test_error = train_error + test_error

    # train and test success in a list
    train_test_success = train_success + test_success

    # Instantiate a figure object
    fig_g_ohlc = go.Figure()

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
                             xaxis=dict(title_text=p_labels['x_title']),
                             yaxis=dict(title_text=p_labels['y_title']))

    # Add layer for the error based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_test_error],
        open=[p_ohlc['open'].iloc[i] for i in train_test_error],
        high=[p_ohlc['high'].iloc[i] for i in train_test_error],
        low=[p_ohlc['low'].iloc[i] for i in train_test_error],
        close=[p_ohlc['close'].iloc[i] for i in train_test_error],
        increasing={'line': {'color': 'red'}},
        decreasing={'line': {'color': 'red'}},
        name='Prediction Error'))

    # Add layer for the success based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_test_success],
        open=[p_ohlc['open'].iloc[i] for i in train_test_success],
        high=[p_ohlc['high'].iloc[i] for i in train_test_success],
        low=[p_ohlc['low'].iloc[i] for i in train_test_success],
        close=[p_ohlc['close'].iloc[i] for i in train_test_success],
        increasing={'line': {'color': 'skyblue'}},
        decreasing={'line': {'color': 'skyblue'}},
        name='Prediction Success'))

    # Update layout for the background
    fig_g_ohlc.update_layout(yaxis=dict(tickfont=dict(color='grey',
     size=p_theme['p_fonts']['font_axis']), tickvals=y0_ticks_vals),
     xaxis=dict(tickfont=dict(color='grey',
     size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the y axis
    fig_g_ohlc.update_xaxes(rangebreaks=[dict(pattern="day of week", bounds=['sat', 'sun'])])

    # If parameter vlines is used
    if p_vlines is not None:
        # Dynamically add vertical lines according to the provided list of x dates.
        shapes_list = list()
        for i in p_vlines:
            shapes_list.append({'type': 'line', 'fillcolor': p_theme['p_colors']['color_1'],
                                'line': {'color': p_theme['p_colors']['color_1'],
                                         'dash': 'dashdot', 'width': 3},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Formato para titulo
    fig_g_ohlc.update_layout(legend=go.layout.Legend(x=.35, y=-.3, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the background
    fig_g_ohlc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text='<b> ' + p_labels['title'] + ' </b>'),
                             yaxis=dict(title=p_labels['y_title'],
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)),
                             xaxis=dict(title=p_labels['x_title'], rangeslider=dict(visible=False),
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)))

    # Final plot dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_theme['p_dims']['width']
    fig_g_ohlc.layout.height = p_theme['p_dims']['height']

    return fig_g_ohlc

# -- ----------------------------------------------------------------------------------- PLOT: ROC + ACU -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_roc_auc(p_casos, p_theme):

    # p_casos = casos
    fig_rocs = go.Figure()
    fig_rocs.update_layout(
        title=dict(x=0.5, text='Grafica 4:' + '<b> ' + p_theme['p_labels']['title'] + ' </b>'),
        xaxis=dict(title_text=p_theme['p_labels']['x_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])),
        yaxis=dict(title_text=p_theme['p_labels']['y_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])))

    fig_rocs.add_shape(type='line', line=dict(width=3, dash='dash', color='grey'), x0=0, x1=1, y0=0, y1=1)
    model_name = ''
    for model in ['model_1', 'model_2', 'model_3']:
        for auc_type in ['auc_min', 'auc_max']:
            p_fpr = p_casos[model][auc_type]['data']['metrics']['train']['fpr']
            p_tpr = p_casos[model][auc_type]['data']['metrics']['train']['tpr']

            if model == 'model_1':
                model_name = 'logistic-elasticnet'
            elif model == 'model_2':
                model_name = 'ls-svm'
            elif model == 'model_3':
                model_name = 'ann-mlp'

            if auc_type == 'auc_min':
                fig_rocs.add_trace(go.Scatter(x=p_fpr, y=p_tpr, name=model_name,
                                              mode='lines+markers', line=dict(width=2, color='red')))
            elif auc_type == 'auc_max':
                fig_rocs.add_trace(go.Scatter(x=p_fpr, y=p_tpr, name=model_name,
                                              mode='lines+markers', line=dict(width=2, color='blue')))

    # Formato para titulo
    fig_rocs.update_layout(legend=go.layout.Legend(x=.090, y=-0.11, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=12)))

    # Formato de tamanos
    fig_rocs.layout.autosize = True
    fig_rocs.layout.width = p_theme['p_dims']['width']
    fig_rocs.layout.height = p_theme['p_dims']['height']

    # Mostrar la grafica
    fig_rocs.show()

    return fig_rocs


# -- ----------------------------------------------------------------------------------- PLOT: ROC + ACU -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_timeseries_auc(p_data_auc, p_theme):
    """
    Plot para series de tiempo de las AUC de los modelos

    Parameters
    ----------
    p_data_auc:dict
        Diccionario con datos para plot de series de tiempo AUC
        p_data_auc = minmax_auc_test

    p_theme: dict
        Diccionario con informacion de tema para plot
        p_theme = theme_plot_4

    Returns
    -------
    fig_ts_auc: plotly
        Objeto tipo plotly para utilizar con .show()

    """

    fig_ts_auc = go.Figure()
    fig_ts_auc.update_layout(
        title=dict(x=0.5, text='Grafica 5:' + '<b> ' + p_theme['p_labels']['title'] + ' </b>'),
        xaxis=dict(title_text=p_theme['p_labels']['x_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])),
        yaxis=dict(title_text=p_theme['p_labels']['y_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_1']['x_period'],
                                    y=p_data_auc['model_1']['y_mins'],

                                    line=dict(color='#004A94', width=3),
                                    marker=dict(color='#004A94', size=9),
                                    name='logistic-elasticnet (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_1']['x_period'], fillcolor='blue',
                                    y=p_data_auc['model_1']['y_maxs'],

                                    line=dict(color='#004A94', width=3),
                                    marker=dict(color='#004A94', size=9),
                                    name='logistic-elasticnet (max)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_2']['x_period'],
                                    y=p_data_auc['model_2']['y_mins'],

                                    line=dict(color='#FB5D41', width=3),
                                    marker=dict(color='#FB5D41', size=9),
                                    name='ls-svm (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_2']['x_period'],
                                    y=p_data_auc['model_2']['y_maxs'],

                                    line=dict(color='#FB5D41', width=3),
                                    marker=dict(color='#FB5D41', size=9),
                                    name='ls-svm (max)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_3']['x_period'],
                                    y=p_data_auc['model_3']['y_mins'],

                                    line=dict(color='#339e62', width=3),
                                    marker=dict(color='#339e62', size=9),
                                    name='ann-mlp (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_3']['x_period'],
                                    y=p_data_auc['model_3']['y_maxs'],

                                    line=dict(color='#339e62', width=3),
                                    marker=dict(color='#339e62', size=9),
                                    name='ann-mlp (min)',
                                    mode='markers+lines'))

    # # Update layout for the background
    # fig_ts_auc.update_layout(paper_bgcolor='white', title_font_size=p_theme['p_theme']['font_title'],
    #                          yaxis=dict(tickvals=np.arange(0, 1.1, 0.1), zeroline=False, automargin=True,
    #                                     titlefont=dict(size=p_theme['p_theme']['font_axis']+4)),
    #                          xaxis=dict(titlefont=dict(size=p_theme['p_theme']['font_axis']+4)))

    # Formato para titulo
    fig_ts_auc.update_layout(legend=go.layout.Legend(x=.1, y=-0.21, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=16)))

    # Formato de tamanos
    fig_ts_auc.layout.autosize = True
    fig_ts_auc.layout.width = p_theme['p_dims']['width']
    fig_ts_auc.layout.height = p_theme['p_dims']['height']

    # Para generar el plot y visualizarlo en explorador
    # fig_ts_auc.show()

    # Para generar el plot con URL en chart studio
    # py.plot(fig_ts_auc)

    return fig_ts_auc
