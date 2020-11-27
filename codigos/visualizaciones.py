
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

def g_ohlc(p_ohlc, p_theme, p_dims, p_labels, p_vlines):
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
    p_dims: dict
        with sizes for visualizations
    p_labels: dict
        with main title and both x and y axes
    p_vlines: list
        with the dates where to visualize the vertical lines, format = pd.to_datetime('2020-01-01 22:15:00')

    Returns
    -------
    fig_g_ohlc: plotly
        objet/dictionary to .show() and plot in the browser

    Debugging
    ---------
    p_ohlc = price_data
    p_theme = p_theme
    p_dims = p_dims

    """

    # default value for lables to use in main title, and both x and y axis
    if p_labels is None:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                              (max(p_ohlc['high']) - min(p_ohlc['low'])) / 10)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # instantiate a figure object
    fig_g_ohlc = go.Figure()

    # Add layer for OHLC candlestick chart
    fig_g_ohlc.add_trace(
        go.Candlestick(name='ohlc',
                       x=p_ohlc['timestamp'],
                       open=p_ohlc['open'],
                       high=p_ohlc['high'],
                       low=p_ohlc['low'],
                       close=p_ohlc['close'],
                       opacity=0.7))

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(
        margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
        xaxis=dict(title_text=p_labels['x_title'], rangeslider=dict(visible=False)),
        yaxis=dict(title_text=p_labels['y_title']))

    # Color and font type for text in axes
    fig_g_ohlc.update_layout(
        xaxis=dict(titlefont=dict(color=p_theme['color_1']),
                   tickfont=dict(color=p_theme['color_1'],
                                 size=p_theme['font_size_1'])),
        yaxis=dict(zeroline=False, automargin=True,
                   titlefont=dict(color=p_theme['color_1']),
                   tickfont=dict(color=p_theme['color_1'],
                                 size=p_theme['font_size_1']),
                   showgrid=True))

    # Size of final plot according to desired dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_dims['width']
    fig_g_ohlc.layout.height = p_dims['height']

    # if parameter vlines is used
    if p_vlines is not None:
        # Dynamically add vertical lines according to the provided list of x dates.
        shapes_list = list()
        for i in p_vlines:
            shapes_list.append({'type': 'line', 'fillcolor': p_theme['color_1'],
                                'line': {'color': p_theme['color_1'], 'dash': 'dashdot'},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Update layout for the background
    fig_g_ohlc.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                                    yaxis=dict(tickvals=y0_ticks_vals, zeroline=False, automargin=True,
                                               tickfont=dict(color='grey', size=p_theme['font_size_1'])),
                                    xaxis=dict(tickfont=dict(color='grey', size=p_theme['font_size_1'])))

    # Update layout for the y axis
    fig_g_ohlc.update_yaxes(showgrid=True, gridwidth=.25, gridcolor='lightgrey')
    fig_g_ohlc.update_xaxes(showgrid=False, rangebreaks=[dict(pattern="day of week",
                                                              bounds=['sat', 'sun'])])

    # Format to main title
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=20),
                               title=dict(x=0.5, text='<b> ' + p_labels['title'] + ' </b>'),
                               legend=go.layout.Legend(x=.3, y=-.15, orientation='h', font=dict(size=15)))

    return fig_g_ohlc


# -- --------------------------------------------------------------------- PLOT: Stacked Horizontal Bars -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_relative_bars(p_x, p_y0, p_y1, p_theme, p_dims):
    """
    Generates a plot with two bars (two series of values) and two horizontal lines (medians of each
    series)

    Requirements
    ------------
    numpy
    pandas
    plotly

    Parameters
    ----------
    p_x : list
        lista con fechas o valores en el eje de x

    p_y0: dict
        values for upper bar plot
        {data: y0 component to plot (left axis), color: for this data, type: line/dash/dash-dot,
        size: for this data, n_ticks: number of ticks for this axis}

    p_y1: dict
        values for lower bar plot
        {data: y0 component to plot (right axis), color: for this data, type: line/dash/dash-dot,
        size: for this data, n_ticks: number of ticks for this axis}

    p_theme: dict
        colors and font sizes
        {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
        'font_size_1': 12, 'font_size_2': 16}

    p_dims: dict
        final dimensions of the plot as a file
        {'width': width in pixels, 'heigth': height in pixels}

    Returns
    -------
    fig_relative_bars: plotly
        Object with plotly generating code for the plot

    """

    # instantiate a figure object
    fig_relative_bars = go.Figure()

    # Add lower bars
    fig_relative_bars.add_trace(go.Bar(name='Prediccion de Modelo', x=p_x, y=p_y1,
                                       marker_color='red',
                                       marker_line_color='red',
                                       marker_line_width=1, opacity=0.99))

    # Add upper bars
    fig_relative_bars.add_trace(go.Bar(name='Observacion', x=p_x, y=p_y0,
                                       marker_color=p_theme['color_1'],
                                       marker_line_color=p_theme['color_1'],
                                       marker_line_width=1, opacity=0.99))

    # Update layout for the background
    fig_relative_bars.update_layout(paper_bgcolor='white',
                                    yaxis=dict(tickvals=[-1, 0, 1], zeroline=True, automargin=True,
                                               tickfont=dict(color='grey', size=p_theme['font_size_1'])),
                                    xaxis=dict(tickfont=dict(color='grey', size=p_theme['font_size_1'])))

    # Update layout for the y axis
    fig_relative_bars.update_yaxes(showgrid=False, range=[-1.2, 1.2])

    # Legend format
    fig_relative_bars.update_layout(paper_bgcolor='white', plot_bgcolor='white', barmode='overlay',
                                    legend=go.layout.Legend(x=.41, y=-.10, orientation='h',
                                                            font=dict(size=14, color='grey')),
                                    margin=go.layout.Margin(l=50, r=50, b=50, t=50, pad=10))

    # Final plot dimensions
    fig_relative_bars.layout.autosize = True
    fig_relative_bars.layout.width = p_dims['width']
    fig_relative_bars.layout.height = p_dims['height']

    return fig_relative_bars


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
        title=dict(x=0.5, text='Grafica 3:' + '<b> ' + p_theme['p_labels']['title'] + ' </b>'),
        xaxis=dict(title_text=p_theme['p_labels']['x_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])),
        yaxis=dict(title_text=p_theme['p_labels']['y_title'],
                   tickfont=dict(color='grey', size=p_theme['p_theme']['font_axis'])))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_1']['x_period'],
                                    y=p_data_auc['model_1']['y_mins'],

                                    line=dict(color='blue', width=3),
                                    marker=dict(color='blue', size=9),
                                    name='logistic-elasticnet (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_1']['x_period'], fillcolor='blue',
                                    y=p_data_auc['model_1']['y_maxs'],

                                    line=dict(color='blue', width=3),
                                    marker=dict(color='blue', size=9),
                                    name='logistic-elasticnet (max)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_2']['x_period'],
                                    y=p_data_auc['model_2']['y_mins'],

                                    line=dict(color='red', width=3),
                                    marker=dict(color='red', size=9),
                                    name='ls-svm (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_2']['x_period'],
                                    y=p_data_auc['model_2']['y_maxs'],

                                    line=dict(color='red', width=3),
                                    marker=dict(color='red', size=9),
                                    name='ls-svm (max)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_3']['x_period'],
                                    y=p_data_auc['model_3']['y_mins'],

                                    line=dict(color='green', width=3),
                                    marker=dict(color='green', size=9),
                                    name='ann-mlp (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data_auc['model_3']['x_period'],
                                    y=p_data_auc['model_3']['y_maxs'],

                                    line=dict(color='green', width=3),
                                    marker=dict(color='green', size=9),
                                    name='ann-mlp (min)',
                                    mode='markers+lines'))

    # Update layout for the background
    fig_ts_auc.update_layout(paper_bgcolor='white', title_font_size=p_theme['p_theme']['font_title'],
                             yaxis=dict(tickvals=np.arange(0, 1.1, 0.1), zeroline=False, automargin=True,
                                        titlefont=dict(size=p_theme['p_theme']['font_axis']+4)),
                             xaxis=dict(titlefont=dict(size=p_theme['p_theme']['font_axis']+4)))

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


# -- ----------------------------------------------------------------------------------- PLOT: ROC + ACU -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_roc_auc(p_casos, p_theme, p_dims):

    # p_casos = casos
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(width=3, dash='dash', color='grey'), x0=0, x1=1, y0=0, y1=1)

    for model in ['model_1', 'model_2', 'model_3']:
        p_fpr = p_casos[model]['auc_max']['data']['metrics']['test']['fpr']
        p_tpr = p_casos[model]['auc_max']['data']['metrics']['test']['tpr']
        fig.add_trace(go.Scatter(x=p_fpr, y=p_tpr, name=model,
                                 mode='lines', line=dict(width=3)))

    fig.show()

    return 1
