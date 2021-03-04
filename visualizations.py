
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: visualizaciones.py - python script with the main functionality                                -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import chart_studio
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
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=60, pad=20),
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
    """

    """

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
    train_val_error = []
    train_val_success = []

    if 'train_y' in list(p_data_class.keys())[0]:

        # p_ohlc has all the prices and p_data_class has the prediction classes
        # since p_data_class is smaller than p_ohlc, a lagged shift is needed
        feature_lag = int(np.where(p_ohlc['timestamp'] == p_data_class['train_y'].index[0])[0]) 
        ohlc_lag = list(np.arange(0, feature_lag, 1))
        
        # add vertical line to indicate where ends the ohlc lag for feature engineering
        p_vlines.append(p_ohlc['timestamp'][feature_lag])
        p_vlines = sorted(p_vlines)

        # error and success in train
        for row_e in np.arange(0, len(p_data_class['train_y'].index.to_list()), 1):
            if p_data_class['train_y'][row_e] != p_data_class['train_y_pred'][row_e]:
                train_val_error.append(feature_lag + row_e)
            else:
                train_val_success.append(feature_lag + row_e)

        # accuracy in train data set
        train_val_acc = round(len(train_val_success) / (len(train_val_error) + len(train_val_success)), 2)

    # ------------------------------------------------------------------------------ In case of val set -- #
    
    if 'val_y' in list(p_data_class.keys())[0]:
        
        # p_ohlc has all the prices and p_data_class has the prediction classes
        # since p_data_class is smaller than p_ohlc, a lagged shift is needed
        feature_lag = int(np.where(p_ohlc['timestamp'] == p_data_class['val_y'].index[0])[0]) 
        ohlc_lag = list(np.arange(0, feature_lag, 1))
        
        # add vertical line to indicate where ends the ohlc lag for feature engineering
        p_vlines.append(p_ohlc['timestamp'][feature_lag])
        p_vlines = sorted(p_vlines)

        # error and success in val
        for row_s in np.arange(0, len(p_data_class['val_y'].index.to_list()), 1):
            if p_data_class['val_y'][row_s] != p_data_class['val_y_pred'][row_s]:
                train_val_error.append(feature_lag + row_s)
            else:
                train_val_success.append(feature_lag + row_s)

        # overall accuracy
        train_val_acc = round(len(train_val_success) / (len(train_val_error) + len(train_val_success)), 2)


    # Instantiate a figure object
    fig_g_ohlc = go.Figure()    

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=60, pad=20),
                             xaxis=dict(title_text=p_labels['x_title']),
                             yaxis=dict(title_text=p_labels['y_title']))

    # Add layer for coloring in gray the non predicted candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in ohlc_lag],
        open=[p_ohlc['open'].iloc[i] for i in ohlc_lag],
        high=[p_ohlc['high'].iloc[i] for i in ohlc_lag],
        low=[p_ohlc['low'].iloc[i] for i in ohlc_lag],
        close=[p_ohlc['close'].iloc[i] for i in ohlc_lag],
        increasing={'line': {'color': 'grey'}},
        decreasing={'line': {'color': 'grey'}},
        name='Feature-Lag'))

    # Add layer for the success based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_val_success],
        open=[p_ohlc['open'].iloc[i] for i in train_val_success],
        high=[p_ohlc['high'].iloc[i] for i in train_val_success],
        low=[p_ohlc['low'].iloc[i] for i in train_val_success],
        close=[p_ohlc['close'].iloc[i] for i in train_val_success],
        increasing={'line': {'color': 'skyblue'}},
        decreasing={'line': {'color': 'skyblue'}},
        name='Prediction Success'))

    # Add layer for the error based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_val_error],
        open=[p_ohlc['open'].iloc[i] for i in train_val_error],
        high=[p_ohlc['high'].iloc[i] for i in train_val_error],
        low=[p_ohlc['low'].iloc[i] for i in train_val_error],
        close=[p_ohlc['close'].iloc[i] for i in train_val_error],
        increasing={'line': {'color': 'red'}},
        decreasing={'line': {'color': 'red'}},
        name='Prediction Error'))

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
                                         'dash': 'dashdot', 'width': 2},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Legend format
    fig_g_ohlc.update_layout(legend=go.layout.Legend(x=.35, y=-.3, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the background
    fig_g_ohlc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text=p_labels['title'] + ' | acc: ' + 
                                                    str(train_val_acc)),
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

def g_timeseries(p_data, p_theme):
    """
    Visualize evolution of a metric for a particular model, with a particular parameter set, 
    and do that for N different cases. 

    Parameters
    ----------
    p_data_auc:dict
        Diccionario con datos para plot de series de tiempo AUC
        p_data_auc = minmax_auc_val

    p_theme: dict
        Diccionario con informacion de tema para plot
        p_theme = theme_plot_4

    Returns
    -------
    fig_ts_auc: plotly
        Objeto tipo plotly para utilizar con .show()

    """

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme['p_labels'] is not None:
        p_labels = p_theme['p_labels']
    else:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    fig_ts_auc = go.Figure()

    fig_ts_auc.update_layout(
        title=dict(x=0.5, text='Grafica 5:' + '<b> ' + p_theme['p_labels']['title'] + ' </b>'),
        xaxis=dict(title_text=p_theme['p_labels']['x_title'],
                   tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])),
        yaxis=dict(title_text=p_theme['p_labels']['y_title'],
                   tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])))

    fig_ts_auc.add_trace(go.Scatter(x=p_data['model_1']['x_period'],
                                    y=p_data['model_1']['y_mins'],

                                    line=dict(color='#004A94', width=3),
                                    marker=dict(color='#004A94', size=9),
                                    name='logistic-elasticnet (min)',
                                    mode='markers+lines'))

    fig_ts_auc.add_trace(go.Scatter(x=p_data['model_1']['x_period'], fillcolor='blue',
                                    y=p_data['model_1']['y_maxs'],

                                    line=dict(color='#004A94', width=3),
                                    marker=dict(color='#004A94', size=9),
                                    name='logistic-elasticnet (max)',
                                    mode='markers+lines'))

    # Legend format
    fig_ts_auc.update_layout(legend=go.layout.Legend(x=.35, y=-.3, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the background
    fig_ts_auc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text='<b> ' + p_labels['title'] + ' </b>'),
                             yaxis=dict(title=p_labels['y_title'],
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)),
                             xaxis=dict(title=p_labels['x_title'], rangeslider=dict(visible=False),
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)))

    # Formato de tamanos
    fig_ts_auc.layout.autosize = True
    fig_ts_auc.layout.width = p_theme['p_dims']['width']
    fig_ts_auc.layout.height = p_theme['p_dims']['height']

    return fig_ts_auc


# -- ------------------------------------------------------------------------------------ PLOT: ALL ROCs -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_multiroc(p_data, p_theme, p_metric):

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme['p_labels'] is not None:
        p_labels = p_theme['p_labels']
    else:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # p_casos = casos
    fig_rocs = go.Figure()

    # Layout for margin, and both x and y axes
    fig_rocs.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=60, pad=20),
                           xaxis=dict(title_text=p_labels['x_title']),
                           yaxis=dict(title_text=p_labels['y_title']))
    
    fig_rocs.update_layout(
        title=dict(x=0.5, text=p_theme['p_labels']['title']),
        xaxis=dict(title_text=p_theme['p_labels']['x_title'],
                   tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])),
        yaxis=dict(title_text=p_theme['p_labels']['y_title'],
                   tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])))

    fig_rocs.add_shape(type='line', line=dict(width=1, dash='dash', color='grey'), x0=0, x1=1, y0=0, y1=1)

    # n line colors (max, min, other)
    line_colors = ['#047CFB', '#FB5D41', '#ABABAB']

    # max metric
    metrics = [p_data[i][p_metric] for i in list(p_data.keys())]
    max_metric = np.argmax(metrics)
    min_metric = np.argmin(metrics)

    for i in list(p_data.keys()):
        model = p_data[i]
        p_fpr = model['fpr']
        p_tpr = model['tpr']
        p_color = line_colors[2]
        p_size = 1
        p_name = p_metric + '_generic: ' +  str(round(metrics[i], 2))
        
        if i == max_metric:
            p_color = line_colors[0]
            p_size = 2
            p_name = p_metric + '_max: ' + str(round(metrics[i], 2))
        elif i == min_metric:
            p_color = line_colors[1]
            p_size = 2
            p_name = p_metric + '_min: ' +  str(round(metrics[i], 2))

        fig_rocs.add_trace(go.Scatter(x=p_fpr, y=p_tpr, name=p_name,
                                        mode='lines', line=dict(width=p_size, color=p_color)))

    # Legend format
    fig_rocs.update_layout(legend=go.layout.Legend(x=1.05, y=1.05, orientation='v',
                                                   bordercolor='dark grey',
                                                   borderwidth=1,
                                                   font=dict(size=p_theme['p_fonts']['font_axis'])))

    # Formato de tamanos
    fig_rocs.layout.autosize = True
    fig_rocs.layout.width = p_theme['p_dims']['width']
    fig_rocs.layout.height = p_theme['p_dims']['height']

    return fig_rocs
