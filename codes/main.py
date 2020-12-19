

"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Script: main.py : python script with the main functionality                                         -- #
# -- Author: IFFranciscoME                                                                               -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository:                                                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from sklearn.metrics import auc
import pandas as pd
import functions as fn
import visualizations as vs
import data as dt

# One period (fast run)
# data = price_data[list(price_data.keys())[9]]

# All periods (slow run)
data = pd.concat([dt.price_data[list(dt.price_data.keys())[0]], dt.price_data[list(dt.price_data.keys())[1]],
                  dt.price_data[list(dt.price_data.keys())[2]], dt.price_data[list(dt.price_data.keys())[3]],
                  dt.price_data[list(dt.price_data.keys())[4]], dt.price_data[list(dt.price_data.keys())[5]],
                  dt.price_data[list(dt.price_data.keys())[6]], dt.price_data[list(dt.price_data.keys())[7]],
                  dt.price_data[list(dt.price_data.keys())[8]], dt.price_data[list(dt.price_data.keys())[9]]])

# --------------------------------------------------------------- PLOT 1: USD/MXN OHLC HISTORICAL PRICES -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# Candlestick chart for historical OHLC prices
plot_1 = vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_1, p_dims=dt.theme_plot_1['p_dims'],
                   p_labels=dt.theme_plot_1['p_labels'], p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)

# ---------------------------------------------------------------------- TABLE 1: SHORT DATA DESCRIPTION -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# table with data description
table_1 = data.describe()

# --------------------------------------------------------------------- Division en K-Folds por periodos -- #
# --------------------------------------------------------------------- -------------------------------- -- #

# Division de periodos de datos, sin filtracion
# en amplitudes de 'trimestre' para obtener 4 folds por cada año
t_folds = fn.f_m_folds(p_data=data, p_periodo='trimestre')
# eliminar el ultimo trimestre por que estara incompleto
t_folds.pop('q_04_2020', None)

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# construccion de fechas para lineas verticales de division de cada fold
dates_folds = []
for fold in list(t_folds.keys()):
    dates_folds.append(t_folds[fold]['timestamp'].iloc[0])
    dates_folds.append(t_folds[fold]['timestamp'].iloc[-1])

# plot_1 with t-folds vertical lines
plot_2 = vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# Show plot in script
# plot_2.show()

# Generate plot online with chartstudio
# py.plot(plot_2)


# list with the names of the models
ml_models = list(dt.models.keys())
