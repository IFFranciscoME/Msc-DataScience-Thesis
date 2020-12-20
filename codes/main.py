
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
from datetime import datetime

# Two periods (fast run)
data = pd.concat([dt.price_data[list(dt.price_data.keys())[0]], dt.price_data[list(dt.price_data.keys())[1]]])

# All periods (slow run)
# data = pd.concat([dt.price_data[list(dt.price_data.keys())[0]], dt.price_data[list(dt.price_data.keys())[1]],
#                  dt.price_data[list(dt.price_data.keys())[2]], dt.price_data[list(dt.price_data.keys())[3]],
#                  dt.price_data[list(dt.price_data.keys())[4]], dt.price_data[list(dt.price_data.keys())[5]],
#                  dt.price_data[list(dt.price_data.keys())[6]], dt.price_data[list(dt.price_data.keys())[7]],
#                  dt.price_data[list(dt.price_data.keys())[8]], dt.price_data[list(dt.price_data.keys())[9]]])

# --------------------------------------------------------------- PLOT 1: USD/MXN OHLC HISTORICAL PRICES -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# Candlestick chart for historical OHLC prices
plot_1 =vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
plot_1.show()

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
t_folds = fn.t_folds(p_data=data, p_period='quarter')
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
# plot_2 = vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# Show plot in script
# plot_2.show()

# Generate plot online with chartstudio
# py.plot(plot_2)

# -- ------------------------------------------------------------------ PROCESS: Fold Evaluation Process -- #
# -- ------------------------------------------------------------------ -------------------------------- -- #

# List with the names of the models
ml_models = list(dt.models.keys())

# File name to save the data
file_name = 'genetic_net_quarter.dat'

# ---------------------------------------------------------------- WARNING: TAKES HOURS TO RUN THIS PART -- #
# Measure the begining of the code execution process
ini_time = datetime.now()
print(ini_time)

# Main code to produce global evaluation for every t-fold for every model ()
memory_palace = fn.fold_evaluation(p_data_folds=t_folds, p_models=ml_models, 
                                   p_saving=False, p_file_name=file_name)
# Save data
dt.data_save_load(p_data_objects=memory_palace, p_data_action='save', p_data_file=file_name)

# Measure the end of the code execution process
end_time = datetime.now()
print(end_time)
# ------------------------------------------------------------------------------------------------------ -- #

# Load previously generated data
# memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load',
#                                   p_data_file='files/pickle_rick/' + file_name)

# -- -------------------------------------------------------------------- PROCESS: AUC min and max cases -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# min and max AUC cases for the models
auc_cases = fn.model_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=t_folds)

# -- ----------------------------------------------------------------- PROCESS: Model Global Performance -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# # model performance for all models, with the min and max AUC parameters
global_evaluations = fn.global_evaluation(p_data=data, p_memory=7, p_global_cases=memory_palace,
                                          p_models=ml_models, p_cases=auc_cases)

# -- ----------------------------------------------------------------------------- AUC Min and Max cases -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# min and max AUC cases for the models
auc_cases = fn.model_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=t_folds)

minmax_auc_test = {i: {'x_period': [], 'y_mins': [], 'y_maxs': []} for i in ml_models}

# get the cases where auc was min and max in all the periods
for model in ml_models:
    minmax_auc_test[model]['x_period'] = list(auc_cases[model]['hof_metrics']['data'].keys())
    minmax_auc_test[model]['y_mins'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_min']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]
    minmax_auc_test[model]['y_maxs'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_max']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]

