
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Script: main.py : python script with the main functionality                                         -- #
# -- Author: IFFranciscoME                                                                               -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository:                                                                                         -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from scipy.sparse.construct import random
from sklearn.metrics import auc
import pandas as pd
import functions as fn
import visualizations as vs
import data as dt

import random
import numpy as np
from data import ohlc_data as data
from datetime import datetime

# reproducible results
random.seed(123)

# --------------------------------------------------------------- PLOT 1: USD/MXN OHLC HISTORICAL PRICES -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# Candlestick chart for historical OHLC prices
plot_1 =vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)

# ---------------------------------------------------------------------- TABLE 1: SHORT DATA DESCRIPTION -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# table with data description
table_1 = data.describe()

# ------------------------------------------------------------------- TIMESERIES FOLDS FOR DATA DIVISION -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# Timeseries data division in t-folds
t_folds = fn.t_folds(p_data=data, p_period='year')

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# Dates for vertical lines in the T-Folds plot
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

# -- ------------------------------------------------------------------ PROCESS: Fold Evaluation Process -- #
# -- ------------------------------------------------------------------ -------------------------------- -- #

# List with the names of the models
ml_models = list(dt.models.keys())

# File name to save the data
file_name = 'files/pickle_rick/genetic_net_year.dat'

# ---------------------------------------------------------------- WARNING: TAKES HOURS TO RUN THIS PART -- #
# Measure the begining of the code execution process
# ini_time = datetime.now()
# print(ini_time)

# Feature engineering + hyperparameter optimization + model metrics for every fold
# memory_palace = fn.fold_evaluation(p_data_folds=t_folds, p_models=ml_models,
#                                    p_saving=True, p_file_name=file_name)

# Measure the end of the code execution process
# end_time = datetime.now()
# print(end_time)
# ------------------------------------------------------------------------------------------------------ -- #

# Load previously generated data
memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load', p_data_file=file_name)
memory_palace = memory_palace['memory_palace']

# -- ---------------------------------------------------------------------- PROCESS: AUC min & max cases -- #
# -- ---------------------------------------------------------------------- ---------------------------- -- #

# min and max AUC cases for the models
auc_cases = fn.model_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=t_folds)

# -- --------------------------------------------------------------- PLOT 3: Classification Fold Results -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# pick case
case = 'max'

# pick model to generate the plot
auc_model = 'logistic-elasticnet'

# generate title
auc_title = 'max AUC for: ' + auc_model + ' found in period: ' + auc_cases[auc_model]['auc_max']['period']

# plot title
dt.theme_plot_3['p_labels']['title'] = auc_title

# get data from auc_cases
train_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['train']
test_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['test']

# get data for prices and predictions
ohlc_prices = t_folds[auc_cases[auc_model]['auc' + '_' + case]['period']]
ohlc_class = {'train_y': train_y['y_train'], 'train_y_pred': train_y['y_train_pred'],
              'test_y': test_y['y_test'], 'test_y_pred': test_y['y_test_pred']}

# make plot
plot_3 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, p_vlines=None)

# visualize plot
# plot_3.show()

# -- ------------------------------------------------------------------ PLOT 4: ROC and AUC Fold Results -- #
# -- ------------------------------------------------------------------ -------------------------------- -- #

# plot title
dt.theme_plot_4['p_labels']['title'] = 'max/min AUC cases'

# Timeseries of the AUCs
plot_4_folds = vs.g_roc_auc(p_cases=auc_cases, p_type='test', p_models=ml_models, p_theme=dt.theme_plot_4)

# offline plot
# plot_4_folds.show()

# -- --------------------------------------------------------- Global Evaluation for AUC min & max Cases -- #
# -- --------------------------------------------------------- ----------------------------------------- -- #

# case to evaluate
fold_case = 'auc_min'
# model to evaluate
fold_model = 'logistic-elasticnet'
# function
global_model = fn.global_evaluation(p_memory_palace=memory_palace, p_data=data, p_cases=auc_cases, 
                                    p_model=fold_model, p_case=fold_case)

# parameters
# global_model['global_parameters']

# auc
# global_model['model']['metrics']['test']['auc']

# accuracy
# global_model['model']['metrics']['test']['acc']

# -- ------------------------------------------------------------- PLOT 5: Global Classification Results -- #
# -- ------------------------------------------------------------- ------------------------------------- -- #

# get data for prices and predictions
ohlc_prices = data

ohlc_class = {'train_y': global_model['model']['results']['data']['train']['y_train'],
              'train_y_pred': global_model['model']['results']['data']['train']['y_train_pred'],
              'test_y': global_model['model']['results']['data']['test']['y_test'],
              'test_y_pred': global_model['model']['results']['data']['test']['y_test_pred']}

# plot title
dt.theme_plot_3['p_labels']['title'] = 'Global results with t-fold optimized parameters'

# make plot
plot_5 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, p_vlines=None)

# visualize plot
# plot_5.show()
