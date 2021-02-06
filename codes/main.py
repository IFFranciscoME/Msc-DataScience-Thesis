
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: main.py - python script with the main operations & results                                    -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from rich import print
from rich import inspect

import pandas as pd
import numpy as np

from scipy.sparse.construct import random
from data import ohlc_data as data
from datetime import datetime

import functions as fn
import visualizations as vs
import data as dt

# Reproducible results
import random
random.seed(123)

# --------------------------------------------------------------- PLOT 1: USD/MXN OHLC HISTORICAL PRICES -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# Candlestick chart for historical OHLC prices
plot_1 =vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)

# ------------------------------------------------------------------------------- SHORT DATA DESCRIPTION -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# Table with data description
table_1 = data.describe()

# Missing values
missing_values = 'No missing values (NAs)' if (len(data.notna()) == len(data)) else 'missing'

# ------------------------------------------------------------------- TIMESERIES FOLDS FOR DATA DIVISION -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# Fold size
fold_size = 'bi-year'

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=data, p_period=fold_size)

# -- ----------------------------------------------------------------- PLOT 2: TIME SERIES BLOCK T-FOLDS -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# Dates for vertical lines in the T-Folds plot
dates_folds = []
for n_fold in list(folds.keys()):
    dates_folds.append(folds[n_fold]['timestamp'].iloc[0])
    dates_folds.append(folds[n_fold]['timestamp'].iloc[-1])

# Plot_1 with t-folds vertical lines
plot_2 = vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# Show plot in script
# plot_2.show()

# Generate plot online with chartstudio
# py.plot(plot_2)

# -- --------------------------------------------------------------------------- FOLD EVALUATION PROCESS -- #
# -- --------------------------------------------------------------------------- ----------------------- -- #

# List with the names of the models
ml_models = list(dt.models.keys())

# File name to save the data
file_name = 'files/pickle_rick/genetic_net_' + fold_size + '.dat'

# ---------------------------------------------------------------- WARNING: TAKES HOURS TO RUN THIS PART -- #
# Measure the begining of the code execution process
ini_time = datetime.now()
print(ini_time)

# Feature engineering + hyperparameter optimization + model metrics for every fold
memory_palace = fn.fold_evaluation(p_data_folds=folds, p_models=ml_models, p_saving=True, p_file=file_name,
                                   p_fit_type='inv-weighted', p_scaling='post-feature', p_transform='robust')

# Measure the end of the code execution process
end_time = datetime.now()
print(end_time)
# ------------------------------------------------------------------------------------------------------ -- #

# Load previously generated data
# memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load', p_data_file=file_name)
# memory_palace = memory_palace['memory_palace']

# -- ------------------------------------------------------------------------------- PARAMETER SET CASES -- #
# -- ------------------------------------------------------------------------------- ------------------- -- #

# -- Min, max and mode AUC cases
auc_cases = fn.model_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=folds,
                         p_cases_type='inv-weighted')

# -- ------------------------------------------------------------------------ SYMBOLIC FEATURES ANALYSIS -- #
# -- ------------------------------------------------------------------------ -------------------------- -- #

# memory_palace['logistic-elasticnet']['b_y_0']['sym_features']['best_programs']
# (pending) parsimony metric
# (pending) fitness metric
# (pending) equations description

# -- ----------------------------------------------------------------------------- ALL FEATURES ANALYSIS -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# (pending) correlation
# (pending) outliers
# (pending) symmetry

# -- --------------------------------------------------------------- PLOT 3: CLASSIFICATION FOLD RESULTS -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# Pick case
case = 'max'

# Pick model to generate the plot
auc_model = 'logistic-elasticnet'

# Generate title
auc_title = 'in Fold max AUC for: ' + auc_model + ' found in period: ' + \
             auc_cases[auc_model]['auc_max']['period']

# Plot title
dt.theme_plot_3['p_labels']['title'] = auc_title

# Get data from auc_cases
train_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['train']
test_y = auc_cases[auc_model]['auc' + '_' + case]['data']['results']['data']['test']

# Get data for prices and predictions
ohlc_prices = folds[auc_cases[auc_model]['auc' + '_' + case]['period']]
ohlc_class = {'train_y': train_y['y_train'], 'train_y_pred': train_y['y_train_pred'],
              'test_y': test_y['y_test'], 'test_y_pred': test_y['y_test_pred']}

# Make plot
plot_3 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, p_vlines=None)

# Show plot in script
# plot_3.show()

# Generate plot online with chartstudio
# py.plot(plot_3)

# -- -------------------------------------------------------------------- PLOT 4: ROC & AUC FOLD RESULTS -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# Plot title
dt.theme_plot_4['p_labels']['title'] = 'in Fold max/min AUC cases'

# Timeseries of the AUCs
plot_4 = vs.g_roc_auc(p_cases=auc_cases, p_type='test', p_models=ml_models, p_theme=dt.theme_plot_4)

# Show plot in script
# plot_4.show()

# Generate plot online with chartstudio
# py.plot(plot_4)

# -- ------------------------------------------------------------------- GLOBAL EVALUATION FOR AUC CASES -- #
# -- ------------------------------------------------------------------- ------------------------------- -- #

# Case to evaluate
fold_case = 'auc_max'
# Model to evaluate
fold_model = 'logistic-elasticnet'
# Function
global_model = fn.global_evaluation(p_memory_palace=memory_palace, p_data=data, p_cases=auc_cases, 
                                    p_model=fold_model, p_case=fold_case)

# Model parameters
global_model['global_parameters']

# Model auc
global_model['model']['metrics']['train']['auc']

# Model accuracy
global_model['model']['metrics']['train']['acc']

# -- ------------------------------------------------------------- PLOT 5: GLOBAL CLASSIFICATION RESULTS -- #
# -- ------------------------------------------------------------- ------------------------------------- -- #

# Get data for prices and predictions
ohlc_prices = data

ohlc_class = {'train_y': global_model['model']['results']['data']['train']['y_train'],
              'train_y_pred': global_model['model']['results']['data']['train']['y_train_pred'],
              'test_y': global_model['model']['results']['data']['test']['y_test'],
              'test_y_pred': global_model['model']['results']['data']['test']['y_test_pred']}

# Plot title
dt.theme_plot_3['p_labels']['title'] = 'Global results with t-fold optimized parameters'

# Make plot
plot_5 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, p_vlines=None)

# Show plot in script
# plot_5.show()

# Generate plot online with chartstudio
# py.plot(plot_5)
