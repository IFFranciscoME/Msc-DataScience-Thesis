
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: document.py - python script with all the output elements for the thesis report                -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from rich import print
from rich import inspect

import pandas as pd
import numpy as np

from data import ohlc_data as data
from datetime import datetime

import functions as fn
import visualizations as vs
import data as dt

# Reproducible results
import random
random.seed(123)

# ---------------------------------------------------------------- PLOT 1: OHLC DATA PLOT (ALL DATA SET) -- #
# ---------------------------------------------------------------- ------------------------------------- -- #

# Candlestick chart for historical OHLC prices
plot_1 =vs.g_ohlc(p_ohlc=data, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)

# ------------------------------------------------------------------------------------ LOAD RESULTS DATA -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

# Fold size
fold_size = 'semester'

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=data, p_period=fold_size)

# List with the names of the models
ml_models = list(dt.models.keys())

# File name to save the data
file_name = 'files/pickle_rick/s_weighted_post-features_scale.dat'

# Load previously generated data
memory_palace = dt.data_save_load(p_data_objects=None, p_data_action='load', p_data_file=file_name)
memory_palace = memory_palace['memory_palace']

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

# ----------------------------------------------------------------------------------------- DATA PROFILE -- #
# ----------------------------------------------------------------------------------------- ------------ -- #

# period to explore results
period = list(folds.keys())[0]
period = 's_01_2011'

# models to explore results
model = list(dt.models.keys())[0]

# input data profile
in_profile = memory_palace[period]['metrics']['data_metrics']

# target variable
tv_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_y']
tv_profile_test = memory_palace[period]['metrics']['feature_metrics']['test_y']

# amount of symbolic features
n_sf = dt.symbolic_params['n_features']

# linear features profile
lf_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_x'].iloc[:, :-n_sf]
lf_profile_test = memory_palace[period]['metrics']['feature_metrics']['test_x'].iloc[:, :-n_sf]

# symbolic features profile
sm_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_x'].iloc[:, -n_sf:]
sm_profile_test = memory_palace[period]['metrics']['feature_metrics']['test_x'].iloc[:, -n_sf:]

# all variables correlation table
all_corr_train = memory_palace[period]['features']['train_x'].corr()
all_corr_test = memory_palace[period]['features']['test_x'].corr()

# All variables with target variable
tgv_corr_train = pd.concat([memory_palace[period]['features']['train_y'],
                            memory_palace[period]['features']['train_x']], axis=1).corr().iloc[:, 0]
tgv_corr_test = pd.concat([memory_palace[period]['features']['test_y'],
                           memory_palace[period]['features']['test_x']], axis=1).corr().iloc[:, 0]

# --------------------------------------------------------------------------------------- VISUAL PROFILE -- #
# ----------------------------------------------------------------------------------------- ------------ -- #

# -- ------------------------------------------------------------------------------- PARAMETER SET CASES -- #
# -- ------------------------------------------------------------------------------- ------------------- -- #

# -- Min, max and mode AUC cases
auc_cases = fn.model_auc(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=folds,
                         p_cases_type='logloss-mean')

# -- -------------------------------------------------------------------------------------- DATA PROFILE -- #
# -- -------------------------------------------------------------------------------------- ------------ -- #

# Input data
# Features
# Target
 
 # memory_palace[period]['metrics']
 
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
