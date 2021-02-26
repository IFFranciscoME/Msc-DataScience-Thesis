
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
fold_case = 'semester'

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=data, p_period=fold_case)

# List with the names of the models
# ml_models = list(dt.models.keys())
ml_models = ['ann-mlp']

# File name to save the data
file_name = 'files/pickle_rick/s_logloss-train_robust_post-features_20.dat'

# Load previously generated data
memory_palace = dt.data_pickle(p_data_objects=None, p_data_action='load', p_data_file=file_name)
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

# ------------------------------------------------------------------------------------------- input data -- # 

# all the input data
in_profile = memory_palace[period]['metrics']['data_metrics']

# -------------------------------------------------------------------------------------- target variable -- #

# train and val data sets with only target variable
tv_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_y']
# tv_profile_val = memory_palace[period]['metrics']['feature_metrics']['val_y']

# ------------------------------------------------------------------------------------- linear variables -- #
# amount of symbolic features
n_sf = dt.symbolic_params['n_features']

# train and val data sets with only autoregressive variables
lf_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_x'].iloc[:, :-n_sf]
# lf_profile_val = memory_palace[period]['metrics']['feature_metrics']['val_x'].iloc[:, :-n_sf]

# ------------------------------------------------------------------------------------ symbolic variables -- #

# train and val data sets with only symbolic variables
sm_profile_train = memory_palace[period]['metrics']['feature_metrics']['train_x'].iloc[:, -n_sf:]
# sm_profile_val = memory_palace[period]['metrics']['feature_metrics']['val_x'].iloc[:, -n_sf:]

# ---------------------------------------------------------------------------------------- All variables -- #

# correlation among all variables
all_corr_train = memory_palace[period]['features']['train_x'].corr()
# all_corr_val = memory_palace[period]['features']['val_x'].corr()

# correlation of all variables with target variable
tgv_corr_train = pd.concat([memory_palace[period]['features']['train_y'],
                            memory_palace[period]['features']['train_x']], axis=1).corr().iloc[:, 0]
# tgv_corr_val = pd.concat([memory_palace[period]['features']['val_y'],
#                            memory_palace[period]['features']['val_x']], axis=1).corr().iloc[:, 0]

# --------------------------------------------------------------------------------------- VISUAL PROFILE -- #
# ----------------------------------------------------------------------------------------- ------------ -- #

# (pending)

# -- ------------------------------------------------------------------------------- PARAMETER SET CASES -- #
# -- ------------------------------------------------------------------------------- ------------------- -- #

# metric type (all the available in iter_opt['fitness'])
metric_case = 'acc-train'

# models to explore results
model_case = 'ann-mlp'

# -- Min, max and mode cases
met_cases = fn.model_cases(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=folds,
                           p_cases_type=metric_case)
 
# period of the best of HoF: according to model_case and metric_case 
best_period = met_cases[model_case]['met_max']['period']
best_params = met_cases[model_case]['met_max']['params']
best_metric = met_cases[model_case]['met_max'][metric_case]

# period of the worst of HoF: according to model_case and metric_case 
worst_period = met_cases[model_case]['met_min']['period']
worst_params = met_cases[model_case]['met_min']['params']
worst_metric = met_cases[model_case]['met_min'][metric_case]

# Modes and their params, no. of repetitions and periods.
mode_repetitions = pd.DataFrame(met_cases[model_case]['met_mode']['data']).T

# -- ------------------------------------------------------------------------ SYMBOLIC FEATURES ANALYSIS -- #
# -- ------------------------------------------------------------------------ -------------------------- -- #

# period to explore results
period_case = 's_01_2009'

# models to explore results
model_case = 'ann-mlp'

# data
sym_data = met_cases[model_case]['hof_metrics']['data'][period_case]['features']['sym_features']

# programs data table
sym_programs = sym_data['best_programs']

# parsimony metrics
sym_depth = sym_data['best_programs']['depth']
sym_length = sym_data['best_programs']['length']

# fitness metric
sym_fitness = sym_data['best_programs']['fitness']

# -- --------------------------------------------------------------- PLOT 3: CLASSIFICATION FOLD RESULTS -- #
# -- ----------------------------------------------------------------------------- --------------------- -- #

# Pick case
case = 'met_max'

# Pick model to generate the plot
model_case = 'ann-mlp'

# Generate title
plot_title = 'inFold ' + case + ' for: ' + model_case + ' ' + met_cases[model_case][case]['period']

# Plot title
dt.theme_plot_3['p_labels']['title'] = plot_title

# Get data from met_cases
train_y = met_cases[model_case][case]['data']['results']['data']['train']

# Get data for prices and predictions
ohlc_prices = folds[met_cases[model_case][case]['period']]

ohlc_class = {'train_y': train_y['train_y'], 'train_y_pred': train_y['train_pred_y']}

# Dates for vertical lines in the T-Folds plot
date_vlines = [ohlc_class['train_y'].index[-1]]

# Make plot
plot_3 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, 
                         p_vlines=date_vlines)

# Show plot in script
# plot_3.show()

# Generate plot online with chartstudio
# py.plot(plot_3)

# -- -------------------------------------------------------------------------- PLOT 4: All ROCs in FOLD -- #
# -- -------------------------------------------------------------------------- ------------------------ -- #

# case to plot
case = 'met_max'

# data subset to use
subset = 'train'

# metric to use
metric_case = 'logloss-mean'

# Model to evaluate
model_case = 'ann-mlp'

# period 
period_case = 's_01_2009'

# parameters of the evaluated models
d_params = memory_palace[period_case][model_case]['p_hof']['hof']

# get all fps and tps for a particular model in a particular fold
d_plot_4 = {i: {'tpr': memory_palace[period_case][model_case]['e_hof'][i]['metrics'][subset]['tpr'],
                'fpr': memory_palace[period_case][model_case]['e_hof'][i]['metrics'][subset]['fpr'],
                metric_case: memory_palace[period_case][model_case]['e_hof'][i]['pro-metrics'][metric_case]}
            for i in range(0, len(d_params))}

# Plot title
dt.theme_plot_4['p_labels']['title'] = 'in Fold max & min ' + metric_case + ' ' + subset + ' data'

# Timeseries of the AUCs
plot_4 = vs.g_multiroc(p_data=d_plot_4, p_metric=metric_case, p_theme=dt.theme_plot_4)

# Show plot in script
# plot_4.show()

# Generate plot online with chartstudio
# py.plot(plot_4)

# -- --------------------------------------------------------------------------------- GLOBAL EVALUATION -- #
# -- --------------------------------------------------------------------------------- ----------------- -- #

# metric type (all the available in iter_opt['fitness'])
metric_case = 'acc-mean'

# Model to evaluate
model_case = 'ann-mlp'

# period 
period_case = 'y_01_2011'

# data subset to use
subset = 'train'

# Min, max and mode cases
met_cases = fn.model_cases(p_models=ml_models, p_global_cases=memory_palace, p_data_folds=folds,
                           p_cases_type=metric_case)

# Global Evaluation for a particular type of case
global_models = fn.global_evaluation(p_case=memory_palace[period_case][model_case],
                                     p_global_data=data,
                                     p_features=memory_palace[period_case],
                                     p_model=model_case)

# the evaluation of the best model
global_model = global_models[0]

# Model parameters
global_model['global_parameters']

# Model auc
global_model['model']['pro-metrics'][metric_case]

# Model accuracy
global_model['model']['pro-metrics']['acc-' + subset]

# Model logloss
global_model['model']['pro-metrics']['logloss-mean']

# -- ------------------------------------------------------------- PLOT 5: GLOBAL CLASSIFICATION RESULTS -- #
# -- ------------------------------------------------------------- ------------------------------------- -- #

# Get data for prices and predictions
ohlc_prices = data

# data for plot
ohlc_class = {'val_y': global_model['model']['results']['data']['val']['val_y'],
              'val_y_pred': global_model['model']['results']['data']['val']['val_pred_y']}

# Plot title
dt.theme_plot_3['p_labels']['title'] = 'Global results with t-fold optimized parameters'

# Dates for vertical lines in the T-Folds plot
date_vlines = [ohlc_class['val_y'].index[-1]]

# Make plot
plot_5 = vs.g_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class,
                         p_vlines=date_vlines)

# Show plot in script
# plot_5.show()

# Generate plot online with chartstudio
# py.plot(plot_5)
