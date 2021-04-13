
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: notebook.py - python script with all the output elements for a jupyter notebook               -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Import other scripts
import functions as fn
import visualizations as vs
import data as dt

# -- basic functions
import pandas as pd
import numpy as np
import random

# -- file operations
from os import listdir, path
from os.path import isfile, join

# -- complementary
from rich import print
from rich import inspect

# Reproducible results
random.seed(123)

# ---------------------------------------------------------- LOAD PRICES, FOLDS AND PROCESS RESULTS DATA -- #
# ------------------------------------------------------------------------------------------------------ -- #

# Route to backup files folder
dir_route = 'files/backups/ludwig/test_1_09042021/'

# Available files with experiment data
abspath = path.abspath(dir_route)
experiment_files = sorted([f for f in listdir(abspath) if isfile(join(abspath, f))])

# Experiments to show
# [11, 0.9, 0.5, 'all']

# Experiment file  
experiment = 11

# Fold case
fold_case = dt.fold_cases[experiment_files[experiment][0]]

# Final route
file_route = dir_route + experiment_files[experiment]

# Historical prices
historical_prices = dt.ohlc_data

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=historical_prices, p_period=fold_case)

# Load data (previously generated results)
memory_palace = dt.data_pickle(p_data_objects=None, p_data_action='load', p_data_file=file_route)
memory_palace = memory_palace['memory_palace']

# List with the names of the models
ml_models = list(dt.models.keys())

# -- -------------------------------------------------------------------- TEXT DESCRIPTION OF EXPERIMENT -- #
# -- ------------------------------------------------------------------------------------- ------------- -- #

# Description of results for the founded result
# TEXT Fold size, Cost function, feature transformation, train-val proportion, embargo

# -- -------------------------------------------------------------------- PLOT TIME SERIES BLOCK T-FOLDS -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# Dates for vertical lines in the T-Folds plot
dates_folds = []
for n_fold in list(folds.keys()):
    dates_folds.append(folds[n_fold]['timestamp'].iloc[0])
    dates_folds.append(folds[n_fold]['timestamp'].iloc[-1])

# Plot_1 with t-folds vertical lines
plot_2 = vs.plot_ohlc(p_ohlc=historical_prices, p_theme=dt.theme_plot_2, p_vlines=dates_folds)

# Show plot in script
# plot_2.show()

# Generate plot online with chartstudio
# py.plot(plot_2)

# --------------------------------------------------------------------- EXPERIMENT 1: OOS GENERALIZATION -- #
# --------------------------------------------------------------------------------------------------------- #

# Filtered cases
filters = {'filter_1': {'metric': 'acc-train', 'objective': 'above_threshold', 'threshold': 0.90},
           'filter_2': {'metric': 'acc-val', 'objective': 'above_threshold', 'threshold': 0.50},
           'filter_3': {'metric': 'acc-diff', 'objective': 'all'}}

# metric type for MIN, MAX, MODE
metric_case = 'acc-train'

# -- get MIN, MAX, MODE, FILTERED Cases
met_cases = fn.model_cases(p_models=ml_models, p_global_cases=memory_palace, 
                           p_data_folds=folds, p_cases_type=metric_case, p_filters=filters)
 
for i in range(0, len(ml_models)):
    model_case = ml_models[i]
    
    # periods with at least 1 matching case
    filtered_periods = met_cases[model_case]['met_filter']['period']

    # in case of at least 1 period found
    if len(filtered_periods) > 0:
        data = []
        # concatenate all ocurrences
        for i_period in filtered_periods:
            data.append(met_cases[model_case]['met_filter']['data'][i_period]['metrics'])
            df_filtered = pd.concat(data, axis=1)

        # format to output dataframe
        df_filtered.index.name=model_case + '-metrics'
        df_filtered.head()

# -- ------------------------------------------------------------------------------------- DATA PROFILES -- #
# -- ------------------------------------------------------------------------------------- ------------- -- #

# Fold to make a description
des_fold = df_filtered.columns[0][:-2]

# TABLE data profile (Target)
exp_train_y = memory_palace[des_fold]['features']['train_y']
exp_val_y = memory_palace[des_fold]['features']['val_y']

tabla_1 = fn.data_profile(p_data=exp_train_y, p_type='target', p_mult=10000)
tabla_2 = fn.data_profile(p_data=exp_val_y, p_type='target', p_mult=10000)

# TABLE data profile (Inputs)
exp_train_x = memory_palace[des_fold]['features']['train_x']
exp_val_x = memory_palace[des_fold]['features']['val_x']

tabla_3 = fn.data_profile(p_data=exp_train_x, p_type='target', p_mult=10000)
tabla_4 = fn.data_profile(p_data=exp_val_x, p_type='target', p_mult=10000)

# -- ----------------------------------------------------------------------- PLOT: MULTI PLOT HISTOGRAMS -- #
# -- ---------------------------------------------------------------------------- ---------------------- -- #

# PLOT histogram (Features)
plot_2_1 = vs.plot_h_histograms(p_data=exp_train_x.iloc[:, 0:9])

# Show plot
plot_2_1.show()

# PLOT histogram (Features)
plot_2_2 = vs.plot_h_histograms(p_data=exp_train_x.iloc[:, -10:])

# Show plot
plot_2_2.show()

# -- ------------------------------------------------------------------------ PLOT: HEATMAP CORRELATIONS -- #
# -- ---------------------------------------------------------------------------- ---------------------- -- #

# -- Target and Auto regressive Features correlation
exp_1 = pd.concat([exp_train_y.copy(), exp_train_x.iloc[:, 0:55].copy()], axis=1)
exp_1_corr_p = exp_1.corr('pearson')
title_txt = 'Linear-Autoregressive Features Vs Target Correlation (pearson)'
exp_1_plot = vs.plot_heatmap_corr(p_data=exp_1_corr_p.copy(), p_title=title_txt)

# Show plot
exp_1_plot.show()

# -- Target and Symbolic Features correlation
exp_2 = pd.concat([exp_train_y.copy(), exp_train_x.iloc[:, -40:].copy()], axis=1)
exp_2_corr_p = exp_1.corr('pearson')
title_txt = 'Symbolic Features Vs Target Correlation (pearson)'
exp_2_plot = vs.plot_heatmap_corr(p_data=exp_1_corr_p.copy(), p_title=title_txt)

# Show plot
exp_2_plot.show()

# -- ---------------------------------------------------------------------------- PLOT: All ROCs in FOLD -- #
# -- ---------------------------------------------------------------------------- ---------------------- -- #

# case to plot
case = 'met_max'

# data subset to use
subset = 'train'

# metric to use
metric_case = 'acc-train'

# Model to evaluate
model_case = 'ann-mlp'

# period 
period_case = 'y_2012'

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
plot_4 = vs.plot_multiroc(p_data=d_plot_4, p_metric=metric_case, p_theme=dt.theme_plot_4)

# Show plot in script
plot_4.show()

# Generate plot online with chartstudio
# py.plot(plot_4)

# -- ----------------------------------------------------------------- PLOT: CLASSIFICATION FOLD RESULTS -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

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
plot_3 = vs.plot_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_3, p_data_class=ohlc_class, 
                            p_vlines=date_vlines)

# Show plot in script
plot_3.show()

# Generate plot online with chartstudio
# py.plot(plot_3)

# --------------------------------------------------------------------------------- MIN, MAX, MODE CASES -- #
# --------------------------------------------------------------------------------------------------------- #

# models to explore results
model_case = ml_models[0]

# period of the best of HoF: according to model_case and metric_case 
maxcase_period = met_cases[model_case]['met_max']['period']
maxcase_params = met_cases[model_case]['met_max']['params']
maxcase_metric = met_cases[model_case]['met_max'][metric_case]

# output DataFrame
df_max = pd.DataFrame([met_cases[model_case]['met_max']['data']['pro-metrics']]).T
df_max.columns = [maxcase_period]
df_max.index.name=model_case + '-metrics'
df_max.head(5)

# period of the worst of HoF: according to model_case and metric_case 
mincase_period = met_cases[model_case]['met_min']['period']
mincase_params = met_cases[model_case]['met_min']['params']
mincase_metric = met_cases[model_case]['met_min'][metric_case]

# output DataFrame
df_min = pd.DataFrame([met_cases[model_case]['met_min']['data']['pro-metrics']]).T
df_min.columns = [mincase_period]
df_min.index.name=model_case + '-metrics'
# df_min.head(5)

# Modes and their params, no. of repetitions and periods.
mode_repetitions = pd.DataFrame(met_cases[model_case]['met_mode']['data']).T
# mode_repetitions.head(5)