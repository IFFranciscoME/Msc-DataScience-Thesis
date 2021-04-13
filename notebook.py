
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
from sklearn.utils import all_estimators
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

# -- Experiment case -- # 
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

# Description of results for the founded result
# TEXT Fold size, Cost function, feature transformation, train-val proportion, embargo

# Descriptive text of experiment
experiment_des = fn.experiment_text(p_filename=experiment_files[experiment])

# TABLE data profile (Target)
# memory_palace['y_2012']['features']['train_y']
# memory_palace['y_2012']['features']['val_y']

# PLOT histogram (Target)
# memory_palace['y_2012']['features']['train_y']
# memory_palace['y_2012']['features']['val_y']

# TABLE data profile (Inputs)
# memory_palace['y_2012']['features']['train_x']
# memory_palace['y_2012']['features']['val_x']

# PLOT histograms (Inputs)
# memory_palace['y_2012']['features']['train_x']
# memory_palace['y_2012']['features']['val_x']

# PLOT correlation inputs and target
# memory_palace['y_2012']['features']['train_y'], memory_palace['y_2012']['features']['train_x']

# PLOT ROC & AUC
# PLOT ohlc class

# PLOT MultiROC (From Fold )
# PLOT AUC TS 

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
