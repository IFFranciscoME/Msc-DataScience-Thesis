
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

# -- complementary
from rich import print
from rich import inspect

# Reproducible results
random.seed(123)

# ---------------------------------------------------------- LOAD PRICES, FOLDS AND PROCESS RESULTS DATA -- #
# ------------------------------------------------------------------------------------------------------ -- #

# Historical prices
historical_prices = dt.ohlc_data

# Fold size
fold_case = 'semester'

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=historical_prices, p_period=fold_case)

# Route to backup files folder
dir_route = 'files/backups/ludwig/test_1_09042021/'

# Final route
file_route = dir_route + 's_logloss-train_standard_20_fix.dat'

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

# metric type for MIN, MAX, MODE
metric_case = 'auc-diff'

# Filtered cases
filters = {'filter_1': {'metric': 'acc-train', 'objective': 'above_threshold', 'threshold': 0.70},
           'filter_2': {'metric': 'acc-val', 'objective': 'above_threshold', 'threshold': 0.70},
           'filter_3': {'metric': 'acc-diff', 'objective': 'abs_min'}}

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

# 'y_logloss-train_robust_20_fix.dat' : ann con .89 en train

# --------------------------------------------------------------------------------- MIN, MAX, MODE CASES -- #
# --------------------------------------------------------------------------------------------------------- #

# models to explore results
model_case = ml_models[1]

# period of the best of HoF: according to model_case and metric_case 
maxcase_period = met_cases[model_case]['met_max']['period']
maxcase_params = met_cases[model_case]['met_max']['params']
maxcase_metric = met_cases[model_case]['met_max'][metric_case]

# period of the worst of HoF: according to model_case and metric_case 
mincase_period = met_cases[model_case]['met_min']['period']
mincase_params = met_cases[model_case]['met_min']['params']
mincase_metric = met_cases[model_case]['met_min'][metric_case]

# Modes and their params, no. of repetitions and periods.
mode_repetitions = pd.DataFrame(met_cases[model_case]['met_mode']['data']).T
