

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
from os import listdir, name, path
from os.path import isfile, join

# -- complementary
from rich import print
from rich import inspect

# Reproducible results

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
from os import listdir, name, path
from os.path import isfile, join

# -- complementary
from rich import print
from rich import inspect

# Reproducible results
random.seed(123)

# ---------------------------------------------------------- LOAD PRICES, FOLDS AND PROCESS RESULTS DATA -- #
# ------------------------------------------------------------------------------------------------------ -- #

# Route to backup files folder
dir_route = 'files/backups/ludwig/test_2_16042021/'

# Available files with experiment data
abspath = path.abspath(dir_route)
experiment_files = sorted([f for f in listdir(abspath) if isfile(join(abspath, f))])

# Experiments to show
# exp = 's_acc-inv-weighted_standard_20_memory.dat'   # 15 in experiment_files
# exp = 'h_logloss-train_standard_20_fix.dat'       # 10 in experiment_files
# Experiment file  
# experiment = np.argmax([file == exp for file in experiment_files])

relevant = []

for experiment in experiment_files:
    
    print('searched file: ', experiment)

    # Fold case
    fold_case = dt.fold_cases[experiment[0]]

    # Final route
    file_route = dir_route + experiment

    # Historical prices
    historical_prices = dt.ohlc_data

    # Timeseries data division in t-folds
    folds = fn.t_folds(p_data=historical_prices, p_period=fold_case)

    # Load data (previously generated results)
    memory_palace = dt.data_pickle(p_data_objects=None, p_data_action='load', p_data_file=file_route)
    memory_palace = memory_palace['memory_palace']

    # List with the names of the models
    ml_models = list(dt.models.keys())

    # ----------------------------------------------------------------- EXPERIMENT 1: OOS GENERALIZATION -- #
    # ----------------------------------------------------------------------------------------------------- #

    # Basic Filtered cases
    filters = {'filter_1': {'metric': 'acc-train', 'objective': 'above_threshold', 'threshold': 0.70},
               'filter_2': {'metric': 'acc-val', 'objective': 'above_threshold', 'threshold': 0.70},
               'filter_3': {'metric': 'acc-diff', 'objective': 'all'}}

    # metric type for MIN, MAX, MODE
    metric_case = 'acc-train'
    model_case = 'logistic-elasticnet'
    # model_case = 'ann-mlp'

    # -- get MIN, MAX, MODE, FILTERED Cases
    met_cases = fn.model_cases(p_models=ml_models, p_global_cases=memory_palace, 
                            p_data_folds=folds, p_cases_type=metric_case, p_filters=filters)
    
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
        df_filtered.index.name=model_case + 'metrics'
    
    else: 
        df_filtered = pd.DataFrame({})

    if not df_filtered.empty: 
        print('Cases where found')
        # Experiment filtered cases
        list_comp = [c for c in df_filtered.columns if (df_filtered[c]['acc-diff'] <= 0.25)  & 
                                                       (df_filtered[c]['acc-train'] >= 0.60) & 
                                                       (df_filtered[c]['acc-val'] >= 0.60) ]

        df_filtered_diff = df_filtered[list_comp]
        relevant.append(df_filtered_diff)

logistic = relevant.copy()
# df_filtered['s_01_2012_9']
# df_filtered['s_01_2010_2']

# -- MLP: s_01_2012_1 | acc_train 0.915556 | acc_val 0.824561
# met_cases['ann-mlp']['met_filter']['data']['s_01_2012_1']['metrics']

# -- LOG: s_01_2012_9 | acc_train 0.831111 | acc_val 0.736842
# met_cases['logistic-elasticnet']['met_filter']['data']['s_01_2012']['metrics']['s_01_2012_9']

# -- -------------------------------------------------------------------- PLOT TIME SERIES BLOCK T-FOLDS -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# Dates for vertical lines in the T-Folds plot
# dates_folds = []
# for n_fold in list(folds.keys()):
#    dates_folds.append(folds[n_fold]['timestamp'].iloc[0])
#    dates_folds.append(folds[n_fold]['timestamp'].iloc[-1])

# Plot_1 with t-folds vertical lines
# dt.theme_plot_1['p_labels']['title'] = ''
# plot_1 = vs.plot_ohlc(p_ohlc=historical_prices, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)


# -- -------------------------------------------------------------------- PLOT TIME SERIES BLOCK T-FOLDS -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# Dates for vertical lines in the T-Folds plot
# dates_folds = []
# for n_fold in list(folds.keys()):
#    dates_folds.append(folds[n_fold]['timestamp'].iloc[0])met_cases[model_case][case]['period']
plot_1 = vs.plot_ohlc(p_ohlc=historical_prices, p_theme=dt.theme_plot_1, p_vlines=None)

# Show plot in script
# plot_1.show()

# Generate plot online with chartstudio
# py.plot(plot_1)

# -- ------------------------------------------------------------------------------------- DATA PROFILES -- #
# -- ------------------------------------------------------------------------------------- ------------- -- #

# Fold to make a description
des_fold = df_filtered.columns[0][:-2]

# print fold that is beign used for data visualization
print(des_fold)

# TABLE data profile (Target)
exp_train_y = memory_palace[des_fold]['features']['train_y']
exp_val_y = memory_palace[des_fold]['features']['val_y']

tabla_1 = fn.data_profile(p_data=exp_train_y, p_type='target', p_mult=10000)
tabla_2 = fn.data_profile(p_data=exp_val_y, p_type='target', p_mult=10000)

# TABLE data profile (Inputs)
exp_train_x = memory_palace[des_fold]['features']['train_x']
u_exp_train_x = exp_train_x.T.drop_duplicates().T

exp_val_x = memory_palace[des_fold]['features']['val_x']
u_exp_val_x = exp_train_x.T.drop_duplicates().T

tabla_3 = fn.data_profile(p_data=u_exp_train_x, p_type='target', p_mult=10000)
tabla_4 = fn.data_profile(p_data=u_exp_val_x, p_type='target', p_mult=10000)

# -- ----------------------------------------------------------------------- PLOT: MULTI PLOT HISTOGRAMS -- #
# -- ---------------------------------------------------------------------------- ---------------------- -- #

# PLOT histogram (Features)
plot_2_1 = vs.plot_h_histograms(p_data=u_exp_train_x.iloc[:, 149:159], p_theme=dt.theme_plot_2)

# Show plot
# plot_2_1.show()

# PLOT histogram (Features)
# plot_2_2 = vs.plot_h_histograms(p_data=exp_train_x.iloc[:, -10:])

# Show plot
# plot_2_2.show()# data subset to use
subset = 'val'

# metric to use
period_case = 's_01_2012'

# parameters of the evaluated models
d_params = memory_palace[period_case][model_case]['p_hof']['hof']

# get all fps and tps for a particula
# -- Target and Auto regressive Features correlation
exp_1 = pd.concat([u_exp_train_x.iloc[:, 0:52].copy()], axis=1)

exp_1_corr_p = exp_1.corr('pearson')
exp_1_p_plot = vs.plot_heatmap_corr(p_data=exp_1_corr_p.copy(), p_colors='Blues')
exp_1_corr_s = exp_1.corr('spearman')
exp_1_s_plot = vs.plot_heatmap_corr(p_data=exp_1_corr_s.copy(), p_colors='Greens')

# Show plotsmet_cases[model_case][case]['period']
# exp_1_p_plot.show()
# exp_1_s_plot.show()

# -- Among Symbolic Features
exp_2 = pd.concat([u_exp_train_x.iloc[:, -25:].copy()], axis=1)

exp_2_corr_p = exp_2.corr('pearson')
exp_2_p_plot = vs.plot_heatmap_corr(p_data=exp_2_corr_p.copy(), p_colors='Blues')

exp_2_corr_s = exp_2.corr('spearman')
exp_2_s_plot = vs.plot_heatmap_corr(p_data=exp_2_corr_s.copy(), p_colors='Greens')

# Show plots
# exp_2_p_plot.show()
# exp_2_s_plot.show()

# -- Target and Symbolic Features correlation
exp_2 = pd.concat([exp_train_y.copy(), exp_train_x.iloc[:, -40:].copy()], axis=1)
exp_2_corr_p = exp_1.corr('pearson')
title_txt = 'Symbolic Features Vs Target Correlation (pearson)'
exp_2_plot = vs.plot_heatmap_corr(p_data=exp_1_corr_p.copy(), p_title=title_txt, p_scale=True)

# Show plot
# exp_2_plot.show()

# -- ---------------------------------------------------------------------------- PLOT: All ROCs in FOLD -- #
# -- ---------------------------------------------------------------------------- ---------------------- -- #

# case to plot
case = 'met_filter'

# data subset to use
subset = 'train'

# metric to use
period_case = 's_01_2012'

# model to analize
model_case = 'ann-mlp'

# parameters of the evaluated models
d_params = memory_palace[period_case][model_case]['p_hof']['hof']

# get all fps and tps for a particular model in a particular fold
d_plot_4 = {i: {'tpr': memory_palace[period_case][model_case]['e_hof'][i]['metrics'][subset]['tpr'],
                'fpr': memory_palace[period_case][model_case]['e_hof'][i]['metrics'][subset]['fpr'],
                metric_case: memory_palace[period_case][model_case]['e_hof'][i]['pro-metrics'][metric_case]}
            for i in range(0, len(d_params))}

# Plot title
dt.theme_plot_4['p_labels']['title'] = ''

# Timeseries of the AUCs
plot_4 = vs.plot_multiroc(p_data=d_plot_4, p_metric=metric_case, p_theme=dt.theme_plot_4)

# Show plot in script
# plot_4.show()

# Generate plot online with chartstudio
# py.plot(plot_4)

# -- ----------------------------------------------------------------- PLOT: CLASSIFICATION FOLD RESULTS -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #
 
# Pick case
case = 'met_filter'

# Pick model to generate the plot
model_case = 'ann-mlp'

# Plot title
dt.theme_plot_4['p_labels']['title'] = ''

# Prices (max or min )
# ohlc_prices = folds[met_cases[model_case][case]['period']]

# Prices for filtered case
ohlc_prices = folds['s_01_2012']

# Get data from met_cases (max or min)
# data_y = met_cases[model_case][case]['data']['results']['data']['train']
# data_y = met_cases[model_case][case]['data']['results']['data']['val']
# ohlc_class = {'train_y': data_y['train_y'], 'train_y_pred': data_y['train_pred_y']}
# date_vlines = [ohlc_class['train_y'].index[-1]]

# Get data from met_cases (filtered) in inverse order
# data_y = met_cases[model_case][case]['data']['s_01_2012']['data'][9][0]['results']['data']['train']
# ohlc_class = {'train_y': data_y['train_y'], 'train_y_pred': data_y['train_pred_y']}
# date_vlines = [ohlc_class['train_y'].index[-1]]

data_y = met_cases[model_case][case]['data']['s_01_2012']['data'][9][0]['results']['data']['val']
ohlc_class = {'val_y': data_y['val_y'], 'val_y_pred': data_y['val_pred_y']}
date_vlines = [ohlc_class['val_y'].index[-1]]

# Make plot
plot_3 = vs.plot_ohlc_class(p_ohlc=ohlc_prices, p_theme=dt.theme_plot_4, p_data_class=ohlc_class, 
                            p_vlines=date_vlines)

plot_3 = plot_3.update_layout(title=None)

# Show plot in script
# plot_3.show()

# ---------------------------------------------------------------------------------- FILTERED PARAMETERS -- #
# --------------------------------------------------------------------------------------------------------- #

# MLP
# -- MLP: s_01_2012_1 | acc_train 0.915556 | acc_val 0.824561
# model_case = 'ann-mlp'
# data_params = met_cases[model_case][case]['data']['s_01_2012']['data'][9][0]

# import json
# model_params = json.loads(data_params['model'])['config']['layers']
# pd.DataFrame(model_params['layers'][5])

# LOGISTIC
# -- LOG: s_01_2012_9 | acc_train 0.831111 | acc_val 0.736842
# model_case = 'logistic-elasticnet'
# data_params = met_cases[model_case][case]['data']['s_01_2012']['data'][1][0]
# modelito = data_params['model']
# inspect(modelito)

# --------------------------------------------------------------------------------- MIN, MAX, MODE CASES -- #
# --------------------------------------------------------------------------------------------------------- #

# Make plot# data subset to use
subset = 'train'

# metric to use
period_case = 's_01_2012'

# parameters of the evaluated models
d_params = memory_palace[period_case][model_case]['p_hof']['hof']

# get all fps and tps for a particula
model_case = 'logistic-elasticnet'

# period of the best of HoF: according to model_case and metric_case 
maxcase_period = met_cases[model_case]['met_max']['period']
maxcase_params = met_cases[model_case]['met_max']['params']
maxcase_metric = met_cases[model_case]['met_max'][metric_case]

# output DataFrame
df_max = pd.DataFrame([met_cases[model_case]['met_max']['data']['pro-metrics']]).T
df_max.columns = [maxcase_period]
df_max.index.name=model_case + 'metrics'
# df_max.head(5)

# period of the worst of HoF: according to model_case and metric_case 
mincase_period = met_cases[model_case]['met_min']['period']
mincase_params = met_cases[model_case]['met_min']['params']
mincase_metric = met_cases[model_case]['met_min'][metric_case]

# output DataFrame
df_min = pd.DataFrame([met_cases[model_case]['met_min']['data']['pro-metrics']]).T
df_min.columns = [mincase_period]# data subset to use
subset = 'train'

# metric to use
period_case = 's_01_2012'

# parameters of the evaluated models
d_params = memory_palace[period_case][model_case]['p_hof']['hof']

# get all fps and tps for a particula
df_min.index.name=model_case + '-metrics'
# df_min.head(5)

# Modes and their params, no. of repetitions and periods.
mode_repetitions = pd.DataFrame(met_cases[model_case]['met_mode']['data']).T
# mode_repetitions.head(5)


# --------------------------------------------------------------------------- SIMILARITY IN DISTRIBUTION -- #
# --------------------------------------------------------------------------------------------------------- #

# Use target variable (discrete for classification problem)
# Produce a Kullback Liebler based distance matrix among all fold

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# p = (folds['q_02_2010']['close'] - folds['q_02_2010']['open'])*10000
# p = np.array(p/max(p))
# p = p[0:190]

# q = (folds['q_04_2009']['close'] - folds['q_04_2009']['open'])*10000 
# q = np.array((q - np.mean(q))/np.std(q))
# q = np.array(q/max(q))
# q = q[0:190]

# sns.jointplot(x=p, y=q, kind='scatter')
# plt.show()

# fig = sns.kdeplot(p, shade=True, color="r")
# fig = sns.kdeplot(q, shade=True, color="b")
# plt.show()

# Example 1: Very different PDFs of two sub-samples
# q_label = 'q_03_2012'
# p_label = 'q_04_2012'

# Example 2: Very similar PDFs of two sub-samples
# q_label = 'q_01_2009'
# p_label = 'q_03_2009'

experiment = 's_acc-inv-weighted_standard_20_memory.dat'   # 15 in experiment_files
# exp = 'h_logloss-train_standard_20_fix.dat'       # 10 in experiment_files

# Fold case
fold_case = dt.fold_cases[experiment[0]]

# Final route
file_route = dir_route + experiment

# Historical prices
historical_prices = dt.ohlc_data

# Timeseries data division in t-folds
folds = fn.t_folds(p_data=historical_prices, p_period=fold_case)

# Load data (previously generated results)
memory_palace = dt.data_pickle(p_data_objects=None, p_data_action='load', p_data_file=file_route)
memory_palace = memory_palace['memory_palace']

# TOG Results
q_label = 's_01_2011'
p_label = 's_01_2012'

# use the variable from wich the sign is extracted, for continuous KLD, since for the
# discrete case the alternative would be to use only 2 types of values because there are
# just two classes.
q = (folds[q_label]['close'] - folds[q_label]['open'])
p = (folds[p_label]['close'] - folds[p_label]['open'])
# shift to only positive numbers by adding the abs of the most negative value 
q = (q + abs(min(q)))/max(q)
p = (p + abs(min(p)))/max(p)

# divergence metric (just two samples)
divergence = fn.kldivergence(p, q)
# divergence

# plot for visual validation
import plotly.figure_factory as ff
import plotly.graph_objects as go

labels = [q_label, p_label]
data = [q, p]
all_dists = ff.create_distplot(hist_data=data, group_labels=labels, bin_size=.05,
                               histnorm='probability', show_curve=False)

# all_dists.update_layout(title='Kull-Back Divergence Metric: ' + str(divergence))

all_dists.layout.width = 700
all_dists.layout.height = 450

all_dists.update_layout(legend=go.layout.Legend(x=0.35, y=-.1, orientation='h',
                                                bordercolor='dark grey',
                                                borderwidth=1))

all_dists.update_layout(yaxis=dict(title='probability'))

# Show plot
all_dists.show()

# PLOT histogram (target variable distribution)
# plot_5 = vs.plot_h_histograms(p_data=exp_train_x.iloc[:, -10:])

# Show plot
# plot_5.show()
