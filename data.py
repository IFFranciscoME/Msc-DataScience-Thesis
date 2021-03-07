
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: data.py - input and output data functions for the project                                     -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pickle
import itertools
import pandas as pd
from os import listdir, path, environ
from os.path import isfile, join

# To detect where the project is executed
from multiprocessing import cpu_count

# Redirect to the respective folder to store files
if cpu_count() > 8:
    print('\nOoooh Yeahhh')
    folder = 'ludwig/'
else:
    print('\nmeh')
    folder = 't490/'

# Modify environmental variable to suppress console log messages from TensorFlow
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------------------------------------------------------------- EXECUTION PARAMETERS -- #
# --------------------------------------------------------------------------------- -------------------- -- #

# number of processing cores to use
exec_workers = cpu_count()-4

# --------------------------------------------------------------------------------------- SUBSET VERSION -- #

# number of years of prices, accepted values range from 2 (2009) to 12 (2020)
exec_data = 4

# models to explore
exec_models = ['ann-mlp', 'logistic-elasticnet', 'l1-svm']

# fold size
exec_fold = ['quarter', 'semester', 'year']

# experiment parameters
exec_opt = {'embargo': ['fix', 'memory'],
            'inner-split': ['20', '0'],
            'trans_function': ['scale', 'robust', 'standard'],
            'trans_order': ['post-features'],
            'fitness': ['logloss-inv-weighted', 'logloss-train']}

# ------------------------------------------------------------------------------------- COMPLETE VERSION -- #

"""
# number of years of prices
exec_data = 12

# models to explore
exec_models = ['logistic-elasticnet', 'l1-svm', 'ann-mlp']

# fold size
exec_fold = ['quarter', 'semester', 'year', 'bi-year', '80-20']

# experiment parameters
exec_opt = {'embargo': ['fix', 'memory', 'False'],
            'inner-split': ['30', '10', '0'],
            'transform': ['scale', 'normalize', 'robust'],
            'scaling': ['post-features', 'pre-features'],
            'fitness': ['auc-train', 'auc-val', 'auc-diff',
                        'auc-mean', 'auc-weighted', 'auc-inv-weighted',
                        'acc-train', 'acc-val', 'acc-diff',
                        'acc-mean', 'acc-weighted', 'acc-inv-weighted',
                        'logloss-train', 'logloss-val', 'logloss-diff',
                        'logloss-mean', 'logloss-weighted', 'logloss-inv-weighted']}
"""

# Iterative/Parallel Experiment Data
exec_exp = list(itertools.product(*[exec_opt['embargo'], exec_opt['inner-split'], exec_opt['trans_function'],
                                    exec_opt['trans_order'], exec_opt['fitness']]))

# --------------------------------------------------------------------- Parameters for Symbolic Features -- #
# --------------------------------------------------------------------- -------------------------------- -- #

# parameters for features formation
features_params = {'lags_diffs': 15}

# paremeters for symbolic features generation process
symbolic_params = {'functions': ['sub', 'add', 'inv', 'mul', 'div', 'abs', 'log', 'sqrt'],
                   'population': 2000, 'tournament': 20, 'hof': 15, 'generations': 9, 'n_features': 15,
                   'init_depth': (6, 18), 'init_method': 'half and half', 'parsimony': 0.05,
                   'constants': None,
                   'metric': 'pearson', 'metric_goal': 0.70, 
                   'prob_cross': 0.4, 'prob_mutation_subtree': 0.4,
                   'prob_mutation_hoist': 0.1, 'prob_mutation_point': 0.1,
                   'verbose': True, 'parallelization': True, 'warm_start': True}

# ------------------------------------------------------------------ Parameters for Genetic Optimization -- #
# ------------------------------------------------------------------ ----------------------------------- -- #

optimization_params = {'halloffame': 10, 'tournament': 15, 'population': 60, 'generations': 9,
                       'mutation': 0.7, 'crossover': 0.7}

# ----------------------------------------------------------------------- Hyperparameters for the Models -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

models = {

    'logistic-elasticnet': {
        'label': 'logistic-elasticnet',

        'params': {'ratio': [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
                             0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],

                   'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5,
                         1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5] }},

    'l1-svm': {
        'label': 'l1-svm',

        'params': {'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5,
                         1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5],

                   'kernel': ['poly', 'linear', 'linear', 'linear', 'poly',
                              'poly', 'linear', 'linear', 'linear', 'poly',

                              'poly', 'rbf', 'rbf', 'rbf', 'poly',
                              'poly', 'rbf', 'rbf', 'rbf', 'poly'],

                   'gamma': ['scale', 'scale', 'scale', 'scale', 'scale',
                             'auto', 'auto', 'auto', 'auto', 'auto',

                             'scale', 'scale', 'scale', 'scale', 'scale',
                             'auto', 'auto', 'auto', 'auto', 'auto'],
                             
                   'degree': [2, 2, 2, 2, 2, 4, 4, 4, 4, 4,
                              6, 6, 6, 6, 6, 9, 9, 9, 9, 9] }},

    'ann-mlp': {
        'label': 'ann-mlp',
        'params': {'hidden_layers': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    
                   'hidden_neurons': [20, 35, 40, 55, 60, 65, 75, 80, 100, 105,
                                      20, 35, 40, 55, 60, 65, 75, 80, 100, 105],

                   'activation': ['relu', 'relu', 'relu', 'relu', 'relu',
                                  'relu', 'relu', 'relu', 'relu', 'relu', 
                                  
                                  'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                                  'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],

                   'reg_1': [[0.001, 0.001], [0.01, 0.01], [0.001, 0.01], [0.01, 0.001], [0, 0],
                             [0.001, 0.001], [0.01, 0.01], [0.001, 0.01], [0.01, 0.001], [0, 0]],
                    
                   'reg_2': [[0, 0], [0.005, 0.005], [0.015, 0.005], [0.005, 0.015], [0.030, 0.030],
                             [0, 0], [0.005, 0.005], [0.015, 0.005], [0.005, 0.015], [0.030, 0.030]],
                    
                    'dropout': [0.001, 0.005, 0.01, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 
                                0.001, 0.005, 0.01, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25],

                   'learning_rate': [.05, 0.075, 0.055, 0.030, 0.015, 0.01, 0.05, 0.02, 0.01, 0.001,
                                     .05, 0.075, 0.055, 0.030, 0.015, 0.01, 0.05, 0.02, 0.01, 0.001],

                   'momentum': [0.4, 0.2, 0.1, 0.07, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                                0.4, 0.2, 0.1, 0.07, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]}}}

# -------------------------------------------------------------------- Historical Minute Prices Grouping -- #
# -------------------------------------------------------------------- --------------------------------- -- #

def group_daily():
    
    # resample by the minute "T", and by the day will be "D", and by 8 hours will be "8H"
    file_nom = 'M1'
    
    main_path_g = 'files/prices/raw/'
    abspath = path.abspath(main_path_g)
    p_years_list = ['2007', '2008']
    r_data = {}
    files = sorted([f for f in listdir(abspath) if isfile(join(abspath, f))])
    # swap high with low since the original data is wrong
    column_names = ["timestamp", "open", "high", "low", "close", "volume"]

    for file in files:
        # file = files[0]
        data = pd.read_csv(main_path_g + file,
                           names=column_names, parse_dates=["timestamp"], index_col=["timestamp"])

        data = data.resample("T").agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                          'volume': 'sum'})
        # eliminar los NAs originados por que ese minuto
        data = data.dropna()

        years = set([str(datadate.year) for datadate in list(data.index)])
        [years.discard(i) for i in p_years_list]
        years = sorted(list(years))

        for year in years:
            data_temp = data.groupby(pd.Grouper(freq='1Y')).get_group(year + '-12-31')
            data_temp.to_csv('files/prices/' + file_nom + '/MP_' + file_nom + '_' + year + '.csv')
            r_data['MP_' + file_nom + '_' + year] = data_temp

    return r_data


# ---------------------------------------------------------------------------- Historical Prices Reading -- #
# ---------------------------------------------------------------------------- ------------------------- -- #

# the price in the file is expressed as the USD to purchase one MXN
# if is needed to convert to the inverse, the MXN to purchase one USD, uncomment the following line
mode = 'MXN_USD'
file_nom = 'H8'

# path in order to read files
main_path = 'files/prices/' + file_nom + '/'
abspath_f = path.abspath(main_path)
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

# iterative data reading
for file_f in files_f:
    # file_f = files_f[3]
    data_f = pd.read_csv(main_path + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])

    # swap since original was inversed and there is a mirror effect like this: min = 1/max
    low = data_f['low'].copy()
    high = data_f['high'].copy()
    data_f['high'] = low
    data_f['low'] = high

    # if mode == 'MXN_USD':
    data_f['open'] = round(1/data_f['open'], 5)
    data_f['high'] = round(1/data_f['high'], 5)
    data_f['low'] = round(1/data_f['low'], 5)
    data_f['close'] = round(1/data_f['close'], 5)

    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data[file_nom + year_f] = data_f

# Number of periods according to execution parameter: exec_data
ohlc_data = pd.concat([price_data[list(price_data.keys())[i]] for i in range(0, exec_data)])

# Test data
# test_data = pd.concat([price_data[list(price_data.keys())[12]]])

# ------------------------------------------------------------------------ SAVE/LOAD DATA: PICKLE FORMAT -- #
# ----------------------------------------------------------------------------- ------------------------ -- #

def data_pickle(p_data_objects, p_data_action, p_data_file):
    """
    Save or load data in pickle format for offline use
    Parameters
    ----------
    p_data_objects: dict
        with data objects to be saved
    p_data_action: str
        'save' to data saving or 'load' to data loading
    p_data_file: str
        with the name of the pickle file
    Returns
    -------
    Message if data file is saved or data objects if data file is loaded
    """

    # if saving is required`
    if p_data_action == 'save':

        # define and create file
        pick = p_data_file
        with open(pick, "wb") as f:
            pickle.dump(p_data_objects, f)

        # Return message
        return 'Data saved in' + p_data_file + ' file'

    # if loading is required
    elif p_data_action == 'load':

        # read the file
        with open(p_data_file, 'rb') as handle:
            loaded_data = pickle.load(handle)

        # return loaded data
        return loaded_data

# ------------------------------------------------------------------------------------- Themes for plots -- #
# ------------------------------------------------------------------------------------- ---------------- -- #

# Plot_1 : Original Historical OHLC prices
theme_plot_1 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 400},
                    p_labels={'title': 'Precios OHLC',
                              'x_title': 'Dates', 'y_title': 'Continuous Future Prices USD/MXN'})

# Plot_2 : Timeseries T-Folds blocks without filtration
theme_plot_2 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 500},
                    p_labels={'title': 'T-Folds por Bloques Sin Filtraciones',
                              'x_title': 'Fechas', 'y_title': 'Continuous Future Prices USD/MXN'})

# Plot_3 Observed Class vs Predicted Class
theme_plot_3 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 500},
                    p_labels={'title': 'Clasifications',
                              'x_title': 'Dates', 'y_title': 'Continuous Future Price USD/MXN'})

# Plot_4 ROC of models
theme_plot_4 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 500},
                    p_labels={'title': 'ROC',
                              'x_title': 'FPR', 'y_title': 'TPR'})

# Plot_5 AUC Timeseries of models
theme_plot_5 = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                    p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                    p_dims={'width': 900, 'height': 500},
                    p_labels={'title': 'AUC por periodo (val Data)',
                              'x_title': 'Periodos', 'y_title': 'AUC'})
