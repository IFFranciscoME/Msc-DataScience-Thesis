
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: datos.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from os import listdir, path
from os.path import isfile, join


# ------------------------------------------------------------------------------ Agrupamiento de precios -- #
# ------------------------------------------------------------------------------ ----------------------- -- #

def group_daily():
    abspath = path.abspath('archivos/continuos/')
    p_years_list = ['2007', '2008', '2009']
    r_data = {}
    files = sorted([f for f in listdir(abspath) if isfile(join(abspath, f))])
    column_names = ["timestamp", "open", "high", "low", "close", "volume"]

    for file in files:
        data = pd.read_csv('archivos/continuos/' + file,
                           names=column_names, parse_dates=["timestamp"], index_col=["timestamp"])

        data.columns = [i.lower() for i in list(data.columns)]
        data = data.resample("T").agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min',
                                         'volume': 'sum'})
        data = data.dropna()

        years = set([str(datadate.year) for datadate in list(data.index)])
        [years.discard(i) for i in p_years_list]

        years = sorted(list(years))

        for year in years:
            data_temp = data.groupby(pd.Grouper(freq='1Y')).get_group(year + '-12-31')
            # data_temp.to_csv('Archivos/' + 'MP_MIN_' + year + '.csv')
            r_data['MP_D_' + year] = data_temp

    return r_data


# ----------------------------------------------------------------------------------- Lectura de precios -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

abspath_f = path.abspath('archivos/diarios/')
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

for file_f in files_f:
    data_f = pd.read_csv('archivos/diarios/' + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])
    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['MP_D_' + year_f] = data_f

# ----------------------------------------------------------------------------- Temas para visualizacion -- #
# ----------------------------------------------------------------------------- ------------------------ -- #

# grafica OHLC
plot_1 = dict(p_theme={'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB',
                       'font_color_1': '#ABABAB', 'font_size_1': 12, 'font_size_2': 16},
              p_dims={'width': 1450, 'height': 800},
              p_labels={'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'})


# --------------------------------------------------------------------------- Hyperparametros de modelos -- #
# --------------------------------------------------------------------------- -------------------------- -- #

# data dictionary for models and their respective hyperparameter value candidates
models = {'model_1': {'label': 'logistic-elasticnet',
                      'params': {'ratio': [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                                 'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5]}},

          'model_2': {'label': 'ls-svm',
                      'params': {'c': [1.5, 1.1, 1, 0.8, 0.5, 1.5, 1.1, 1, 0.8, 0.5],
                                 'kernel': ['linear', 'linear', 'linear', 'linear', 'linear',
                                            'rbf', 'rbf', 'rbf', 'rbf', 'rbf'],
                                 'gamma': ['scale', 'scale', 'scale', 'scale', 'scale',
                                           'auto', 'auto', 'auto', 'auto', 'auto']}},

          'model_3': {'label': 'ann-mlp',
                      'params': {'hidden_layers': [(10, ), (20, ), (5, 5), (20, 20), (50, ),
                                                   (10, ), (10, ), (5, 5), (10, 10), (20, )],
                                 'activation': ['relu', 'relu', 'relu', 'relu', 'relu',
                                                'logistic', 'logistic', 'logistic', 'logistic', 'logistic'],
                                 'alpha': [0.2, 0.1, 0.01, 0.001, 0.0001, 0.2, 0.1, 0.01, 0.001, 0.0001],
                                 'learning_r': ['constant', 'constant', 'constant', 'constant', 'constant',
                                                'adaptive', 'adaptive', 'adaptive', 'adaptive', 'adaptive'],
                                 'learning_r_init': [0.2, 0.1, 0.01, 0.001, 0.0001,
                                                     0.2, 0.1, 0.01, 0.001, 0.0001]}}}

