
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import codigos.funciones as fn
from codigos.datos import price_data
import codigos.visualizaciones as vs

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
datos = data

# ------------------------------------------------------------------------------- visualizacion de datos -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# grafica OHLC
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [datos['timestamp'].head(1), datos['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc = vs.g_ohlc(p_ohlc=datos, p_theme=p_theme, p_dims=p_dims, p_vlines=p_vlines, p_labels=p_labels)

# mostrar grafica
# ohlc.show()

# ----------------------------------------------------------------------- analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos
datos.describe()

# ------------------------------------------------------------------------ Division en K-Folds mensuales -- #
# ------------------------------------------------------------------------ ----------------------------- -- #

# -- Division de periodos de datos, sin filtracion, en amplitudes de 1 mes para obtener 12 "Folds".
m_folds = fn.f_m_folds(p_data=datos, p_periodo='mes')

# --------------------------------------------------------------------------- Hyperparametros de modelos -- #
# --------------------------------------------------------------------------- -------------------------- -- #

# data dictionary for models and their respective hyperparameter value candidates
models = {'model_1': {'label': 'ols-elasticnet',
                      'params': {'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.8, 1.10, 1.50, 2.50],
                                 'ratio': [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]}},

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
                                                     0.2, 0.1, 0.01, 0.001, 0.0001]}}
          }

# -------------------------------------------------------------------------------------- M_Folds Results -- #
# -------------------------------------------------------------------------------------- --------------- -- #

# -- todos los resultados
memory_palace = {j: {i: {'pop': [], 'logs': [], 'hof': [], 'e_hof': []} for i in m_folds} for j in models}

for model in models:
    for period in m_folds:
        # period = 'periodo_2'
        print('Corriendo el modelo: ', models[model]['label'])
        print('Corriendo el periodo: ', period)

        # -- generacion de features
        m_features = fn.genetic_programed_features(p_data=m_folds[period], p_memory=7)

        # -- resultados de optimizacion
        hof_model = fn.genetic_algo_optimisation(p_data=m_features, p_model=models[model])

        # -- evaluacion de modelo para cada modelo y cada periodo de todos los Hall of Fame
        for i in range(0, len(list(hof_model['hof']))):
            hof_eval = fn.evaluaciones_periodo(p_features=m_features,
                                               p_model=model,
                                               p_optim_data=hof_model['hof'])
            memory_palace[model][period]['e_hof'].append(hof_eval)

# -- --------------------------------------------------------------------------------------- Data Tables -- #
# -- --------------------------------------------------------------------------------------- ----------- -- #
