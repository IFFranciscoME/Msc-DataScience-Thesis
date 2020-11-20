
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

# ------------------------------------------------------------------ Seleccion y Optimizacion Simultanea -- #
# ------------------------------------------------------------------ ----------------------------------- -- #
#  m_folds results of feature engineering/selection & Hyperparameter Optimization processes

# data dictionary for models and their respective hyperparameter value candidates
models = {'model_1': {'label': 'ols-elasticnet',
                      'params': {'alpha': [0.1, 0.2, 0.5, 0.6, 0.7],
                                 'ratio': [0.1, 0.2, 0.5, 0.6, 0.7]}},

          'model_2': {'label': 'ls-svm',
                      'params': {'c': [0.1, 0.2, 0.5, 0.6, 0.7],
                                 'b': [0.1, 0.2, 0.5, 0.6, 0.7],
                                 'k': [0.1, 0.2, 0.5, 0.6, 0.7]}},

          'model_3': {'label': 'ann-mlp',
                      'params': {'c': [0.1, 0.2, 0.5, 0.6, 0.7],
                                 'b': [0.1, 0.2, 0.5, 0.6, 0.7],
                                 'k': [0.1, 0.2, 0.5, 0.6, 0.7]}}
          }

# paralelizar esta funcion
m_folds_results = fn.f_FeatureModelOptimizer(p_data=m_folds['periodo_1'],
                                             p_memory=7,
                                             p_model=models['model_3'])

# ----------------------------------------------------------------------------- M_Folds Results Analysis -- #
# ----------------------------------------------------------------------------- ------------------------ -- #

# ------------------------------------------------------------------------ M_Folds Results Visualization -- #
# ----------------------------------------------------------------------------- ------------------------ -- #
