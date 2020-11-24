
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
from codigos.datos import plot_1

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
datos = data

# ------------------------------------------------------------------------------- visualizacion de datos -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# cargar funcion importada de visualizaciones
ohlc = vs.g_ohlc(p_ohlc=datos,
                 p_theme=plot_1['p_theme'], p_dims=plot_1['p_dims'], p_labels=plot_1['p_labels'])

# mostrar grafica
ohlc.show()

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
                                                     0.2, 0.1, 0.01, 0.001, 0.0001]}}}

# -------------------------------------------------------------------------------------- M_Folds Results -- #
# -------------------------------------------------------------------------------------- --------------- -- #

# Objeto para almacenar todos los datos
memory_palace = {j: {i: {'pop': [], 'logs': [], 'hof': [], 'e_hof': []} for i in m_folds} for j in models}

# -- Ciclo para evaluar todos los resultados
for model in models:
    for period in m_folds:
        # generacion de features
        m_features = fn.genetic_programed_features(p_data=m_folds[period], p_memory=7)
        # resultados de optimizacion
        hof_model = fn.genetic_algo_optimisation(p_data=m_features, p_model=models[model])
        # -- evaluacion de modelo para cada modelo y cada periodo de todos los Hall of Fame
        for i in range(0, len(list(hof_model['hof']))):
            # evaluar modelo
            hof_eval = fn.evaluaciones_periodo(p_features=m_features,
                                               p_model=model,
                                               p_optim_data=hof_model['hof'])
            # guardar evaluaciones de todos los cromosomas del Hall of Fame
            memory_palace[model][period]['e_hof'].append(hof_eval)

# -- --------------------------------------------------------------------------------------- Data Tables -- #
# -- --------------------------------------------------------------------------------------- ----------- -- #

# -- Obtener 3 cromosomas (individuos) para cada modelo

# -- caso 1
# El cromosoma de todos los HOF de todos los periodos que produjo el menor accuracy

# -- caso 2
# El cromosoma de todos los HOF de todos los periodos que produjo el mayor accuracy

# -- caso 3
# El cromosoma que mas se repitio en todos los HOF de todos los periodos

# Utilizar cada uno de los 3 cromosomas y hacer un backtest de todos los periodos
# Obtener metricas descriptivas del backtest

# -- ----------------------------------------------------------------------- Visualizacion de resultados -- #
# -- ----------------------------------------------------------------------- --------------------------- -- #

# -- TABLA: Caso 1
# Modelo 1 + Matriz de Confusion para Caso 1 (Original) y para Caso 1 (Total)
# Modelo 2 + Matriz de Confusion para Caso 1 (Original) y para Caso 1 (Total)
# Modelo 3 + Matriz de Confusion para Caso 1 (Original) y para Caso 1 (Total)

# -- TABLA: Caso 2
# Modelo 1 + Matriz de Confusion para Caso 2 (Original) y para Caso 2 (Total)
# Modelo 2 + Matriz de Confusion para Caso 2 (Original) y para Caso 2 (Total)
# Modelo 3 + Matriz de Confusion para Caso 2 (Original) y para Caso 2 (Total)

# -- TABLA: Caso 3
# Modelo 1 + Matriz de Confusion para Caso 3 (Original) y para Caso 3 (Total)
# Modelo 2 + Matriz de Confusion para Caso 3 (Original) y para Caso 3 (Total)
# Modelo 3 + Matriz de Confusion para Caso 3 (Original) y para Caso 3 (Total)

# -- GRAFICA ASERTIVIDAD : DATOS + 3 MODELOS
# grafica de barras, de 4 renglones:
# -- 1er renglon barras verticales de clase original en el tiempo
# -- 2do renglon barras verticales de clase pronosticada con modelo 1
# -- 3er renglon barras verticales de clase pronosticada con modelo 2
# -- 4to renglon barras verticales de clase pronosticada con modelo 3

# -- -------------------------------------------------------------------- BackTest de Sistema de trading -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# incorporar valores para criterio 3 y 4
# Criterio 3 (25 y 50)
# Criterio 4 (1% del capital)
