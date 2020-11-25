
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
from codigos.datos import plot_1, models
import pandas as pd

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]
# data_complete = pd.concat([price_data[list(price_data.keys())[8]], price_data[list(price_data.keys())[9]]])

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
# datos = data.iloc[0:129]
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
# datos.describe()

# ------------------------------------------------------------------------ Division en K-Folds mensuales -- #
# ------------------------------------------------------------------------ ----------------------------- -- #

# -- Division de periodos de datos, sin filtracion, en amplitudes de 1 mes para obtener 12 "Folds".
m_folds = fn.f_m_folds(p_data=data, p_periodo='trimestre')

# -------------------------------------------------------------------------------------- M_Folds Results -- #
# -------------------------------------------------------------------------------------- --------------- -- #

# Objeto para almacenar todos los datos
memory_palace = {j: {i: {'pop': [], 'logs': [], 'hof': [], 'e_hof': []} for i in m_folds} for j in models}

# -- Ciclo para evaluar todos los resultados
for model in models:
    for period in m_folds:
        print('modelo: ', model)
        print('periodo: ', period)
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
            # guardar evaluaciones de todos los individuos del Hall of Fame
            memory_palace[model][period]['e_hof'].append(hof_eval)


# -- --------------------------------------------------------------------------- 3 Casos Representativos -- #
# -- --------------------------------------------------------------------------- ----------------------- -- #
# -- Funcion de casos representativos

# diccionario para almacenar resultados de busqueda
casos = {j: {i: {'data': {}} for i in ['auc_min', 'auc_max', 'mode']} for j in models}

# ciclo para busqueda de auc_min y auc_max
for model in models:
    auc_min = 1
    auc_max = 0
    for period in m_folds:
        for i in range(0, 10):
            # -- caso 1
            # El individuo de todos los HOF de todos los periodos que produjo la minima AUC
            if memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] < auc_min:
                auc_min = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                casos[model]['auc_min']['data'] = memory_palace[model][period]['e_hof'][i]
            # -- caso 2
            # El individuo de todos los HOF de todos los periodos que produjo la maxima AUC
            elif memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] > auc_max:
                auc_max = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                casos[model]['auc_max']['data'] = memory_palace[model][period]['e_hof'][i]


# -- ------------------------------------------------------------------------- Backtest global de modelo -- #
# -- ------------------------------------------------------------------------- ------------------------- -- #

# Evaluation of auc_min and auc_max cases in all the models
global_models = {model: {'auc_min': {}, 'auc_max': {}} for model in models}
cases = ['auc_min', 'auc_max']

# Global features (whole dataset)
global_features = fn.genetic_programed_features(p_data=data, p_memory=7)

# Evaluate all global cases
for model in models:
    for case in cases:
        if model == 'model_1':
            global_models[model][case] = fn.logistic_net(p_data=global_features,
                                                         p_params=casos[model][case]['data']['params'])
        elif model == 'model_2':
            global_models[model][case] = fn.ls_svm(p_data=global_features,
                                                   p_params=casos[model][case]['data']['params'])
        elif model == 'model_3':
            global_models[model][case] = fn.ann_mlp(p_data=global_features,
                                                    p_params=casos[model][case]['data']['params'])

# -- ----------------------------------------------------------------------- Visualizacion de resultados -- #
# -- ----------------------------------------------------------------------- --------------------------- -- #

# -- PRIORIDAD 1
# ------------------------------------------------------------ Hacer una GRAFICA para todos los periodos -- #
# -- TITULO: Separacion de datos
# precios OHLC de todos los periodos con lineas verticales de separacion de periodos de m_folds

# construccion de fechas para lineas verticales de division de cada fold
fechas_folds = []
for fold in m_folds:
    fechas_folds.append(m_folds[fold]['timestamp'].iloc[0])
    fechas_folds.append(m_folds[fold]['timestamp'].iloc[-1])

# grafica OHLC
ohlc = vs.g_ohlc(p_ohlc=datos,
                 p_theme=plot_1['p_theme'], p_dims=plot_1['p_dims'], p_labels=plot_1['p_labels'],
                 p_vlines=fechas_folds)

# mostrar grafica
ohlc.show()


# -- PRIORIDAD 2
# ------------------------------------------------------------ Hacer una GRAFICA para todos los periodos -- #
# -- TITULO: Clases Observadas Vs Clases Ajustadas
# Para todos los datos, grafica de barras con Clases Verdaderas y Clases ajustadas.
# Renglon 1: Clases originales
# Renglon 2: Clases ajustadas con el 1er modelo utilizando los parametros del individuo max AUC global
# Renglon 3: Clases ajustadas con el 2do modelo utilizando los parametros del individuo max AUC global
# Renglon 4: Clases ajustadas con el 3er modelo utilizando los parametros del individuo max AUC global

# -- PRIORIDAD 3
# ------------------------------------ Hacer una GRAFICA con AUC_prom por modelo para todos los periodos -- #
# -- TITULO: Comparativa de Estabilidad de AUC entre modelos
# con los HoF de cada periodo, calcular promedio AUC_min y AUC_max. Mostrar en una linea los valores
# para todos los periodos, se tendran 3 lineas, 1 por modelo
# ejex: Periodo
# ejey: valor de AUC promedio por modelo (3 modelos)

# -- PRIORIDAD 4
# --------------------------------------------------------------------- Hacer una TABLA para cada modelo -- #
# -- TITULO: Estabilidad de hyperparametros de modelo
# Para cada periodo, de los HoF, mostrar los parametros del individuo con mayor max AUC.
# columnas: 01_2018 | 02_2018 | ... | nn_yyyy |
# renglones: max AUC  + parametros del modelo

# -- PRIORIDAD 5
# ------------------------------------------------------- hacer una GRAFICA por periodo para cada modelo -- #
# --- TITULO: Desempeño de poblacion de configuraciones para + 'Nombre de modelo' en periodo + 'periodo'
# Mostrar en gris las ROC de los 10 individuos en el HoF, mostrar la ROC con max AUC y min AUC con color
# diferente
# ejex: FPR
# ejey: TPR
# leyenda: max AUC, min AUC

# -- PRIORIDAD 5
# -------------------------------------------------------------- hacer una TABLA para todos los periodos -- #
# -- TITULO: DESEMPEÑO GENERAL DE MODELOS
# Para todos los periodos, para los 3 modelos, para individuos con Max AUC y Min AUC por periodo, mostrar
# todos los parametros de cada caso.
# columnas: modelo_1_min | modelo_1_max | modelo_2_min | modelo_2_max | modelo_3_min | Modelo_3_max |
# renglones: matriz de confusion vertical + acc + auc

# -- -------------------------------------------------------------------- BackTest de Sistema de trading -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

# incorporar valores para criterio 3 y 4
# Criterio 3 (25 y 50)
# Criterio 4 (1% del capital)
