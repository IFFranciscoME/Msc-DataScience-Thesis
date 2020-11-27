
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
from codigos.datos import models, theme_plot_1, theme_plot_2, theme_plot_3, theme_plot_4
import pandas as pd

# Datos con un solo periodo
datos = price_data[list(price_data.keys())[9]]

# Datos con todos los periodos
# data_complete = pd.concat([price_data[list(price_data.keys())[0]], price_data[list(price_data.keys())[1]],
#                            price_data[list(price_data.keys())[2]], price_data[list(price_data.keys())[3]],
#                            price_data[list(price_data.keys())[4]], price_data[list(price_data.keys())[5]],
#                            price_data[list(price_data.keys())[6]], price_data[list(price_data.keys())[7]],
#                            price_data[list(price_data.keys())[8]], price_data[list(price_data.keys())[9]]])

# --------------------------------------------------------------- PLOT 1: PRECIOS OHLC DE FUTURO USD/MXN -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# cargar funcion importada de visualizaciones
plot_1 = vs.g_ohlc(p_ohlc=datos, p_theme=theme_plot_1['p_theme'], p_dims=theme_plot_1['p_dims'],
                   p_labels=theme_plot_1['p_labels'], p_vlines=None)

# mostrar grafica
# plot_1.show()

# ----------------------------------------------------------------------- Analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos
table_1 = datos.describe()

# --------------------------------------------------------------------- Division en K-Folds por periodos -- #
# --------------------------------------------------------------------- -------------------------------- -- #

# Division de periodos de datos, sin filtracion
# en amplitudes de 'trimestre' para obtener 4 folds por cada año
m_folds = fn.f_m_folds(p_data=datos, p_periodo='trimestre')

# -------------------------------------------------------- PROCESO: Feratures - Train/Optimizatio - Test -- #
# -------------------------------------------------------- --------------------------------------------- -- #

# Objeto para almacenar todos los datos
memory_palace = {j: {i: {'pop': [], 'logs': [], 'hof': [], 'e_hof': []} for i in m_folds} for j in models}

# -- Ciclo para evaluar todos los resultados
for model in models:
    for period in m_folds:

        print('\n')
        print('--------------------')
        print('modelo: ', model)
        print('periodo: ', period)
        print('--------------------')
        print('\n')
        print('----------------------- Ingenieria de Variables por Periodo ------------------------')
        print('----------------------- ----------------------------------- ------------------------')

        # generacion de features
        m_features = fn.genetic_programed_features(p_data=m_folds[period], p_memory=7)

        # resultados de optimizacion
        print('\n')
        print('--------------------- Optimizacion de hiperparametros por Periodo ------------------')
        print('--------------------- ------------------------------------------- ------------------')

        hof_model = fn.genetic_algo_optimisation(p_data=m_features, p_model=models[model])
        # -- evaluacion de modelo para cada modelo y cada periodo de todos los Hall of Fame
        for i in range(0, len(list(hof_model['hof']))):
            # evaluar modelo
            hof_eval = fn.evaluaciones_periodo(p_features=m_features, p_model=model,
                                               p_optim_data=hof_model['hof'])
            # guardar evaluaciones de todos los individuos del Hall of Fame
            memory_palace[model][period]['e_hof'].append(hof_eval)

# -- -------------------------------------------------------------------- RESULTS: AUC Min and Max cases -- #
# -- --------------------------------------------------------------------------- ----------------------- -- #
# -- Funcion de casos representativos

# diccionario para almacenar resultados de busqueda
auc_cases = {j: {i: {'data': {}} for i in ['auc_min', 'auc_max', 'hof_metrics']} for j in models}

# ciclo para busqueda de auc_min y auc_max
for model in models:
    auc_min = 1
    auc_max = 0
    auc_max_params = {}
    auc_min_params = {}
    for period in m_folds:
        auc_cases[model]['hof_metrics']['data'][period] = {}
        auc_s = []
        for i in range(0, 10):
            auc_s.append(memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'])

            # -- caso 1
            # El individuo de todos los HOF de todos los periodos que produjo la minima AUC
            if memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] < auc_min:
                auc_min = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_min']['data'] = memory_palace[model][period]['e_hof'][i]
                auc_min_params = memory_palace[model][period]['e_hof'][i]['params']

            # -- caso 2
            # El individuo de todos los HOF de todos los periodos que produjo la maxima AUC
            elif memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc'] > auc_max:
                auc_max = memory_palace[model][period]['e_hof'][i]['metrics']['test']['auc']
                auc_cases[model]['auc_max']['data'] = memory_palace[model][period]['e_hof'][i]
                auc_max_params = memory_palace[model][period]['e_hof'][i]['params']

        # Guardar info por periodo
        auc_cases[model]['hof_metrics']['data'][period]['auc_s'] = auc_s
        auc_cases[model]['hof_metrics']['data'][period]['auc_max'] = auc_max
        auc_cases[model]['hof_metrics']['data'][period]['auc_max_params'] = auc_max_params
        auc_cases[model]['hof_metrics']['data'][period]['auc_min'] = auc_min
        auc_cases[model]['hof_metrics']['data'][period]['auc_min_params'] = auc_min_params

# -- --------------------------------------------------------------- RESULTS: Global AUC Min & Max Cases -- #
# -- --------------------------------------------------------------- ----------------------------------- -- #

# Evaluation of auc_min and auc_max cases in all the models
global_cases = {model: {'auc_min': {}, 'auc_max': {}} for model in models}

# Global features (CORRECCION: )
global_features = fn.genetic_programed_features(p_data=datos, p_memory=7)

# Evaluate all global cases
for model in models:
    for case in ['auc_min', 'auc_max']:
        if model == 'model_1':
            global_cases[model][case] = fn.logistic_net(p_data=global_features,
                                                         p_params=auc_cases[model][case]['data']['params'])
        elif model == 'model_2':
            global_cases[model][case] = fn.ls_svm(p_data=global_features,
                                                   p_params=auc_cases[model][case]['data']['params'])
        elif model == 'model_3':
            global_cases[model][case] = fn.ann_mlp(p_data=global_features,
                                                    p_params=auc_cases[model][case]['data']['params'])

# -- ----------------------------------------------------------------- PLOT 2: Time Series Block T-Folds -- #
# -- ----------------------------------------------------------------- --------------------------------- -- #

# construccion de fechas para lineas verticales de division de cada fold
fechas_folds = []
for fold in m_folds:
    fechas_folds.append(m_folds[fold]['timestamp'].iloc[0])
    fechas_folds.append(m_folds[fold]['timestamp'].iloc[-1])

# grafica OHLC
plot_2 = vs.g_ohlc(p_ohlc=datos,
                   p_theme=theme_plot_2['p_theme'], p_dims=theme_plot_2['p_dims'],
                   p_labels=theme_plot_2['p_labels'], p_vlines=fechas_folds)

# mostrar grafica
# plot_2.show()

# -- PRIORIDAD 2
# ------------------------------------------------------------ Hacer una GRAFICA para todos los periodos -- #
# -- TITULO: Clases Observadas Vs Clases Ajustadas
# Para todos los datos, grafica de barras con Clases Verdaderas y Clases ajustadas.
# Renglon 1: Clases originales
# Renglon 2: Clases ajustadas con el 1er modelo utilizando los parametros del individuo max AUC global
# Renglon 3: Clases ajustadas con el 2do modelo utilizando los parametros del individuo max AUC global
# Renglon 4: Clases ajustadas con el 3er modelo utilizando los parametros del individuo max AUC global

obs_class = list(global_features['train_y']) + list(global_features['test_y'])
obs_class = [-1 if x == 0 else 1 for x in obs_class]

model_data = global_cases['model_1']['auc_min']['results']['data']
pred_class = list(model_data['train']['y_train_pred']) + list(model_data['test']['y_test_pred'])
pred_class = [-1 if x == 0 else 1 for x in pred_class]
x_series = list(datos['timestamp'])

# Hacer grafica
plot_3 = vs.g_relative_bars(p_x=x_series, p_y0=obs_class, p_y1=pred_class,
                            p_theme=theme_plot_3['p_theme'], p_dims=theme_plot_3['p_dims'])

# mostrar grafica
# plot_3.show()

# -- PRIORIDAD 3
# ------------------------------------ Hacer una GRAFICA con AUC_prom por modelo para todos los periodos -- #
# -- TITULO: Comparativa de Estabilidad de AUC entre modelos
# con los HoF de cada periodo, calcular promedio AUC_min y AUC_max. Mostrar en una linea los valores
# para todos los periodos, se tendran 3 lineas, 1 por modelo
# ejex: Periodo
# ejey: valor de AUC promedio por modelo (3 modelos)

minmax_auc_test = {i: {'x_period': [], 'y_mins': [], 'y_maxs': []} for i in models}

for model in models:
    minmax_auc_test[model]['x_period'] = list(auc_cases[model]['hof_metrics']['data'].keys())
    minmax_auc_test[model]['y_mins'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_min']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]
    minmax_auc_test[model]['y_maxs'] = [auc_cases[model]['hof_metrics']['data'][periodo]['auc_max']
                                        for periodo in list(auc_cases[model]['hof_metrics']['data'].keys())]

# Hacer grafica
plot_4 = vs.g_timeseries_auc(p_data_auc=minmax_auc_test, p_theme=theme_plot_4)

# mostrar grafica
# plot_4.show()

# -- PRIORIDAD 4
# --------------------------------------------------------------------- Hacer una TABLA para cada modelo -- #
# -- TITULO: Estabilidad de hyperparametros de modelo
# Para cada periodo, para cada modelo, mostrar los parametros del max_AUC.
# columnas: 01_2018 | 02_2018 | ... | nn_yyyy |
# renglones: max AUC  + parametros del modelo

data_stables = {model: {'df_auc_max': {period: {} for period in m_folds},
                        'df_auc_min': {period: {} for period in m_folds}} for model in models}

period_max_auc = {model: {period: {} for period in m_folds} for model in models}
period_min_auc = {model: {period: {} for period in m_folds} for model in models}

for model in list(models.keys()):
    for period in list(m_folds.keys()):
        period_max_auc[model][period] = auc_cases[model]['hof_metrics']['data'][period]['auc_max_params']
        period_min_auc[model][period] = auc_cases[model]['hof_metrics']['data'][period]['auc_min_params']

# Tabla 1: Parametros de modelos para los AUC max y AUC min
table_2 = {'model_1': {'max': pd.DataFrame(period_max_auc['model_1']),
                       'min': pd.DataFrame(period_min_auc['model_1'])},
           'model_2': {'max': pd.DataFrame(period_max_auc['model_2']),
                       'min': pd.DataFrame(period_min_auc['model_2'])},
           'model_3': {'max': pd.DataFrame(period_max_auc['model_3']),
                       'min': pd.DataFrame(period_min_auc['model_3'])}}

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
# ---------------------------------------------------------------------- ------------------------------ -- #

# incorporar valores para criterio 3 y 4
# Criterio 3 (25 y 50)
# Criterio 4 (1% del capital)
