
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

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]

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
# ohlc.show()

# ----------------------------------------------------------------------- analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos
# datos.describe()

# ------------------------------------------------------------------------ Division en K-Folds mensuales -- #
# ------------------------------------------------------------------------ ----------------------------- -- #

# -- Division de periodos de datos, sin filtracion, en amplitudes de 1 mes para obtener 12 "Folds".
m_folds = fn.f_m_folds(p_data=datos, p_periodo='trimestre')

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


# -- -------------------------------------------------------------------------- Caracterizacion de casos -- #
# -- -------------------------------------------------------------------------- ------------------------ -- #

# Hacer este proceso para todos los modelos
# Paso 1: Encontrar el individuo en todos los HOF de todos los periodos que cumpla con la regla
# Paso 2: Extraer todos los datos disponibles del individuo y modelo elegido
# Paso 3: Mostrar grafica de precios OHLC del periodo (Train y Test del periodo)
# Paso 4: Mostrar grafica resumen del individuo encontrado (ROC + AUC + Modelo + Periodo)
# Paso 5: Mostrar grafica de barras con Clases Verdaderas y Clases ajustadas (Train y Test del periodo)
# Paso 6: Mostrar tabla con parametros del modelo

# -- ------------------------------------------------------------------------- Backtest global de modelo -- #
# -- ------------------------------------------------------------------------- ------------------------- -- #

# Utilizar cada uno de los 3 individuo y hacer un backtest, con todos los modelos, para todos los periodos

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

# -- GRAFICA PRECISION : DATOS + 3 MODELOS
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
