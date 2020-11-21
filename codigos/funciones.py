
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler  # estandarizacion de variables
from gplearn.genetic import SymbolicTransformer                               # variables simbolicas

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------- transformacio de datos -- #
# --------------------------------------------------------------------------------- -------------------- -- #

def f_transformacion(p_datos, p_trans):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_trans: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)

    p_datos: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    if p_trans == 'Standard':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        # armar objeto de salida
        p_datos[list(p_datos.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_trans == 'Robust':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        # armar objeto de salida
        p_datos[list(p_datos.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_trans == 'Scale':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        p_datos[list(p_datos.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_datos


# ------------------------------------------------------- FUNCTION: Divide the data in M-Folds (montlhy) -- #
# ------------------------------------------------------- ------------------------------------------------- #

def f_m_folds(p_data, p_periodo):
    """
    Funcion para dividir los datos en m-bloques, donde m es un valor basado en tiempo:
        m={'mensual', 'trimestral'}

    Parameters
    ----------
    p_data : pd.DataFrame
        DataFrame con los datos a dividir

    p_periodo : str
        'mes': Para dividir datos por periodos mensuales
        'trimestre' para dividir datos por periodos trimestrales

    Returns
    -------
    {'periodo_': pd.DataFrame}

    """

    if p_periodo == 'mes':
        per = list(set(time.month for time in list(p_data['timestamp'])))
        return {'periodo_' + str(i):
                      p_data[pd.to_datetime(p_data['timestamp']).dt.month == i].copy() for i in per}

    elif p_periodo == 'trimestre':
        per = list(set(time.quarter for time in list(p_data['timestamp'])))
        return {'periodo_' + str(i):
                      p_data[pd.to_datetime(p_data['timestamp']).dt.quarter == i].copy() for i in per}

    return 'Error: verificar parametros de entrada'


# ------------------------------------------------------------------------------ Autoregressive Features -- #
# --------------------------------------------------------------------------------------------------------- #

def f_autoregressive_features(p_data, p_nmax):
    """
    Creacion de variables de naturaleza autoregresiva (resagos, promedios, diferencias)

    Parameters
    ----------
    p_data: pd.DataFrame
        Con columnas OHLCV para construir los features

    p_nmax: int
        Para considerar n calculos de features (resagos y promedios moviles)

    Returns
    -------
    r_features: pd.DataFrame
        Con dataframe de features (timestamp + co + co_d + features)

    """

    # reasignar datos
    data = p_data.copy()

    # pips descontados al cierre
    data['co'] = (data['close'] - data['open']) * 10000

    # pips descontados alcistas
    data['ho'] = (data['high'] - data['open']) * 10000

    # pips descontados bajistas
    data['ol'] = (data['open'] - data['low']) * 10000

    # pips descontados en total (medida de volatilidad)
    data['hl'] = (data['high'] - data['low']) * 10000

    # clase a predecir
    data['co_d'] = [1 if i > 0 else 0 for i in list(data['co'])]

    # ciclo para calcular N features con logica de "Ventanas de tamaño n"
    for n in range(0, p_nmax):

        # rezago n de Open Interest
        data['lag_vol_' + str(n + 1)] = data['volume'].shift(n + 1)

        # rezago n de Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)

        # rezago n de High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)

        # rezago n de High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)

        # promedio movil de open-high de ventana n
        data['ma_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).mean()

        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 1)] = data['ol'].rolling(n + 1).mean()

        # promedio movil de ventana n
        data['ma_ho_' + str(n + 1)] = data['ho'].rolling(n + 1).mean()

        # promedio movil de ventana n
        data['ma_hl_' + str(n + 1)] = data['hl'].rolling(n + 1).mean()

    # asignar timestamp como index
    data.index = pd.to_datetime(data.index)
    # quitar columnas no necesarias para modelos de ML
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
    # borrar columnas donde exista solo NAs
    r_features = r_features.dropna(axis='columns', how='all')
    # borrar renglones donde exista algun NA
    r_features = r_features.dropna(axis='rows')
    # convertir a numeros tipo float las columnas
    r_features.iloc[:, 2:] = r_features.iloc[:, 2:].astype(float)
    # reformatear columna de variable binaria a 0 y 1
    r_features['co_d'] = [0 if i <= 0 else 1 for i in r_features['co_d']]
    # resetear index
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# ------------------------------------------------------------------------------------ Hadamard Features -- #
# --------------------------------------------------------------------------------------------------------- #

def f_hadamard_features(p_data, p_nmax):
    """
    Creacion de variables haciendo un producto hadamard entre todas las variables

    Parameters
    ----------
    p_data: pd.DataFrame
        Con columnas OHLCV para construir los features

    p_nmax: int
        Para considerar n calculos de features (resagos y promedios moviles)

    Returns
    -------
    r_features: pd.DataFrame
        Con dataframe de features con producto hadamard

    """

    # ciclo para crear una combinacion secuencial
    for n in range(p_nmax):

        # lista de features previos
        list_hadamard = ['lag_vol_' + str(n + 1),
                         'lag_ol_' + str(n + 1),
                         'lag_ho_' + str(n + 1),
                         'lag_hl_' + str(n + 1)]

        # producto hadamard con los features previos
        for x in list_hadamard:
            p_data['h_' + x + '_' + 'ma_ol_' + str(n + 1)] = p_data[x] * p_data['ma_ol_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_ho_' + str(n + 1)] = p_data[x] * p_data['ma_ho_' + str(n + 1)]
            p_data['h_' + x + '_' + 'ma_hl_' + str(n + 1)] = p_data[x] * p_data['ma_hl_' + str(n + 1)]

    return p_data


# ------------------------------------------------------------------ MODEL: Symbolic Features Generation -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y):
    """
    Funcion para crear regresores no lineales

    Parameters
    ----------
    p_x: pd.DataFrame
        with regressors or predictor variables
        p_x = data_features.iloc[0:30, 3:]

    p_y: pd.DataFrame
        with variable to predict
        p_y = data_features.iloc[0:30, 1]

    Returns
    -------
    score_gp: float
        error of prediction

    """

    # funcion de generacion de variables simbolicas
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
                                population_size=5000, hall_of_fame=100, n_components=20,
                                generations=20, tournament_size=20,  stopping_criteria=.05,
                                const_range=None, init_method='half and half', init_depth=(4, 12),
                                metric='pearson', parsimony_coefficient=0.001,
                                p_crossover=0.4, p_subtree_mutation=0.2, p_hoist_mutation=0.1,
                                p_point_mutation=0.3, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
                                warm_start=True)

    # resultado de ajuste con la funcion SymbolicTransformer
    model_fit = model.fit_transform(p_x, p_y)

    # dejar en un dataframe las variables
    data = pd.DataFrame(model_fit)
    # data.columns = list(model.feature_names)

    # parametros de modelo
    model_params = model.get_params()

    # resultados
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data}

    return results


# -------------------------- MODEL: Multivariate Linear Regression Model with ELASTIC NET regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def ols_elastic_net(p_data, p_params):
    """
    Funcion para ajustar varios modelos lineales

    Parameters
    ----------
    p_data: dict
        Diccionario con datos de entrada como los siguientes:

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]

    p_params: dict
        Diccionario con parametros de entrada para modelos, como los siguientes

        p_alpha: float
                alpha for the models
                p_alpha = 0.1

        p_ratio: float
            elastic net ratio between L1 and L2 regularization coefficients
            p_ratio = 0.1

        p_iterations: int
            Number of iterations until stop the model fit process
            p_iter = 200

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    References
    ----------
    ElasticNet
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    """

    # Datos de entrenamiento
    x_train = p_data['train_x']
    y_train = p_data['train_y']

    # Datos de prueba
    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # alpha, l1_ratio,

    # computations parameters
    # fit_intercept, normalize, precompute, copy_X, tol, warm_start, positive, selection

    # Fit model
    en_model = ElasticNet(alpha=p_params['alpha'], l1_ratio=p_params['ratio'],
                          max_iter=10000, fit_intercept=False, normalize=False, precompute=True,
                          copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=123,
                          selection='random')

    # model fit
    en_model.fit(x_train, y_train)

    # fitted train values
    p_y_train = en_model.predict(x_train)
    p_y_train_d = [1 if i > 0 else 0 for i in p_y_train]
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])

    # fitted test values
    p_y_test = en_model.predict(x_test)
    p_y_test_d = [1 if i > 0 else 0 for i in p_y_test]
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': en_model, 'intercept': en_model.intercept_, 'coef': en_model.coef_}

    return r_models


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def ls_svm(p_data, p_params):
    """
    Least Squares Support Vector Machines

    Parameters
    ----------
    p_data
    p_params

    Returns
    -------

    References
    ----------
    https://scikit-learn.org/stable/modules/svm.html#

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # C, kernel, degree (if kernel = poly), gamma (if kernel = {rbf, poly, sigmoid},
    # coef0 (if kernel = {poly, sigmoid})

    # computations parameters
    # shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape,
    # break_ties, random_state

    # model function
    svm_model = SVC(C=p_params['C'], kernel=p_params['kernel'], gamma=p_params['gamma'],

                    degree=3, coef0=0, shrinking=True, probability=False, tol=1e-3, cache_size=2000,
                    class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                    break_ties=False, random_state=None)

    # model fit
    svm_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = svm_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])

    # fitted test values
    p_y_test_d = svm_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': svm_model, 'intercept': svm_model.intercept_, 'coef': svm_model.coef_}

    return r_models


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def ann_mlp(p_data, p_params):
    """
    Artificial Neural Network, particularly, a MultiLayer Perceptron for Supervised Classification

    Parameters
    ----------
    p_data
    p_params

    Returns
    -------

    References
    ----------
    https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    # ------------------------------------------------------------------------------ FUNCTION PARAMETERS -- #
    # model hyperparameters
    # hidden_layer_sizes, activation, solver, alpha, learning_rate,

    # batch_size, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose,
    # warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction

    # computations parameters

    mlp_model = MLPClassifier(hidden_layer_sizes=p_params['hidden_layer_sizes'],
                              activation=p_params['activation'], alpha=p_params['alpha'],
                              learning_rate=p_params['learning_rate'],
                              learning_rate_init=p_params['learning_rate_init'],

                              batch_size='auto', solver='sgd', power_t=0.5,
                              max_iter=200, shuffle=False,
                              random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                              nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                              n_iter_no_change=10)

    # model fit
    mlp_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = mlp_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])

    # fitted test values
    p_y_test_d = mlp_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': mlp_model}

    return r_models


# ----------------------- FUNCTION: Simultaneous Feature Engieering/Selection & Hyperparameter Optimizer -- #
# ------------------------------------------------------- ------------------------------------------------- #

def genetic_programed_features(p_data, p_memory):
    """
    El uso de programacion genetica para generar variables independientes simbolicas

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        p_data = m_folds['periodo_1']

    p_memory: int
        valor de memoria maxima para hacer calculo de variables autoregresivas
        p_memory = 7

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'test_x': pd.DataFrame, 'test_y': pd.DataFrame}

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """

    # ----------------------------------------------------------- ingenieria de variables autoregresivas -- #
    # ----------------------------------------------------------- -------------------------------------- -- #

    # funcion para generar variables autoregresivas
    datos_arf = f_autoregressive_features(p_data=p_data, p_nmax=p_memory)

    # separacion de variable dependiente
    datos_y = datos_arf['co_d'].copy()

    # separacion de variable dependiente
    datos_timestamp = datos_arf['timestamp'].copy()

    # separacion de variables independientes
    datos_arf = datos_arf.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)

    # ----------------------------------------------------------------- ingenieria de variables hadamard -- #
    # ----------------------------------------------------------------- -------------------------------- -- #

    # funcion para generar variables con producto hadamard
    datos_had = f_hadamard_features(p_data=datos_arf, p_nmax=p_memory)

    # --------------------------------------------------------------- ingenieria de variables simbolicas -- #
    # --------------------------------------------------------------- ---------------------------------- -- #

    # Lista de operaciones simbolicas
    fun_sym = symbolic_features(p_x=datos_had, p_y=datos_y)

    # variables
    datos_sym = fun_sym['data']
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

    # ecuaciones de todas las variables
    equaciones = [i.__str__() for i in list(fun_sym['model'])]

    # datos para utilizar en la siguiente etapa
    datos_modelo = pd.concat([datos_arf.copy(), datos_had.copy(), datos_sym.copy()], axis=1)
    model_data = {}

    # -- -- Dividir datos 80-20
    xtrain, xtest, ytrain, ytest = train_test_split(datos_modelo, datos_y, test_size=.2, shuffle=False)

    # division de datos
    model_data['train_x'] = xtrain
    model_data['train_y'] = ytrain
    model_data['test_x'] = xtest
    model_data['test_y'] = ytest

    return model_data


# -------------------------------------------------------------------------- FUNCTION: Genetic Algorithm -- #
# ------------------------------------------------------- ------------------------------------------------- #

def genetic_algo_optimisation(p_data, p_model):
    """
    El uso de algoritmos geneticos para optimizacion de hiperparametros de varios modelos

    Parameters
    ----------
    p_model: dict
        'label' con etiqueta del modelo, 'params' llaves con parametros y listas de sus valores a optimizar

    p_data: pd.DataFrame
        data frame con datos del m_fold

    Returns
    -------
    r_model_ols_elasticnet: dict
        resultados de modelo OLS con regularizacion elastic net

    r_model_ls_svm: dict
        resultados de modelo Least Squares Support Vector Machine

    r_model_ann_mlp: dict
        resultados de modelo Red Neuronal Artificial tipo perceptron multicapa

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """

    # -- ------------------------------------------------------- OLS con regularizacion tipo Elastic Net -- #
    if p_model['label'] == 'ols-elasticnet':

        # -- Genetic Algorithm Function -- #

        # output of genetic algorithm
        chromosome_eval = {'alpha': 0.1, 'ratio': 0.5}

        # model evaluation
        model_ols_elasticnet = ols_elastic_net(p_data=p_data, p_params=chromosome_eval)

        # model result
        r_model_ols_elasticnet = model_ols_elasticnet

        # return of function
        return r_model_ols_elasticnet

    # -- --------------------------------------------------------- Least Squares Support Vector Machines -- #
    elif p_model['label'] == 'ls-svm':

        # -- Genetic Algorithm Function -- #

        # output of genetic algorithm
        chromosome_eval = {'C': 1, 'kernel': 'linear', 'gamma': 1e-3}

        # model evaluation
        model_ls_svm = ls_svm(p_data=p_data, p_params=chromosome_eval)

        # model result
        r_model_ls_svm = model_ls_svm

        # return of function
        return r_model_ls_svm

    # -- ----------------------------------------------- Artifitial Neural Network MultiLayer Perceptron -- #
    elif p_model['label'] == 'ann-mlp':

        # -- Genetic Algorithm Function -- #

        # output of genetic algorithm
        chromosome_eval = {'hidden_layer_sizes': 100, 'activation': 'relu', 'alpha': 0.0001,
                           'batch_size': 'auto', 'learning_rate': 'constant', 'learning_rate_init': 0.001}

        # model evaluation
        model_ann_mlp = ann_mlp(p_data=p_data, p_params=chromosome_eval)

        # model result
        r_model_ann_mlp = model_ann_mlp

        # return of function
        return r_model_ann_mlp

    return 'error, sin modelo seleccionado'
