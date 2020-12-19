
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
import numpy as np
import random
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from gplearn.genetic import SymbolicTransformer
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore", category=RuntimeWarning)


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

    # For monthly separation of the data
    if p_periodo == 'mes':
        # List of months in the dataset
        months = list(set(time.month for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = list(set(time.year for time in list(p_data['timestamp'])))
        m_data = {}
        # New key for every month_year
        for j in years:
            m_data.update({'m_' + str('0') + str(i) + '_' + str(j) if i <= 9 else str(i) + '_' + str(j):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.month == i) &
                                      (pd.to_datetime(p_data['timestamp']).dt.year == j)]
                           for i in months})
        return m_data

    # For quarterly separation of the data
    elif p_periodo == 'trimestre':
        # List of quarters in the dataset
        quarters = list(set(time.quarter for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = list(set(time.year for time in list(p_data['timestamp'])))
        q_data = {}
        # New key for every quarter_year
        for j in years:
            q_data.update({'q_' + str('0') + str(i) + '_' + str(j) if i <= 9 else str(i) + '_' + str(j):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.quarter == i) &
                                      (pd.to_datetime(p_data['timestamp']).dt.year == j)]
                           for i in quarters})
        return q_data

    # In the case a different label has been receieved
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
                                generations=50, tournament_size=20,  stopping_criteria=.65,
                                const_range=None, init_method='half and half', init_depth=(4, 16),
                                metric='pearson', parsimony_coefficient=0.01,
                                p_crossover=0.4, p_subtree_mutation=0.3, p_hoist_mutation=0.1,
                                p_point_mutation=0.2, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
                                warm_start=True)

    # resultado de ajuste con la funcion SymbolicTransformer
    model_fit = model.fit_transform(p_x, p_y)

    # dejar en un dataframe las variables
    data = pd.DataFrame(model_fit)

    # parametros de modelo
    model_params = model.get_params()

    # resultados
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data}

    return results


# -------------------------- MODEL: Multivariate Linear Regression Model with ELASTIC NET regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def logistic_net(p_data, p_params):
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
    en_model = LogisticRegression(l1_ratio=p_params['ratio'], C=p_params['c'], tol=1e-3,
                                  penalty='elasticnet', solver='saga', multi_class='ovr', n_jobs=-1,
                                  max_iter=1000, fit_intercept=False)

    # model fit
    en_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = en_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    # Confussion matrix
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = en_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = en_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = en_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1]) 

    return auc_test


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #
@ignore_warnings(category=ConvergenceWarning)
def ls_svm(p_data, p_params):
    """
    Least Squares Support Vector Machines

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

        p_kernel: str
                kernel de LS_SVM
                p_alpha = ['linear']

        p_c: float
            Valor de coeficiente C
            p_ratio = 0.1

        p_gamma: int
            Valor de coeficiente gamma
            p_iter = 0.1

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

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
    svm_model = SVC(C=p_params['c'], kernel=p_params['kernel'], gamma=p_params['gamma'],

                    shrinking=True, probability=True, tol=1e-5, cache_size=4000,
                    class_weight=None, verbose=False, max_iter=100000, decision_function_shape='ovr',
                    break_ties=False, random_state=None)

    # model fit
    svm_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = svm_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = svm_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = svm_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = svm_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': svm_model,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

    return r_models


# --------------------------------------------------- MODEL: Artificial Neural Net Multilayer Perceptron -- #
# --------------------------------------------------------------------------------------------------------- #

def ann_mlp(p_data, p_params):
    """
    Artificial Neural Network, particularly, a MultiLayer Perceptron for Supervised Classification

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

        hidden_layers: ()
        activation: float
        alpha: int
        learning_r: int
        learning_r_init: int

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

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

    # model function
    mlp_model = MLPClassifier(hidden_layer_sizes=p_params['hidden_layers'],
                              activation=p_params['activation'], alpha=p_params['alpha'],
                              learning_rate=p_params['learning_r'],
                              learning_rate_init=p_params['learning_r_init'],

                              batch_size='auto', solver='sgd', power_t=0.5, max_iter=80000, shuffle=False,
                              random_state=None, tol=1e-3, verbose=False, warm_start=False, momentum=0.5,
                              nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=10)

    # model fit
    mlp_model.fit(x_train, y_train)

    # fitted train values
    p_y_train_d = mlp_model.predict(x_train)
    p_y_result_train = pd.DataFrame({'y_train': y_train, 'y_train_pred': p_y_train_d})
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = mlp_model.predict_proba(x_train)

    # Accuracy rate
    acc_train = accuracy_score(list(y_train), p_y_train_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds_train = roc_curve(list(y_train), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = roc_auc_score(list(y_train), probs_train[:, 1])

    # fitted test values
    p_y_test_d = mlp_model.predict(x_test)
    p_y_result_test = pd.DataFrame({'y_test': y_test, 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = mlp_model.predict_proba(x_test)

    # Accuracy rate
    acc_test = accuracy_score(list(y_test), p_y_test_d)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(y_test), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for test data
    auc_test = roc_auc_score(list(y_test), probs_test[:, 1])

    # Return the result of the model
    r_models = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                            'matrix': {'train': cm_train, 'test': cm_test}},
                'model': mlp_model,
                'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                            'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                     'probs': probs_test, 'auc': auc_test}},
                'params': p_params}

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
    # datos_timestamp = datos_arf['timestamp'].copy()

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
    # equaciones = [i.__str__() for i in list(fun_sym['model'])]

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
    https://github.com/deap/deap

    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """

    # -- ------------------------------------------------------- OLS con regularizacion tipo Elastic Net -- #
    # ----------------------------------------------------------------------------------------------------- #
    if p_model['label'] == 'logistic-elasticnet':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_en
            del creator.Individual_en
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_en", base.Fitness, weights=(1.0,))
        creator.create("Individual_en", list, fitness=creator.FitnessMax_en)
        toolbox_en = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_en.register("attr_ratio", random.choice, p_model['params']['ratio'])
        toolbox_en.register("attr_c", random.choice, p_model['params']['c'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_en.register("Individual_en", tools.initCycle, creator.Individual_en,
                            (toolbox_en.attr_ratio, toolbox_en.attr_c), n=1)

        # population definition
        toolbox_en.register("population", tools.initRepeat, list, toolbox_en.Individual_en)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_en(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['ratio'])
            elif gene == 1:
                individual[1] = random.choice(p_model['params']['c'])

            return individual,

        # --------------------------------------------------- funcion de evaluacion para OLS Elastic Net -- #
        def evaluate_en(eva_individual):

            # output of genetic algorithm
            chromosome = {'ratio': eva_individual[0], 'c': eva_individual[1]}

            # model results
            model = logistic_net(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_en.register("mate", tools.cxOnePoint)
        toolbox_en.register("mutate", mutate_en)
        toolbox_en.register("select", tools.selTournament, tournsize=10)
        toolbox_en.register("evaluate", evaluate_en)

        population_size = 50
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 4

        en_pop = toolbox_en.population(n=population_size)
        en_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algorithm Implementation
        en_pop, en_log = algorithms.eaSimple(population=en_pop, toolbox=toolbox_en, stats=stats,
                                             cxpb=crossover_probability, mutpb=mutation_probability,
                                             ngen=number_of_generations,
                                             halloffame=en_hof, verbose=True)

        return {'population': en_pop, 'logs': en_log, 'hof': en_hof}

    # -- --------------------------------------------------------- Least Squares Support Vector Machines -- #
    # ----------------------------------------------------------------------------------------------------- #

    elif p_model['label'] == 'ls-svm':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_svm
            del creator.Individual_svm
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_svm", base.Fitness, weights=(1.0, ))
        creator.create("Individual_svm", list, fitness=creator.FitnessMax_svm)
        toolbox_svm = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_svm.register("attr_c", random.choice, p_model['params']['c'])
        toolbox_svm.register("attr_kernel", random.choice, p_model['params']['kernel'])
        toolbox_svm.register("attr_gamma", random.choice, p_model['params']['gamma'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_svm.register("Individual_svm", tools.initCycle, creator.Individual_svm,
                             (toolbox_svm.attr_c, toolbox_svm.attr_kernel, toolbox_svm.attr_gamma), n=1)

        # population definition
        toolbox_svm.register("population", tools.initRepeat, list, toolbox_svm.Individual_svm)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_svm(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['c'])
            elif gene == 1:
                if individual[1] == 'linear':
                    individual[1] = 'rbf'
                else:
                    individual[1] = 'linear'
            elif gene == 2:
                if individual[2] == 'scale':
                    individual[2] = 'auto'
                else:
                    individual[2] = 'scale'
            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_svm(eval_individual):

            # output of genetic algorithm
            chromosome = {'c': eval_individual[0], 'kernel': eval_individual[1], 'gamma': eval_individual[2]}

            # model results
            model = ls_svm(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_svm.register("mate", tools.cxOnePoint)
        toolbox_svm.register("mutate", mutate_svm)
        toolbox_svm.register("select", tools.selTournament, tournsize=10)
        toolbox_svm.register("evaluate", evaluate_svm)

        population_size = 50
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 4

        svm_pop = toolbox_svm.population(n=population_size)
        svm_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algortihm implementation
        svm_pop, svm_log = algorithms.eaSimple(population=svm_pop, toolbox=toolbox_svm, stats=stats,
                                               cxpb=crossover_probability, mutpb=mutation_probability,
                                               ngen=number_of_generations,
                                               halloffame=svm_hof, verbose=True)

        return {'population': svm_pop, 'logs': svm_log, 'hof': svm_hof}

    # -- ----------------------------------------------- Artificial Neural Network MultiLayer Perceptron -- #
    # ----------------------------------------------------------------------------------------------------- #

    elif p_model['label'] == 'ann-mlp':

        # borrar clases previas si existen
        try:
            del creator.FitnessMax_mlp
            del creator.Individual_mlp
        except AttributeError:
            pass

        # inicializar ga
        creator.create("FitnessMax_mlp", base.Fitness, weights=(1.0,))
        creator.create("Individual_mlp", list, fitness=creator.FitnessMax_mlp)
        toolbox_mlp = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_mlp.register("attr_hidden_layers", random.choice, p_model['params']['hidden_layers'])
        toolbox_mlp.register("attr_activation", random.choice, p_model['params']['activation'])
        toolbox_mlp.register("attr_alpha", random.choice, p_model['params']['alpha'])
        toolbox_mlp.register("attr_learning_r", random.choice, p_model['params']['learning_r'])
        toolbox_mlp.register("attr_learning_r_init", random.choice, p_model['params']['learning_r_init'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_mlp.register("Individual_mlp", tools.initCycle, creator.Individual_mlp,
                             (toolbox_mlp.attr_hidden_layers,
                              toolbox_mlp.attr_activation,
                              toolbox_mlp.attr_alpha,
                              toolbox_mlp.attr_learning_r,
                              toolbox_mlp.attr_learning_r_init), n=1)

        # population definition
        toolbox_mlp.register("population", tools.initRepeat, list, toolbox_mlp.Individual_mlp)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_mlp(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['hidden_layers'])
            elif gene == 1:
                individual[1] = random.choice(p_model['params']['activation'])
            elif gene == 2:
                individual[2] = random.choice(p_model['params']['alpha'])
            elif gene == 3:
                individual[3] = random.choice(p_model['params']['learning_r'])
            elif gene == 4:
                individual[4] = random.choice(p_model['params']['learning_r_init'])
            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_mlp(eval_individual):

            # output of genetic algorithm
            chromosome = {'hidden_layers': eval_individual[0], 'activation': eval_individual[1],
                          'alpha': eval_individual[2], 'learning_r': eval_individual[3],
                          'learning_r_init': eval_individual[4]}

            # model results
            model = ann_mlp(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,

        toolbox_mlp.register("mate", tools.cxOnePoint)
        toolbox_mlp.register("mutate", mutate_mlp)
        toolbox_mlp.register("select", tools.selTournament, tournsize=10)
        toolbox_mlp.register("evaluate", evaluate_mlp)

        population_size = 50
        crossover_probability = 0.8
        mutation_probability = 0.1
        number_of_generations = 4

        mlp_pop = toolbox_mlp.population(n=population_size)
        mlp_hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # ga algorithjm
        mlp_pop, mlp_log = algorithms.eaSimple(population=mlp_pop, toolbox=toolbox_mlp, stats=stats,
                                               cxpb=crossover_probability, mutpb=mutation_probability,
                                               ngen=number_of_generations,
                                               halloffame=mlp_hof, verbose=True)

        return {'population': mlp_pop, 'logs': mlp_log, 'hof': mlp_hof}

    return 'error, sin modelo seleccionado'


# ------------------------------------------------------------------------------ Evaluaciones en periodo -- #
# --------------------------------------------------------------------------------------------------------- #

def evaluaciones_periodo(p_features, p_optim_data, p_model):

    for params in list(p_optim_data):

        if p_model == 'model_1':
            parameters = {'ratio': params[0], 'c': params[1]}

            return logistic_net(p_data=p_features, p_params=parameters)

        elif p_model == 'model_2':
            parameters = {'c': params[0], 'kernel': params[1], 'gamma': params[2]}

            return ls_svm(p_data=p_features, p_params=parameters)

        elif p_model == 'model_3':
            parameters = {'hidden_layers': params[0], 'activation': params[1], 'alpha': params[2],
                          'learning_r': params[3], 'learning_r_init': params[4]}

            return ann_mlp(p_data=p_features, p_params=parameters)
