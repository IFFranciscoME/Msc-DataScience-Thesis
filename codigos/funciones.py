
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler  # estandarizacion de variables
from gplearn.genetic import SymbolicTransformer                               # variables simbolicas
from sklearn.svm import SVC


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

    Documentation
    -------------
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
    en_model = ElasticNet(alpha=p_params['alpha'][0], l1_ratio=p_params['ratio'][0],
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
    Parameters
    ----------
    p_data
    p_params

    Returns
    -------

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    svm_model = SVC(kernel='linear', C=1, gamma=1e-3)

    svm_model.fit(x_train, y_train)

    return 1


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def ann_mlp(p_data, p_params):
    """
    Parameters
    ----------
    p_data
    p_params

    Returns
    -------

    """

    return 1


# ----------------------- FUNCTION: Simultaneous Feature Engieering/Selection & Hyperparameter Optimizer -- #
# ------------------------------------------------------- ------------------------------------------------- #

def f_FeatureModelOptimizer(p_data, p_memory, p_model):
    """

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos

    p_memory: int
        valor de memoria maxima para hacer calculo de variables autoregresivas

    p_model: dict
        'label' con etiqueta del modelo, 'params' llaves con parametros y listas de sus valores a optimizar

    p_data = m_folds['periodo_1']
    p_memory = 7
    p_model = models['model_1']

    Returns
    -------

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

    model_data['train_x'] = xtrain
    model_data['train_y'] = ytrain
    model_data['test_x'] = xtest
    model_data['test_y'] = ytest

    # Elegir valor de parametro para todos y cada uno de los parametros
    # Correr modelo

    # -- ------------------------------------------------------- OLS con regularizacion tipo Elastic Net -- #
    if p_model['label'] == 'ols-elasticnet':
        model_ols_elasticnet = ols_elastic_net(p_data=model_data,
                                               p_params=p_model['params'])
        print(model_ols_elasticnet)

    # -- --------------------------------------------------------- Least Squares Support Vector Machines -- #
    if p_model == 'ls-svm':
        model_ls_svm = ls_svm(p_data=model_data,
                              p_params=0.5)
        print(model_ls_svm)

    # -- ---------------------------------------------- Artifitial Neural Network Multi Layer Perceptron -- #
    if p_model == 'ann-mlp':
        model_ann_mlp = ann_mlp(p_data=model_data,
                                p_params=0.5)
        print(model_ann_mlp)

    # -- -- Hacer prediccion con el 20
    # -- -- Obtener metricas de desempeño

    return 1
