
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler  # estandarizacion de variables
from gplearn.genetic import SymbolicTransformer                               # variables simbolicas


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
    p_data = datos

    Returns
    -------
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

def Elastic_Net(p_data, p_params):
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
                p_alpha = alphas[1e-3]

        p_iter: int
            Number of iterations until stop the model fit process
            p_iter = 1e6

        p_intercept: bool
            Si se incluye o no el intercepto en el ajuste
            p_intercept = True

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    """

    x_train = p_data['train_x']
    y_train = p_data['train_y']

    x_test = p_data['test_x']
    y_test = p_data['test_y']

    p_alpha = p_params['alpha']
    p_iter = p_params['iter']

    # Fit ElasticNet regression
    enetreg = ElasticNet(alpha=p_alpha, normalize=False, max_iter=p_iter, l1_ratio=0.5, fit_intercept=False)
    enetreg.fit(x_train, y_train)
    y_p_enet = enetreg.predict(x_test)

    # Return the result of the model
    r_models = {'elasticnet': {'rss': sum((y_p_enet - y_test) ** 2),
                               'predict': y_p_enet,
                               'model': enetreg,
                               'intercept': enetreg.intercept_,
                               'coef': enetreg.coef_}}

    return r_models


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def f_SVM(p_data):


    return 1


# ----------------------- FUNCTION: Simultaneous Feature Engieering/Selection & Hyperparameter Optimizer -- #
# ------------------------------------------------------- ------------------------------------------------- #

def f_FeatureModelOptimizer(p_data):

    # ----------------------------------------------------------- ingenieria de variables autoregresivas -- #
    # ----------------------------------------------------------- -------------------------------------- -- #

    # funcion para generar variables autoregresivas
    datos_arf = f_autoregressive_features(p_data=datos, p_nmax=7)

    # Visualizacion: head del DataFrame
    datos_arf.head(5)

    # ----------------------------------------------------------------- ingenieria de variables hadamard -- #
    # ----------------------------------------------------------------- -------------------------------- -- #

    # funcion para generar variables con producto hadamard
    datos_had = f_hadamard_features(p_data=datos_arf, p_nmax=29)

    # Visualizacion: head del DataFrame
    datos_had.head(5)

    # --------------------------------------------------------------- ingenieria de variables simbolicas -- #
    # --------------------------------------------------------------- ---------------------------------- -- #

    # Lista de operaciones simbolicas
    fun_sym = symbolic_features(p_x=datos_had.iloc[:, 3:], p_y=datos_had.iloc[:, 2])

    # variables
    datos_sym = fun_sym['data']
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(fun_sym['data'].iloc[0, :]))]

    # ecuaciones de todas las variables
    equaciones = [i.__str__() for i in list(fun_sym['model'])]

    # -- Para cada K Fold:
    # -- -- Hacer proceso de ing de variables (autoregresivas (1 semana max resagos), hadamard, simbolicas)
    # -- -- Dividir datos 80-20
    # -- -- Hacer proceso de seleccion de variables y optimizacion de hiperparametros con el 80 y GP
    # -- -- Hacer prediccion con el 20
    # -- -- Obtener metricas de desempeño del modelo

    return 1
