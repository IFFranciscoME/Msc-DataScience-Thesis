
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: Masters in Data Science Thesis Project                                                     -- #
# -- Statistical Learning and Genetic Methods to Design, Optimize and Calibrate Trading Systems          -- #
# -- File: functions.py - python script with general functions                                           -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- License: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/IFFranciscoME/Msc_Thesis                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

from pickle import TRUE
import random
import warnings
import logging
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import data as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

from datetime import datetime
from scipy.stats import kurtosis as m_kurtosis
from scipy.stats import skew as m_skew

from gplearn.genetic import SymbolicTransformer
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# ------------------------------------------------------------------------------- transformacio de datos -- #
# --------------------------------------------------------------------------------- -------------------- -- #

def data_scaler(p_data, p_trans):
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
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_trans == 'Robust':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        # armar objeto de salida
        p_data[list(p_data.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_trans == 'Scale':

        # estandarizacion de todas las variables independientes
        lista = p_data[list(p_data.columns[1:])]

        p_data[list(p_data.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_data


# --------------------------------------------------------------------------- Divide the data in T-Folds -- #
# --------------------------------------------------------------------------- ----------------------------- #

def t_folds(p_data, p_period):
    """
    Function to separate in T-Folds the data, the functions guarantees not having filtrations
    
    Parameters
    ----------
    p_data : pd.DataFrame
        DataFrame with data
    
    p_period : str
        'month': every T-Fold will be of one month of historical data
        'quarter': every T-Fold will be of three months (quarter) of historical data
        'year': every T-Fold will be of twelve months of historical data
        'bi-year': every T-Fold will be of 2 years of historical data
        '80-20': Hold out method, 80% for training and 20% for testing
    
    Returns
    -------
    m_data or q_data : 'period_'
    
    References
    ----------
    https://web.stanford.edu/~hastie/ElemStatLearn/
    """

    # data scaling by standarization
    # p_data.iloc[:, 1:] = data_scaler(p_data=p_data.copy(), p_trans='Standard')

    # For quarterly separation of the data
    if p_period == 'quarter':
        # List of quarters in the dataset
        quarters = list(set(time.quarter for time in list(p_data['timestamp'])))
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        q_data = {}
        # New key for every quarter_year
        for y in sorted(list(years)):
            q_data.update({'q_' + str('0') + str(i) + '_' + str(y) if i <= 9 else str(i) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == i)]
                           for i in quarters})
        return q_data

    # For semester separation of the data
    elif p_period == 'semester':
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        s_data = {}
        # New key for every semester_year
        for y in sorted(list(years)):
            s_data.update({'s_' + str('0') + str(1) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 1) |
                                      (pd.to_datetime(p_data['timestamp']).dt.quarter == 2))]})

            s_data.update({'s_' + str('0') + str(2) + '_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y) &
                                      ((pd.to_datetime(p_data['timestamp']).dt.quarter == 3) |
                                       (pd.to_datetime(p_data['timestamp']).dt.quarter == 4))]})

        return s_data

    # For yearly separation of the data
    elif p_period == 'year':
        # List of years in the dataset
        years = set(time.year for time in list(p_data['timestamp']))
        y_data = {}
        
        # New key for every year
        for y in sorted(list(years)):
            y_data.update({'y_' + str(y):
                               p_data[(pd.to_datetime(p_data['timestamp']).dt.year == y)]})
                               
        return y_data

    # For bi-yearly separation of the data
    elif p_period == 'bi-year':
        # List of years in the dataset
        years = sorted(list(set(time.year for time in list(p_data['timestamp']))))
        # dict to store data
        b_y_data = {}
        # even or odd number of years
        folds = len(years)%2
        
        # even number of years
        if folds > 0:
            # fill years
            for y in np.arange(0, len(years), 2):
                b_y_data.update({'b_y_' + str(y):
                                p_data[(pd.to_datetime(p_data['timestamp']).dt.year == years[y]) |
                                       (pd.to_datetime(p_data['timestamp']).dt.year == years[y+1])]})
        # odd number of years (pending)
        else:
            # fill years
            for y in np.arange(0, len(years), 2):
                b_y_data.update({'b_y_' + str(y):
                                p_data[(pd.to_datetime(p_data['timestamp']).dt.year == years[y]) |
                                       (pd.to_datetime(p_data['timestamp']).dt.year == years[y+1])]})

        return b_y_data

    # For yearly separation of the data
    elif p_period == '80-20':

        # List of years in the dataset
        years = sorted(list(set(time.year for time in list(p_data['timestamp']))))
        
        # dict to store data
        a_8 = int(len(years)*0.80) 
        a_2 = int(len(years)*0.20)
        
        # data construction
        y_80_20_data = {'h_8': p_data[pd.to_datetime(p_data['timestamp']).dt.year.isin(years[0:a_8])],
                        'h_2': p_data[pd.to_datetime(p_data['timestamp']).dt.year.isin(years[a_8:a_8+a_2])]}
    
        return y_80_20_data

    # In the case a different label has been receieved
    return 'Error: verify parameters'


# ------------------------------------------------------------------------------ Autoregressive Features -- #
# --------------------------------------------------------------------------------------------------------- #

def autoregressive_features(p_data, p_nmax):
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

def hadamard_features(p_data, p_nmax):
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


# --------------------------------------------- FUNCTION: Autoregressive and Hadamard Feature Engieering -- #
# ------------------------------------------------------- ------------------------------------------------- #

def linear_features(p_data, p_memory):
    """
    autoregressive and hadamard product for feature engineering

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

    """

    # ----------------------------------------------------------- ingenieria de variables autoregresivas -- #
    # ----------------------------------------------------------- -------------------------------------- -- #

    # funcion para generar variables autoregresivas
    data_ar = autoregressive_features(p_data=p_data, p_nmax=p_memory)

    # separacion de variable dependiente
    data_y = data_ar['co_d'].copy()

    # separacion de variable dependiente
    # datos_timestamp = datos_arf['timestamp'].copy()

    # separacion de variables independientes
    data_arf = data_ar.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)

    # ----------------------------------------------------------------- ingenieria de variables hadamard -- #
    # ----------------------------------------------------------------- -------------------------------- -- #

    # funcion para generar variables con producto hadamard
    datos_had = hadamard_features(p_data=data_arf, p_nmax=p_memory)

    # datos para utilizar en la siguiente etapa
    features_data = pd.concat([data_y.copy(), datos_had.copy()], axis=1)

    # keep the timestamp as index
    features_data.index = data_ar['timestamp']
  
    return features_data


# ------------------------------------------------------------------ MODEL: Symbolic Features Generation -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y, p_params):
    """
    Feature engineering process with symbolic variables by using genetic programming. 

    Parameters
    ----------
    p_x: pd.DataFrame / np.array / list
        with regressors or predictor variables

        p_x = data_features.iloc[:, 1:]

    p_y: pd.DataFrame / np.array / list
        with variable to predict

        p_y = data_features.iloc[:, 0]

    p_params: dict
        with parameters for the genetic programming function

        p_params = {'functions': ["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
        'population': 5000, 'tournament':20, 'hof': 20, 'generations': 5, 'n_features':20,
        'init_depth': (4,8), 'init_method': 'half and half', 'parsimony': 0.1, 'constants': None,
        'metric': 'pearson', 'metric_goal': 0.65, 
        'prob_cross': 0.4, 'prob_mutation_subtree': 0.3,
        'prob_mutation_hoist': 0.1. 'prob_mutation_point': 0.2,
        'verbose': True, 'random_cv': None, 'parallelization': True, 'warm_start': True }

    Returns
    -------
    results: dict
        With response information

        {'fit': model fitted, 'params': model parameters, 'model': model,
         'data': generated data with variables, 'best_programs': models best programs}

    References
    ----------
    https://gplearn.readthedocs.io/en/stable/reference.html#gplearn.genetic.SymbolicTransformer
    
    """
    
    # Function to produce Symbolic Features
    model = SymbolicTransformer(function_set=p_params['functions'], population_size=p_params['population'],
                                tournament_size=p_params['tournament'], hall_of_fame=p_params['hof'],
                                generations=p_params['generations'], n_components=p_params['n_features'],

                                init_depth=p_params['init_depth'], init_method=p_params['init_method'],
                                parsimony_coefficient=p_params['parsimony'],
                                const_range=p_params['constants'],
                                
                                metric=p_params['metric'], stopping_criteria=p_params['metric_goal'],

                                p_crossover=p_params['prob_cross'],
                                p_subtree_mutation=p_params['prob_mutation_subtree'],
                                p_hoist_mutation=p_params['prob_mutation_hoist'],
                                p_point_mutation=p_params['prob_mutation_point'],

                                verbose=p_params['verbose'], warm_start=p_params['warm_start'],
                                random_state=123, n_jobs=-1 if p_params['parallelization'] else 1,
                                feature_names=p_x.columns)

    # SymbolicTransformer fit
    model_fit = model.fit_transform(p_x, p_y)

    # output data of the model
    data = pd.DataFrame(model_fit)

    # parameters of the model
    model_params = model.get_params()

    # best programs dataframe
    best_programs = {}
    for p in model._best_programs:
        factor_name = 'sym' + str(model._best_programs.index(p) + 1)
        best_programs[factor_name] = {'fitness': p.fitness_, 'expression': str(p),
                                      'depth': p.depth_, 'length': p.length_}

    # format and sorting
    best_programs = pd.DataFrame(best_programs).T
    best_programs = best_programs.sort_values(by='fitness', ascending=False)

    # results
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data,
               'best_programs': best_programs, 'details': model.run_details_}

    return results


# ------------------------------------------------- FUNCTION: Genetic Programming for Feature Engieering -- #
# ------------------------------------------------- ------------------------------------------------------- #

def genetic_programed_features(p_data):
    """
    El uso de programacion genetica para generar variables independientes simbolicas

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        p_data = m_folds['periodo_1']

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'test_x': pd.DataFrame, 'test_y': pd.DataFrame}

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """
   
    # separacion de variable dependiente
    datos_y = p_data['co_d'].copy()

    # separacion de variables independientes
    datos_had = p_data.drop(['co_d'], axis=1, inplace=False)

    # --------------------------------------------------------------- ingenieria de variables simbolicas -- #
    # --------------------------------------------------------------- ---------------------------------- -- #

    # Lista de operaciones simbolicas
    sym_data = symbolic_features(p_x=datos_had, p_y=datos_y, p_params=dt.symbolic_params)

    # variables
    datos_sym = sym_data['data']
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(sym_data['data'].iloc[0, :]))]
    datos_sym.index = datos_had.index

    # datos para utilizar en la siguiente etapa
    datos_modelo = pd.concat([datos_had.copy(), datos_sym.copy()], axis=1)
    model_data = {}

    # -- -- Dividir datos 80-20
    xtrain, xtest, ytrain, ytest = train_test_split(datos_modelo, datos_y, test_size=.2, shuffle=False)

    # division de datos
    model_data['train_x'] = xtrain
    model_data['train_y'] = ytrain
    model_data['test_x'] = xtest
    model_data['test_y'] = ytest

    return {'model_data': model_data, 'sym_data': sym_data}


# --------------------------------------------------------- EXPLORATORY DATA ANALYSIS & FEATURES METRICS -- #
# --------------------------------------------------------- ----------------------------------------------- #

def data_profile(p_data, p_type, p_mult):
    """
    OHLC Prices Profiling (Inspired in the pandas-profiling existing library)

    Parameters
    ----------

    p_data: pd.DataFrame
        A data frame with columns of data to be processed

    p_type: str
        indication of the data type: 
            'ohlc': dataframe with TimeStamp-Open-High-Low-Close columns names
            'ts': dataframe with unknown quantity, meaning and name of the columns
    
    p_mult: int
        multiplier to re-express calculation with prices,
        from 100 to 10000 in forex, units multiplication in cryptos, 1 for fiat money based assets
        p_mult = 10000

    Return
    ------
    r_data_profile: dict
        {}
    
    References
    ----------
    https://github.com/pandas-profiling/pandas-profiling


    """

    # p_data.columns

    # -- OHLCV PROFILING -- #
    if p_type == 'ohlc':

        # -- init and end dates, amount of data, data type, range of values (all values)
        # -- missing data (granularity vs calendar if data is based on timestamp labeling)
        # -- CO (grow), HL (volatility), OL (downside move), HO (upside move)
        # -- -- min, max, mean, median, sd, IQR, 90% quantile, outliers (+/- 1.5*IQR)
        # -- -- skewness and kurtosis

        # initial data
        ohlc_data = p_data.copy()

        # interquantile range
        def f_iqr(param_data):
            q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
            q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
            return  q1 - q3
        
        # outliers function
        def f_out(param_data):
            # param_data = ohlc_data['co']
            inf = param_data - 1.5*f_iqr(param_data)
            sup = param_data + 1.5*f_iqr(param_data)
            return [inf, sup]

        # data calculations
        ohlc_data['co'] = round((ohlc_data['close'] - ohlc_data['open'])*p_mult, 2)
        ohlc_data['hl'] = round((ohlc_data['high'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ol'] = round((ohlc_data['open'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ho'] = round((ohlc_data['high'] - ohlc_data['open'])*p_mult, 2)

        mins = [min(ohlc_data['co']), min(ohlc_data['hl']), 
                min(ohlc_data['ol']), min(ohlc_data['ho'])]

        maxs = [max(ohlc_data['co']), max(ohlc_data['hl']), 
                max(ohlc_data['ol']), max(ohlc_data['ho'])]

        means = [np.mean(ohlc_data['co']), np.mean(ohlc_data['hl']), 
                 np.mean(ohlc_data['ol']), np.mean(ohlc_data['ho'])]

        medians = [np.median(ohlc_data['co']), np.median(ohlc_data['hl']), 
                   np.median(ohlc_data['ol']), np.median(ohlc_data['ho'])]

        sds = [np.std(ohlc_data['co']), np.std(ohlc_data['hl']), 
               np.std(ohlc_data['ol']), np.std(ohlc_data['ho'])]

        iqr = [f_iqr(ohlc_data['co']), f_iqr(ohlc_data['hl']), 
               f_iqr(ohlc_data['ol']), f_iqr(ohlc_data['ho'])]
        
        q90 = [np.percentile(ohlc_data['co'], 90, interpolation = 'midpoint'),
               np.percentile(ohlc_data['hl'], 90, interpolation = 'midpoint'),
               np.percentile(ohlc_data['ol'], 90, interpolation = 'midpoint'),
               np.percentile(ohlc_data['ho'], 90, interpolation = 'midpoint')]
        
        skew = [m_skew(ohlc_data['co']), m_skew(ohlc_data['hl']),
                m_skew(ohlc_data['ol']), m_skew(ohlc_data['ho'])]

        kurt = [m_kurtosis(ohlc_data['co']), m_kurtosis(ohlc_data['hl']),
                m_kurtosis(ohlc_data['ol']), m_kurtosis(ohlc_data['ho'])]

        # final data
        profile = pd.DataFrame({'min': mins, 'max': maxs, 'mean': means, 'median': medians,
                                'sd': sds, 'iqr': iqr, 'q_90': q90, 'skew': skew, 'kurt': kurt}).T

        profile.columns = ['co', 'hl', 'ol', 'ho']

        return np.round(profile, 2)

    # -- TIMESERIES PROFILING -- #
    elif p_type == 'ts':

        # initial data
        ts_data = p_data.copy()

        # interquantile range
        def f_iqr(param_data):
            q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
            q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
            return  q1 - q3
        
        # outliers function
        def f_out(param_data):
            # param_data = ohlc_data['co']
            inf = param_data - 1.5*f_iqr(param_data)
            sup = param_data + 1.5*f_iqr(param_data)
            return [inf, sup]
        
        ts_des = ts_data.describe(percentiles=[0.25, 0.50, 0.75, 0.90])

        # skews = pd.DataFrame(m_skew(ts_data)).T
        # skews.columns = list(ts_data.columns)
        # ts_data = ts_data.append(skews, ignore_index=True)

        # kurts = pd.DataFrame(m_kurtosis(ts_data)).T
        # kurts.columns = list(ts_data.columns)
        # ts_data = p_data.append(kurts, ignore_index=True)
        
        # p_data.iloc[:, 1]

        # negative_out = []
        # positive_out = []
        
        # rows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis',
                # 'negative_out', 'positive_out']

        # -- missing data (granularity vs calendar if data is based on timestamp labeling)
        # -- For every column in the data frame
        # -- -- min, max, mean, median, sd, IQR, 90% quantile, outliers (+/- 1.5*IQR)
        
        # tsc_iqr = q1 - q3
        # tsc_qr9 = np.percentile(column_data, 90, interpolation = 'midpoint') 
        # out_lims = [q1 - 1.5*tsc_iqr, q3 + 1.5*tsc_iqr]
        
        # outliers = column_data.index(list(column_data) <= out_lims[0] or list(column_data) >= out_lims[1])

        # -- -- skewness and kurtosis

        return ts_des

    else:
        print('error: Type of data not correctly specified')
        return 1


# -- ---------------------------------------------------- DATA PROCESSING: Metrics for Model Performance -- # 
# -- ---------------------------------------------------- ---------------------------------------------- -- #

def model_metrics(p_model, p_data):
    """
    
    Parameters
    ----------
    p_model: str
        string with the name of the model
    
    p_data: dict
        With x_train, x_test, y_train, y_test keys of its respective pd.DataFrames
   
    Returns
    -------
    r_model_metrics

    References
    ----------


    """

    # fitted train values
    p_y_train_d = p_model.predict(p_data['x_train'])
    p_y_result_train = pd.DataFrame({'y_train': p_data['y_train'], 'y_train_pred': p_y_train_d})
    # Confussion matrix
    cm_train = confusion_matrix(p_y_result_train['y_train'], p_y_result_train['y_train_pred'])
    # Probabilities of class in train data
    probs_train = p_model.predict_proba(p_data['x_train'])
    # in case of a nan, replace it with zero (to prevent errors)
    probs_train = np.nan_to_num(probs_train)

    # Accuracy rate
    acc_train = round(accuracy_score(list(p_data['y_train']), p_y_train_d), 4)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_train, tpr_train, thresholds = roc_curve(list(p_data['y_train']), probs_train[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_train = round(roc_auc_score(list(p_data['y_train']), probs_train[:, 1]), 4)

    # fitted test values
    p_y_test_d = p_model.predict(p_data['x_test'])
    p_y_result_test = pd.DataFrame({'y_test': p_data['y_test'], 'y_test_pred': p_y_test_d})
    cm_test = confusion_matrix(p_y_result_test['y_test'], p_y_result_test['y_test_pred'])
    # Probabilities of class in test data
    probs_test = p_model.predict_proba(p_data['x_test'])
    # in case of a nan, replace it with zero (to prevent errors)
    probs_test = np.nan_to_num(probs_test)

    # Accuracy rate
    acc_test = round(accuracy_score(list(p_data['y_test']), p_y_test_d), 4)
    # False Positive Rate, True Positive Rate, Thresholds
    fpr_test, tpr_test, thresholds_test = roc_curve(list(p_data['y_test']), probs_test[:, 1], pos_label=1)
    # Area Under the Curve (ROC) for train data
    auc_test = round(roc_auc_score(list(p_data['y_test']), probs_test[:, 1]), 4)

     # Return the result of the model
    r_model_metrics = {'results': {'data': {'train': p_y_result_train, 'test': p_y_result_test},
                                   'matrix': {'train': cm_train, 'test': cm_test}},
                       'model': p_model, 
                       'metrics': {'train': {'acc': acc_train, 'tpr': tpr_train, 'fpr': fpr_train,
                                      'probs': probs_train, 'auc': auc_train},
                                   'test': {'acc': acc_test, 'tpr': tpr_test, 'fpr': fpr_test,
                                            'probs': probs_test, 'auc': auc_test}}}

    return r_model_metrics


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
                                  max_iter=1e6, fit_intercept=False, random_state=123)

    # model fit
    en_model.fit(x_train, y_train)

   # performance metrics of the model
    metrics_en_model = model_metrics(p_model=en_model, p_data={'x_train': x_train, 'y_train': y_train,
                                                               'x_test': x_test, 'y_test': y_test})

    return metrics_en_model


# --------------------------------------------------------- MODEL: Least Squares Support Vector Machines -- #
# --------------------------------------------------------------------------------------------------------- #

def l1_svm(p_data, p_params):
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
                kernel de L1_SVM
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

                    shrinking=True, probability=True, tol=1e-3, cache_size=5000,
                    class_weight=None, verbose=False, max_iter=1e6, decision_function_shape='ovr',
                    break_ties=False, random_state=123)

    # model fit
    svm_model.fit(x_train, y_train)

    # performance metrics of the model
    metrics_svm_model = model_metrics(p_model=svm_model, p_data={'x_train': x_train, 'y_train': y_train,
                                                                 'x_test': x_test, 'y_test': y_test})

    return metrics_svm_model


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
        learning_rate_init: int

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
    # hidden_layer_sizes, activation, solver, alpha, 

    # learning_rate, batch_size, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose,
    # warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction

    # the batch size will be 50% of the training data length or 75
    batch = max(25, len(x_train)//8)
    # print(batch)

    # model function
    mlp_model = MLPClassifier(hidden_layer_sizes=p_params['hidden_layers'],
                              activation=p_params['activation'], alpha=p_params['alpha'],
                              learning_rate_init=p_params['learning_rate_init'],

                              learning_rate='adaptive',
                              batch_size=batch, solver='sgd', power_t=0.5, max_iter=10000, shuffle=False,
                              random_state=123, tol=1e-7, verbose=False, warm_start=True, momentum=0.9,
                              nesterovs_momentum=True, early_stopping=True, validation_fraction=0.2,
                              n_iter_no_change=100)

    # model fit
    mlp_model.fit(x_train, y_train)

    # performance metrics of the model
    metrics_mlp_model = model_metrics(p_model=mlp_model, p_data={'x_train': x_train, 'y_train': y_train,
                                                                 'x_test': x_test, 'y_test': y_test})

    return metrics_mlp_model


# --------------------------------------------------------------- FUNCTION: Genetic Algorithm Evaluation -- #
# --------------------------------------------------------------- ----------------------------------------- #

def genetic_algo_evaluate(p_individual, p_data, p_model, p_fit_type):
    """
    To evaluate an individual used in the genetic optimization process

    Parameters
    ----------

    p_model: str
        with the model name: 'logistic-elasticnet', 'l1-svm', 'ann-mlp'

    p_fit_type: str
        type of fitness metric for the optimization process:
        'train': the train AUC is used
        'test': the test AUC is used
        'simple': a simple average is calculated between train and test AUC
        'weighted': a weighted average is calculated between train (80%) and test (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and test (80%) AUC

    """

    optimization_params = dt.models
    model_params = list(optimization_params[p_model]['params'].keys())

    # dummy initialization
    model = {}

    # chromosome construction
    chromosome = {model_params[param]: p_individual[param] for param in range(0, len(model_params))}

    if p_model == 'logistic-elasticnet':
        # model results
        model = logistic_net(p_data=p_data, p_params=chromosome)
        
    elif p_model == 'l1-svm':
        # model results
        model = l1_svm(p_data=p_data, p_params=chromosome)

    elif p_model == 'ann-mlp':
      # model results  
        model = ann_mlp(p_data=p_data, p_params=chromosome)
   
    else:
        print('error en genetic_algo_evaluate')

    # get the AUC of the selected model
    model_train_auc = model['metrics']['train']['auc'].copy()
    model_test_auc = model['metrics']['test']['auc'].copy()
       
    # -- type of fitness metric for the evaluation of the genetic individual -- #
        
    # train AUC
    if p_fit_type == 'train':
        return round(model_train_auc, 4)

    elif p_fit_type == 'test':
        return round(model_test_auc, 4)
    
    # simple average of AUC in sample and out of sample
    elif p_fit_type == 'simple':
        return round((model_train_auc + model_test_auc)/2, 4)

    # weighted average of AUC in sample and out of sample
    elif p_fit_type == 'weighted':
        return round((model_train_auc*0.80 + model_test_auc*0.20)/2, 4)

    # inversely weighted average of AUC in sample and out of sample
    elif p_fit_type == 'inv-weighted':
        return round((model_train_auc*0.20 + model_test_auc*0.80)/2, 4)

    else:
        print('error in type of model fitness metric')
        return 'error'


# -------------------------------------------------------------------------- FUNCTION: Genetic Algorithm -- #
# ------------------------------------------------------- ------------------------------------------------- #

def genetic_algo_optimization(p_data, p_model, p_opt_params, p_fit_type):
    """
    El uso de algoritmos geneticos para optimizacion de hiperparametros de varios modelos

    Parameters
    ----------
    p_data: pd.DataFrame
        data frame con datos del m_fold
    
    p_model: dict
        'label' con etiqueta del modelo, 'params' llaves con parametros y listas de sus valores a optimizar
    
    p_opt_params: dict
        with optimization parameters from data.py
    
    p_fit_type: str
    type of fitness metric for the optimization process:
        'train': the train AUC is used
        'test': the test AUC is used
        'simple': a simple average is calculated between train and test AUC
        'weighted': a weighted average is calculated between train (80%) and test (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and test (80%) AUC

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

        # ------------------------------------------------ Evaluate Logistic Regression with Elastic Net -- #
        def evaluate_en(eva_individual):
            # the return of the function has to be always a tupple, thus the inclusion of the ',' at the end

            model_fit = genetic_algo_evaluate(p_individual=eva_individual,
                                              p_data=p_data, p_model='logistic-elasticnet',
                                              p_fit_type=p_fit_type)

            return model_fit,

        toolbox_en.register("mate", tools.cxOnePoint)
        toolbox_en.register("mutate", mutate_en)
        toolbox_en.register("select", tools.selTournament, tournsize=p_opt_params['tournament'])
        toolbox_en.register("evaluate", evaluate_en)

        en_pop = toolbox_en.population(n=p_opt_params['population'])
        en_hof = tools.HallOfFame(p_opt_params['halloffame'])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algorithm Implementation
        en_pop, en_log = algorithms.eaSimple(population=en_pop, toolbox=toolbox_en, stats=stats,
                                             cxpb=p_opt_params['crossover'], mutpb=p_opt_params['mutation'],
                                             ngen=p_opt_params['generations'], halloffame=en_hof, verbose=True)

        # transform the deap objects into list so it can be serialized and stored with pickle
        en_pop = [list(pop) for pop in list(en_pop)]
        # r_log = [list(log) for log in list(en_log)]
        en_hof = [list(hof) for hof in list(en_hof)]

        return {'population': en_pop, 'logs': en_log, 'hof': en_hof}

    # -- --------------------------------------------------------- Least Squares Support Vector Machines -- #
    # ----------------------------------------------------------------------------------------------------- #

    elif p_model['label'] == 'l1-svm':

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
        def evaluate_svm(eva_individual):

            model_fit = genetic_algo_evaluate(p_individual=eva_individual,
                                                p_data=p_data, p_model='l1-svm',
                                                p_fit_type=p_fit_type)

            return model_fit,

        toolbox_svm.register("mate", tools.cxOnePoint)
        toolbox_svm.register("mutate", mutate_svm)
        toolbox_svm.register("select", tools.selTournament, tournsize=p_opt_params['tournament'])
        toolbox_svm.register("evaluate", evaluate_svm)

        svm_pop = toolbox_svm.population(n=p_opt_params['population'])
        svm_hof = tools.HallOfFame(p_opt_params['halloffame'])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Genetic Algortihm implementation
        svm_pop, svm_log = algorithms.eaSimple(population=svm_pop, toolbox=toolbox_svm, stats=stats,
                                               cxpb=p_opt_params['crossover'], mutpb=p_opt_params['mutation'],
                                               ngen=p_opt_params['generations'], halloffame=svm_hof, verbose=True)

        # transform the deap objects into list so it can be serialized and stored with pickle
        svm_pop = [list(pop) for pop in list(svm_pop)]
        # svm_log = [list(log) for log in list(svm_log)]
        svm_hof = [list(hof) for hof in list(svm_hof)]

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
        toolbox_mlp.register("attr_learning_rate_init",
                             random.choice, p_model['params']['learning_rate_init'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_mlp.register("Individual_mlp", tools.initCycle, creator.Individual_mlp,
                             (toolbox_mlp.attr_hidden_layers,
                              toolbox_mlp.attr_activation,
                              toolbox_mlp.attr_alpha,
                              toolbox_mlp.attr_learning_rate_init), n=1)

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
                individual[3] = random.choice(p_model['params']['learning_rate_init'])
            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_mlp(eva_individual):

            model_fit = genetic_algo_evaluate(p_individual=eva_individual,
                                                p_data=p_data, p_model='ann-mlp',
                                                p_fit_type=p_fit_type)

            return model_fit,

        toolbox_mlp.register("mate", tools.cxOnePoint)
        toolbox_mlp.register("mutate", mutate_mlp)
        toolbox_mlp.register("select", tools.selTournament, tournsize=p_opt_params['tournament'])
        toolbox_mlp.register("evaluate", evaluate_mlp)

        mlp_pop = toolbox_mlp.population(n=p_opt_params['population'])
        mlp_hof = tools.HallOfFame(p_opt_params['halloffame'])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # ga algorithjm
        mlp_pop, mlp_log = algorithms.eaSimple(population=mlp_pop, toolbox=toolbox_mlp, stats=stats,
                                               cxpb=p_opt_params['crossover'], mutpb=p_opt_params['mutation'],
                                               ngen=p_opt_params['generations'], halloffame=mlp_hof, verbose=True)

        # transform the deap objects into list so it can be serialized and stored with pickle
        mlp_pop = [list(pop) for pop in list(mlp_pop)]
        # mlp_log = [list(log) for log in list(mlp_log)]
        mlp_hof = [list(hof) for hof in list(mlp_hof)]

        return {'population': mlp_pop, 'logs': mlp_log, 'hof': mlp_hof}

    return 'error, sin modelo seleccionado'


# -------------------------------------------------------------------------- Model Evaluations by period -- #
# --------------------------------------------------------------------------------------------------------- #

def model_evaluation(p_features, p_optim_data, p_model):

    if p_model == 'logistic-elasticnet':
        parameters = {'ratio': p_optim_data[0], 'c': p_optim_data[1]}

        return logistic_net(p_data=p_features, p_params=parameters)

    elif p_model == 'l1-svm':
        parameters = {'c': p_optim_data[0], 'kernel': p_optim_data[1], 'gamma': p_optim_data[2]}

        return l1_svm(p_data=p_features, p_params=parameters)

    elif p_model == 'ann-mlp':
        parameters = {'hidden_layers': p_optim_data[0], 'activation': p_optim_data[1],
                      'alpha': p_optim_data[2], 'learning_rate_init': p_optim_data[3]}

        return ann_mlp(p_data=p_features, p_params=parameters)


# ----------------------------------------------------------------------------- Parallel Loggin Function -- #
# --------------------------------------------------------------------------------------------------------- #

def setup_logger(name_logfile, path_logfile):
                      
        logger = logging.getLogger(name_logfile)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')
        fileHandler = logging.FileHandler(path_logfile, mode='w')
        fileHandler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)

        return logger

# -------------------------------------------------------------------------- Model Evaluations by period -- #
# --------------------------------------------------------------------------------------------------------- #

def fold_process(p_data_folds, p_models, p_fit_type, p_transform, p_scaling):
    """
    Global evaluations for specified data folds for specified models

    Parameters
    ----------
    p_data_folds: dict
        with all the folds of data
        p_data_folds = folds

    p_models: list
        with the name of the models

    p_fit_type: str
        type of fitness metric for the optimization process:
            'train': the train AUC is used
            'test': the test AUC is used
            'simple': a simple average is calculated between train and test AUC
            'weighted': a weighted average is calculated between train (80%) and test (20%) AUC
            'inv-weighted': an inverse weighted average is calculated between train (20%) and test (80%) AUC
    
    p_transform: str
        type of transformation to perform to the data
        'scale': x/max(x)
        'standard': [x - mean(x)] / sd(x)
        'robust': x - median(x) / iqr(x)

        p_transform = 'scale'

    p_scaling: str
        Order to perform the data transformation
        'pre-features': previous to the feature engineering (features will not be transformed)
        'post-features': after the feature engineering (features will be transformed)       

        p_scaling = 'pre-features'

    Returns
    -------
    memory_palace: dict

    """
    
    # main data structure for calculations
    memory_palace = {j: {i: {'e_hof': [], 'p_hof': {}, 'time': [], 'features': {}}
                     for i in list(dt.models.keys())} for j in p_data_folds}

    iteration = list(p_data_folds.keys())[0][0]

    if iteration == 'q':
        msg = 'quarter'
    elif iteration == 's':
        msg = 'semester'
    elif iteration == 'y':
        msg = 'year'
    elif iteration == 'b':
        msg = 'bi-year'
    elif iteration == 'h':
        msg = '80-20'
    else:
        msg = 'na'

    # Construct the file name for the logfile
    name_log = iteration + '_' + p_fit_type + '_' + p_transform + '_' + p_scaling

    # Create logfile
    logger = setup_logger('log%s' %name_log, 'files/logs/%s.txt' %name_log)

    logger.debug('                                                            ')
    logger.debug(' ********************************************************************************')
    logger.debug('                           T-FOLD SIZE: ' + msg + '                              ')
    logger.debug(' ********************************************************************************\n')

    # cycle to iterate all periods
    for period in list(p_data_folds.keys()):
        # period = list(p_data_folds.keys())[0]

        # time measurement
        init = datetime.now()

        logger.debug('|| ---------------------- ||')
        logger.debug('|| period: ' + period)
        logger.debug('|| ---------------------- ||\n')
        
        logger.debug('------------------- Feature Engineering on the Current Fold ---------------------')
        logger.debug('------------------- --------------------------------------- ---------------------')
        
        # Feature metrics for ORIGINAL DATA: OHLCV
        dt_metrics = data_profile(p_data=p_data_folds[period].copy(), p_type='ohlc', p_mult=10000)

        # dummy initialization
        data_folds = {}
        m_features = {}

        # -- DATA SCALING OPTIONS -- #
        
        # OPTION 1: Scaling original data before feature engineering
        if p_scaling == 'pre-features':
            
            # Original data
            data_folds = data_scaler(p_data=p_data_folds[period].copy(), p_trans=p_transform)
        
            # Feature engineering (Autoregressive, Hadamard)
            linear_data = linear_features(p_data=data_folds, p_memory=7)
            
            # Symbolic features generation with genetic programming
            m_features = genetic_programed_features(p_data=linear_data)

            # print to have it in the log
            df_log = pd.DataFrame(m_features['sym_data']['details'])
            df_log.columns = ['gen', 'avg_len', 'avg_fit', 'best_len', 'best_fit', 'best_oob', 'gen_time']
            logger.debug('\n\n{}\n'.format(df_log))

        # OPTION 2: Scaling original data after feature engineering
        elif p_scaling == 'post-features':
            
            # Original data
            data_folds = p_data_folds[period].copy()

            # Feature engineering (Autoregressive, Hadamard)
            linear_data = linear_features(p_data=data_folds, p_memory=7)
            
            # Symbolic features generation with genetic programming
            m_features = genetic_programed_features(p_data=linear_data)

            # print to have it in the log
            df_log_2 = pd.DataFrame(m_features['sym_data']['details'])
            df_log_2.columns = ['gen', 'avg_len', 'avg_fit', 'best_len', 'best_fit', 'best_oob', 'gen_time']
            logger.debug('\n\n{}\n'.format(df_log_2))

            # Data scaling in train
            m_features['model_data']['train_x'] = data_scaler(p_data=m_features['model_data']['train_x'],
            p_trans=p_transform)

            # Data scaling in test
            m_features['model_data']['test_x'] = data_scaler(p_data=m_features['model_data']['test_x'],
            p_trans=p_transform)
        
        else:
            print('error in p_scaling value')

        # Feature metrics for FEATURES and TARGET variable
        ft_metrics = {'train_x': data_profile(p_data=m_features['model_data']['train_x'],
                                              p_type='ts', p_mult=10000),
                      'test_x': data_profile(p_data=m_features['model_data']['test_x'],
                                             p_type='ts', p_mult=10000),
                      'train_y': data_profile(p_data=m_features['model_data']['train_y'],
                                              p_type='ts', p_mult=10000),
                      'test_y': data_profile(p_data=m_features['model_data']['train_y'],
                                             p_type='ts', p_mult=10000)}
        
        # save calculated metrics
        memory_palace[period]['metrics'] = {'data_metrics': dt_metrics, 'feature_metrics': ft_metrics}

        logger.debug('----------------- Hyperparameter Optimization on the Current Fold ---------------')
        logger.debug('------------------- --------------------------------------- ---------------------\n')

        logger.debug('---- Optimization Fitness: ' + p_fit_type)
        logger.debug('---- Data Scaling Order: ' + p_scaling)
        logger.debug('---- Data Transformation: ' + p_transform + '\n')

        logger.info("Feature Engineering in Fold done in = " + str(datetime.now() - init) + '\n')

        # Save data of features used in the evaluation in memory_palace (only once per fold)
        memory_palace[period]['features'] = m_features['model_data']

        # Save equations of features used in the evaluation in memory_palace (only once per fold)
        memory_palace[period]['sym_features'] = m_features['sym_data']

        # cycle to iterate all models
        for model in p_models:

            # Optimization
            
            logger.debug('---------------------------------------------------------------------------------')
            logger.debug('model: ' + model)
            logger.debug('---------------------------------------------------------------------------------\n')

            # -- model optimization and evaluation for every element in the Hall of Fame for every period
            # optimization process
            hof_model = genetic_algo_optimization(p_data=m_features['model_data'], p_model=dt.models[model],
                                                  p_opt_params=dt.optimization_params, p_fit_type=p_fit_type)

            # log the result of genetic algorithm
            logger.info('\n\n{}\n'.format(hof_model['logs']))

            # evaluation process
            for i in range(0, len(list(hof_model['hof']))):
                hof_eval = model_evaluation(p_features=m_features['model_data'], p_model=model,
                                            p_optim_data=hof_model['hof'][i])

                # save evaluation in memory_palace
                memory_palace[period][model]['e_hof'].append(hof_eval)

            # save the parameters from optimization process
            memory_palace[period][model]['p_hof'] = hof_model

            # time measurement
            memory_palace[period][model]['time'] = datetime.now() - init

            logger.info("Model Optimization in Fold done in = " + str(datetime.now() - init) + '\n')
    
    # -- ------------------------------------------------------------------------------- DATA BACKUP -- #
    # -- ------------------------------------------------------------------------------- ----------- -- #

    # File name to save the data
    route = 'files/pickle_rick/'
    file_name = route + period[0] + '_' + p_fit_type + '_' + p_scaling + '_' + p_transform + '.dat'

    # objects to be saved
    pickle_rick = {'data': dt.ohlc_data, 't_folds': period, 'memory_palace': memory_palace}

    # pickle format function
    dt.data_save_load(p_data_objects=pickle_rick, p_data_file=file_name, p_data_action='save')

    return memory_palace


# ------------------------------------------------------------------------------ Model Global Evaluation -- #
# --------------------------------------------------------------------------------------------------------- #

def global_evaluation(p_memory_palace, p_data, p_cases, p_model, p_case):
    """
    Evaluation of models with global data and features for particular selected cases of parameters

    Parameters
    ----------
    p_data: dict
        The data to use in the model evaluation

    p_model: dict
        With information of the model that is going to be tested

    p_features: dict
        With information of the features to build to use in the model that is going to be tested

    Returns
    -------
    global_auc_cases: dict
        with the evaluations

    """

    # p_memory_palace = memory_palance
    # p_data = data
    # p_cases = auc_cases
    # p_case = fold_case
    # p_model = fold_model

    # get period of ocurring case
    fold_period = p_cases[p_model][p_case]['period']

    # model parameters
    fold_mod_params = p_cases[p_model]['hof_metrics']['data'][fold_period][p_case + '_params']

    # Get all the linear features 
    linear_data = linear_features(p_data=p_data.copy(), p_memory=7)
    y_target = linear_data['co_d'].copy()
    linear_data = linear_data.drop(['co_d'], axis=1, inplace=False)

    # use equations to generate symbolic features
    global_features = p_memory_palace[p_model][fold_period]['sym_features']['model'].transform(linear_data)
    global_features = pd.DataFrame(np.hstack((linear_data, global_features)))

    # data division    
    xtrain, xtest, ytrain, ytest = train_test_split(global_features, y_target, test_size=0.01, shuffle=False)

    global_features = {}
    global_features['train_x'] = xtrain
    global_features['train_y'] = ytrain
    global_features['test_x'] = xtest
    global_features['test_y'] = ytest

    if p_model == 'logistic-elasticnet':
        parameters = {'ratio': fold_mod_params[0], 'c': fold_mod_params[1]}

        return {'global_data': global_features, 'global_parameters': parameters,
                'model': logistic_net(p_data=global_features, p_params=parameters)}

    elif p_model == 'l1-svm':
        parameters = {'c': fold_mod_params[0], 'kernel': fold_mod_params[1], 'gamma': fold_mod_params[2]}

        return {'global_data': global_features, 'global_parameters': parameters,
                'model': l1_svm(p_data=global_features, p_params=parameters)}

    elif p_model == 'ann-mlp':
        parameters = {'hidden_layers': fold_mod_params[0], 'activation': fold_mod_params[1],
                      'alpha': fold_mod_params[2], 'learning_rate_init': fold_mod_params[3]}

        return {'global_data': global_features, 'global_parameters': parameters,
                'model': ann_mlp(p_data=global_features, p_params=parameters)}


# -------------------------------------------------------------------------- Model AUC Min and Max Cases -- #
# --------------------------------------------------------------------------------------------------------- #

def model_auc(p_models, p_global_cases, p_data_folds, p_cases_type):
    """
    AUC min and max cases for the models

    Parameters
    ----------
    p_models: list
        with the models name

    p_global_cases: dict
        With all the info for the global cases

    p_data_folds: dict
        with all the historical data info in folds

    p_cases_type: str
        'train': the train AUC is used
        'simple': a simple average is calculated between train and test AUC
        'weighted': a weighted average is calculated between train (80%) and test (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and test (80%) AUC

    Returns
    -------
    auc_cases:dict
        with all the info of the min and the max case for every model

    """

    # diccionario para almacenar resultados de busqueda
    auc_cases = {j: {i: {'data': {}, 'period':''}
                     for i in ['auc_min', 'auc_max', 'hof_metrics']} for j in p_models}

    # catch mode on model params
    auc_mode = {}

    # search for every model
    for model in p_models:
        # dummy initializations
        auc_min = 1
        auc_max = 0
        auc_max_params = {}
        auc_min_params = {}
        auc_mode[model] = {}
        
        # search for every fold
        for period in p_data_folds:
            auc_cases[model]['hof_metrics']['data'][period] = {}
            # Dummy initialization
            auc_s = []
            
            # -- For debugging -- #
            # p_global_cases = memory_palace
            # model = 'logistic-elasticnet'
            # period = 'b_y_0'

            # -- Case 0 (MODE INDIVIDUAL)
            # get the number of repeated individuals in the whole HoF
            for p in p_global_cases[model][period]['p_hof']['hof']:
                if tuple(p) in list(auc_mode[model].keys()):
                    auc_mode[model][tuple(p)] += 1
                else:
                    auc_mode[model][tuple(p)] = 0

            # search for every individual in hall of fame
            for i in range(0, len(p_global_cases[model][period]['e_hof'])):
                
                # initialize value in 0 (in case of error)
                c_auc = 0
                
                # -- Calculate fitness metric 

                # using only train data
                if p_cases_type == 'train':
                    c_auc = p_global_cases[period][model]['e_hof'][i]['metrics']['train']['auc']

                # using only test data
                elif p_cases_type == 'test':
                    c_auc = p_global_cases[period][model]['e_hof'][i]['metrics']['test']['auc']

                # a simple average with train and test data
                elif p_cases_type == 'simple':
                    c_auc = round((p_global_cases[period][model]['e_hof'][i]['metrics']['train']['auc'] + 
                                   p_global_cases[period][model]['e_hof'][i]['metrics']['test']['auc'])/2, 4)

                # a weighted average with train and test data
                elif p_cases_type == 'weighted':
                    c_auc = round((p_global_cases[period][model]['e_hof'][i]['metrics']['train']['auc']*.8 + 
                                   p_global_cases[period][model]['e_hof'][i]['metrics']['test']['auc']*.2)/2, 4)
                
                # an inversely weighted average with train and test data
                elif p_cases_type == 'inv-weighted':
                    c_auc = round((p_global_cases[period][model]['e_hof'][i]['metrics']['train']['auc']*.2 + 
                                   p_global_cases[period][model]['e_hof'][i]['metrics']['test']['auc']*.8)/2, 4)
                
                # error in parameter input
                else:
                    print('type of auc case is wrong')
                
                # save current auc data for later use
                auc_s.append(c_auc)

                # -- Case 1 (MIN INDIVIDUAL)
                # get the individual of all of the HoF that produced the minimum AUC
                if c_auc < auc_min:
                    auc_min = c_auc
                    auc_cases[model]['auc_min']['data'] = p_global_cases[model][period]['e_hof'][i]
                    auc_cases[model]['auc_min']['period'] = period
                    auc_min_params = p_global_cases[model][period]['p_hof']['hof'][i]

                # -- Case 2 (MAX INDIVIDUAL)
                # get the individual of all of the HoF that produced the maximum AUC
                elif c_auc > auc_max:
                    auc_max = c_auc
                    auc_cases[model]['auc_max']['data'] = p_global_cases[model][period]['e_hof'][i]
                    auc_cases[model]['auc_max']['period'] = period
                    auc_max_params = p_global_cases[model][period]['p_hof']['hof'][i]

            # Get features used for every case, therefore, for min and max AUC cases
            features = p_global_cases[model][period]['features']['train_x']

            # Guardar info por periodo
            auc_cases[model]['hof_metrics']['data'][period]['auc_s'] = auc_s
            auc_cases[model]['hof_metrics']['data'][period]['auc_max'] = auc_max
            auc_cases[model]['hof_metrics']['data'][period]['auc_max_params'] = auc_max_params
            auc_cases[model]['hof_metrics']['data'][period]['auc_min'] = auc_min
            auc_cases[model]['hof_metrics']['data'][period]['auc_min_params'] = auc_min_params
            auc_cases[model]['hof_metrics']['data'][period]['features'] = features
            auc_cases[model]['hof_metrics']['data'][period]['mode'] = auc_mode[model]

    return auc_cases
