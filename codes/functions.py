
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

# Modify environmental variable to suppress console log messages from TensorFlow
import os
from re import T
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from rich import inspect
import random
import warnings
import logging
import pandas as pd
import numpy as np
import data as dt

import h5py
import io
import copy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, log_loss

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, models, regularizers, optimizers

from datetime import datetime
from scipy.stats import kurtosis as m_kurtosis
from scipy.stats import skew as m_skew

from gplearn.genetic import SymbolicTransformer
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# ------------------------------------------------------------------ PROCESSING RESOURCES FOR TENSORFLOW -- #
# ------------------------------------------------------------------ ----------------------------------- -- #

def tf_processing(p_option, p_cores):
    """
    function to stablish which type of processing will be used with tensorflow

    Parameters
    ----------
    
    p_option: str
        The option of process engine to use: 'cpu', 'gpu'

    p_cores: int
        The number of cores to use

    References
    ----------
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/ConfigProto
    https://www.tensorflow.org/tutorials/distribute/parameter_server_training

    """

    if p_option == 'gpu':
        num_GPU = 1
        num_CPU = 1
    elif p_option == 'cpu':
        num_CPU = p_cores
        num_GPU = 0
    else:
        print('error in process, p_option not valid')

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=p_cores,
                                      inter_op_parallelism_threads=p_cores,
                                      allow_soft_placement=True,
                                      device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

    session = tf.compat.v1.Session(config=config)
    K.set_session(session)


# ------------------------------------------------------------------------------------ DATA PRE-SCALLING -- #
# ------------------------------------------------------------------------------------ ----------------- -- #

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

    # hardcopy of the data
    data = p_data.copy()
    
    # list with columns to transform
    lista = data[list(data.columns[1:])]

    if p_trans == 'standard':
        
        # armar objeto de salida
        data[list(data.columns[1:])] = StandardScaler().fit_transform(lista)
        return data

    elif p_trans == 'robust':

        # armar objeto de salida
        data[list(data.columns[1:])] = RobustScaler().fit_transform(lista)
        return data

    elif p_trans == 'scale':

        # armar objeto de salida
        data[list(data.columns[1:])] = MaxAbsScaler().fit_transform(lista)
        return data
    
    else:
        print('error, p_trans not valid')


# ------------------------------------------------------------------------ EMBARGO TECHNIQUE FOR T-FOLDS -- #
# ------------------------------------------------------------------------ -------------------------------- #

def folds_embargo(p_folds, p_mode, p_memory):
    """
    
    Parameters
    ----------
    p_data: dic
        Dictionary in which every key is a T-Fold in pd.DataFrame TOHLCV format

    p_mode: str
        Modality to calculate the embargo number
        'autocorrelation': Performs a PACF test and takes the most significative lag if any
        'memory': an integer that represents the memory in the data
    
    Return
    ------
    same dictionary and pd.DataFrames with resulting from embargo operation

    p_folds = folds.copy()
    """

    embargo_data = p_folds.copy()

    # select keys as periods, all but the first one
    embargo_periods = list(p_folds.keys())[1:]
    embargo_data = pd.concat(embargo_data)
    # embargo_data.reset_index(inplace=True, drop=True)
    # embargo_data.index = list(embargo_data['timestamp'])
    # embargo_data = embargo_data[['open', 'high', 'low', 'close', 'volume']].copy()

    # -- check for criteria to calculate quantity of observations to drop
    for period in embargo_periods:
        # period = embargo_periods[0] 

        # memory output
        if p_mode == 'fix':
            n_embargo = p_memory
        
        # autocorrelation function
        elif p_mode == 'memory':
            pacf_values = []
            for column in embargo_data[['open', 'high', 'low', 'close', 'volume']]:
                data_pacf = sm.tsa.pacf(embargo_data[column].diff()[1:])
                data_acf = sm.tsa.acf(embargo_data[column].diff()[1:])
                pacf_values.append(np.amax(data_acf))
                pacf_values.append(np.amax(data_pacf))

            # calculate autocorrelations
            n_embargo = np.where(pacf_values == np.amax(pacf_values))[0][0] + 1
            
        embargo_indexes = {}
        for period in embargo_periods:
            # period = periods[0]
            embargo_indexes[period] = list(embargo_data['timestamp'][p_folds[period].index[0:n_embargo]])
            p_folds[period] = p_folds[period].drop(p_folds[period].index[0:n_embargo]).copy()
    
    return p_folds, embargo_indexes


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
        '80-20': Hold out method, 80% for training and 20% for valing

    p_embargo : int
        According to bibliography, embargo is the 'memory' kept between datasets of timeseries data,
        this memory is proposed as the "lags" of the prices, i.e. historical observations to drop
        from the T-Fold division. e.g. with daily prices, an embargo of 7 means 7 days must be taken out
        at the beggining of a subdataset that proceeds a previous T-Fold sub dataset
    
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
            # y = sorted(list(years))[0]
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

# ---------------------------------------------------------- FUNCTION: Autoregressive Feature Engieering -- #
# ---------------------------------------------------------- ---------------------------------------------- #

def linear_features(p_data, p_memory):
    """
    autoregressive process for feature engineering

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
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}

    References
    ----------

    """

    # hardcopy of data
    data = p_data.copy()

    # funcion para generar variables autoregresivas
    data_ar = autoregressive_features(p_data=data, p_nmax=p_memory)

    # separacion de variable dependiente
    data_y = data_ar['co_d'].copy()

    # separacion de variables independientes
    data_arf = data_ar.drop(['timestamp', 'co', 'co_d'], axis=1, inplace=False)

    # datos para utilizar en la siguiente etapa
    next_data = pd.concat([data_y.copy(), data_arf.copy()], axis=1)

    # keep the timestamp as index
    next_data.index = data_ar['timestamp'].copy()
  
    return next_data


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

def genetic_programed_features(p_data, p_split):
    """
    El uso de programacion genetica para generar variables independientes simbolicas

    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        
        p_data = m_folds['periodo_1']

    p_split: int
        split in val

        p_split = '0'

    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}

    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming

    """
   
    # separacion de variable dependiente
    datos_y = p_data['co_d'].copy()

    # separacion de variables independientes
    datos_had = p_data.copy().drop(['co_d'], axis=1, inplace=False)

    # --------------------------------------------------------------- ingenieria de variables simbolicas -- #
    # --------------------------------------------------------------- ---------------------------------- -- #

    # Lista de operaciones simbolicas
    sym_data = symbolic_features(p_x=datos_had, p_y=datos_y, p_params=dt.symbolic_params)

    # variables
    datos_sym = sym_data['data'].copy()
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(sym_data['data'].iloc[0, :]))]
    datos_sym.index = datos_had.index

    # datos para utilizar en la siguiente etapa
    datos_modelo = pd.concat([datos_had.copy(), datos_sym.copy()], axis=1)
    model_data = {}

    # if size != 0 then an inner fold division is performed with size*100 % as val and the rest for train
    size = float(p_split)/100
    
    # there is a inner-split in order to have train-val inside fold
    if size != 0:
        
        # automatic data sub-sets division according to inner-split
        xtrain, xval, ytrain, yval = train_test_split(datos_modelo, datos_y, test_size=size, shuffle=False)

        # data organization
        model_data['train_x'] = xtrain.copy()
        model_data['train_y'] = ytrain.copy()
        model_data['val_x'] = xval.copy()
        model_data['val_y'] = yval.copy()

        return {'model_data': model_data, 'sym_data': sym_data}
    
    # No inner-split in the fold, therefore, all data is considered 1 train set and wont have a val set
    else:

        # data organization
        model_data['train_x'] = datos_modelo.copy()
        model_data['train_y'] = datos_y.copy()

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

    # copy of input data
    f_data = p_data.copy()

    # interquantile range
    def f_iqr(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        return  q1 - q3
    
    # outliers function
    def f_out(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        lower_out = len(np.where(param_data < q1 - 1.5*f_iqr(param_data))[0])
        upper_out = len(np.where(param_data > q3 + 1.5*f_iqr(param_data))[0])
        return [lower_out, upper_out]

    # in the case of a binary target variable
    if p_type == 'target':
        # print(type(p_data))
        return p_data.value_counts()

    # -- OHLCV PROFILING -- #
    elif p_type == 'ohlc':

        # -- init and end dates, amount of data, data type, range of values (all values)
        # -- missing data (granularity vs calendar if data is based on timestamp labeling)

        # initial data
        ohlc_data = p_data[['open', 'high', 'low', 'close', 'volume']].copy()

        # data calculations
        ohlc_data['co'] = round((ohlc_data['close'] - ohlc_data['open'])*p_mult, 2)
        ohlc_data['hl'] = round((ohlc_data['high'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ol'] = round((ohlc_data['open'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ho'] = round((ohlc_data['high'] - ohlc_data['open'])*p_mult, 2)

        # original data + co, hl, ol, ho columns
        f_data = ohlc_data.copy()
        
    # basic data description
    data_des = f_data.describe(percentiles=[0.25, 0.50, 0.75, 0.90])

    # add skewness metric
    skews = pd.DataFrame(m_skew(f_data)).T
    skews.columns = list(f_data.columns)
    data_des = data_des.append(skews, ignore_index=False)

    # add kurtosis metric
    kurts = pd.DataFrame(m_kurtosis(f_data)).T
    kurts.columns = list(f_data.columns)
    data_des = data_des.append(kurts, ignore_index=False)
    
    # add outliers count
    outliers = [f_out(param_data=f_data[col]) for col in list(f_data.columns)]
    negative_series = pd.Series([i[0] for i in outliers], index = data_des.columns)
    positive_series = pd.Series([i[1] for i in outliers], index = data_des.columns)
    data_des = data_des.append(negative_series, ignore_index=True)
    data_des = data_des.append(positive_series, ignore_index=True)
    
    # index names
    data_des.index = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'p90',
                      'max', 'skew', 'kurt', 'n_out', 'p_out']

    return np.round(data_des, 2)


# -- ---------------------------------------------------- DATA PROCESSING: Metrics for Model Performance -- # 
# -- ---------------------------------------------------- ---------------------------------- TENSOR FLOW -- #

def tf_model_metrics(p_model, p_model_data, p_history):
    """
    Tensorflow ready model metrics
    
    Parameters
    ----------
    p_model: str
        string with the name of the model
    
    p_data: dict
        With x_train, x_val, y_train, y_val keys of its respective pd.DataFrames
   
    Returns
    -------
    r_model_metrics

    References
    ----------
    https://keras.io/api/metrics/

    """

     # Data keys, could be ['x_train', 'y_train'] or ['x_train', 'y_train', 'x_val', 'y_val']
    data_keys = list(p_model_data.keys())

    if 'train_x' in data_keys and 'train_y' in data_keys:
    
        # Fitted train values
        p_y_train_d = p_model.predict(p_model_data['train_x'])
        p_y_train_d = np.nan_to_num(p_y_train_d)

        # tensorflow output
        # reshape array as a 1d array (list)
        p_y_train_d = p_y_train_d.reshape(len(p_y_train_d),)
        # p_y_train_d[p_y_train_d.reshape(len(p_y_train_d),) > 0.5] = 1
        # p_y_train_d[p_y_train_d.reshape(len(p_y_train_d),) <= 0.5] = 0

        # **** hardcode at least 1 case for each class in order to avoid loss function error
        # p_y_train_d[0] = 1
        # p_y_train_d[1] = 0

        p_y_result_train = pd.DataFrame({'train_y': p_model_data['train_y'], 'train_pred_y': p_y_train_d})
        # Confussion matrix
        cm_train = confusion_matrix(p_y_result_train['train_y'], p_y_result_train['train_pred_y'])

        # Probabilities of class in train data
        probs_train = p_model.predict(p_model_data['train_x']).reshape(len(p_model_data['train_x']),)
        # In case of a nan, replace it with zero (to prevent errors)
        probs_train = np.nan_to_num(probs_train)

        # -- Accuracy rate
        
        # historical info
        acc_h_train = p_history['accuracy']
        # last configuration
        acc_train = acc_h_train[-1]
        # verification
        acc_train_sk = accuracy_score(list(p_model_data['train_y']), p_y_train_d)

        # False Positive Rate, True Positive Rate, Thresholds
        fpr_train, tpr_train, thresholds = roc_curve(list(p_model_data['train_y']), probs_train, pos_label=1)
        # Area Under the Curve (ROC) for train data
        # historical info
        # auc_h_train = p_history['auc']
        # last configuration
        auc_train = roc_auc_score(list(p_model_data['train_y']), probs_train) + 1e-5
        # Logloss (Binary cross-entropy function)
        # `historic`al info
        logloss_h_train = p_history['loss']
        # last configuration
        logloss_train = logloss_h_train[-1] + 1e-5

        if 'val_x' not in data_keys and 'val_y' not in data_keys:
            cm_val = cm_train
            probs_val = probs_train
            acc_val = acc_train
            fpr_val = fpr_train
            tpr_val = tpr_train
            auc_val = auc_train
            logloss_val = logloss_train
            p_y_result_val = p_y_result_train

     # -- val SET ALSO: In the case of the presence of a val set, do calculations accordingly
    if 'val_x' in data_keys and 'val_y' in data_keys:

        # Fitted val values
        p_y_val_d = p_model.predict(p_model_data['val_x'])
        p_y_val_d = np.nan_to_num(p_y_val_d)

        # tensorflow output
        # reshape array as a 1d array (list)
        p_y_val_d = p_y_val_d.reshape(len(p_y_val_d),)
        # trigger for class is 1 if > 0.5
        # p_y_val_d[p_y_val_d.reshape(len(p_y_val_d),) >= 0.5] = 1
        # p_y_val_d[p_y_val_d.reshape(len(p_y_val_d),) < 0.5] = 0

        # **** hardcode at least 1 case for each class in order to avoid loss function error
        # p_y_val_d[0] = 1
        # p_y_val_d[1] = 0
        
        p_y_result_val = pd.DataFrame({'val_y': p_model_data['val_y'], 'val_pred_y': p_y_val_d})
        cm_val = confusion_matrix(p_y_result_val['val_y'], p_y_result_val['val_pred_y'])
        # Probabilities of class in val data
        probs_val = p_model.predict(p_model_data['val_x']).reshape(len(p_model_data['val_x']),)
        # In case of a nan, replace it with zero (to prevent errors)
        probs_val = np.nan_to_num(probs_val)

        # Accuracy rate
        # historical info
        # acc_h_val = p_history['accuracy']
        # last configuration
        # acc_val = acc_h_val[-1]
        acc_val = accuracy_score(list(p_model_data['val_y']), p_y_val_d)

        # False Positive Rate, True Positive Rate, Thresholds
        fpr_val, tpr_val, thresholds = roc_curve(list(p_model_data['val_y']), probs_val, pos_label=1)
        # Area Under the Curve (ROC) for train data
        # historical info
        # auc_h_val = p_history['auc']
        # last configuration
        auc_val = roc_auc_score(list(p_model_data['val_y']), probs_val) + 1e-5

        # Logloss (Binary cross-entropy function)
        # historical info
        logloss_h_val = p_history['loss']
        # last configuration
        logloss_val = logloss_h_val[-1] + 1e-5

        if 'train_x' not in data_keys and 'train_y' not in data_keys:
            cm_train = cm_val
            probs_train = probs_val
            acc_train = acc_val
            fpr_train = fpr_val
            tpr_train = tpr_val
            auc_train = auc_val
            logloss_train = logloss_val
            p_y_result_train = p_y_result_val
       
    # -- ----------------------------------------------------------------------------------------------------

    # calculate relevant metrics
    pro_metrics = {'acc-train': acc_train,
                   'acc-val': acc_val, 
                   'acc-mean': (acc_train + acc_val)/2 + 1e-5, 
                   'acc-diff': abs(acc_train - acc_val), 
                   'acc-weighted': (acc_train*0.80 + acc_val*0.20)/2 + 1e-5,
                   'acc-inv-weighted': (acc_train*0.20 + acc_val*0.80)/2 + 1e-5,

                   'auc-train': auc_train,
                   'auc-val': auc_val,
                   'auc-diff': abs(auc_train - auc_val), 
                   'auc-mean': (auc_train + auc_val)/2 + 1e-5, 
                   'auc-weighted': (auc_train*0.80 + auc_val*0.20)/2 + 1e-5,
                   'auc-inv-weighted': (auc_train*0.20 + auc_val*0.80)/2 + 1e-5,

                   'logloss-train': logloss_train,
                   'logloss-val': logloss_val,
                   'logloss-diff': abs(logloss_train - logloss_val),
                   'logloss-mean': (logloss_train + logloss_val)/2 + 1e-5,
                   'logloss-weighted': (logloss_train*0.80 + logloss_val*0.20)/2 + 1e-5,
                   'logloss-inv-weighted': (logloss_train*0.20 + logloss_val*0.80)/2 + 1e-5}

    # -- -------------------------------------------------------------------------------------------------- #
    # weights = p_model.model.weights
    # layers = p_model.model.layers

    # Return all the results for the model
    r_model_metrics = {'model': p_model.model.to_json(), 'pro-metrics': pro_metrics, 
                       'results': {'data': {'train': p_y_result_train, 'val': p_y_result_val},
                                   'matrix': {'train': cm_train, 'val': cm_val}},
                       'metrics': {'train': {'tpr': tpr_train, 'fpr': fpr_train, 'probs': probs_train},
                                   'val': {'tpr': tpr_val, 'fpr': fpr_val, 'probs': probs_val},
                                   'history': {i: p_history[i] for i in list(p_history.keys())}}}

    return r_model_metrics


# -- ---------------------------------------------------- DATA PROCESSING: Metrics for Model Performance -- # 
# -- -------------------------------------------------------- ---------------------------------- SKLEARN -- #

def sk_model_metrics(p_model, p_model_data):
    """
    Scikit learn ready model metrics

    Parameters
    ----------
    p_model: str
        string with the name of the model
    
    p_data: dict
        With x_train, x_val, y_train, y_val keys of its respective pd.DataFrames
   
    Returns
    -------
    r_model_metrics

    References
    ----------
    https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation

    """

    # Data keys, could be ['x_train', 'y_train'] or ['x_train', 'y_train', 'x_val', 'y_val']
    data_keys = list(p_model_data.keys())

    if 'train_x' in data_keys and 'train_y' in data_keys:

        # Fitted train values
        p_y_train_d = p_model.predict(p_model_data['train_x'])
        p_y_result_train = pd.DataFrame({'train_y': p_model_data['train_y'], 'train_pred_y': p_y_train_d})
        # Confussion matrix
        cm_train = confusion_matrix(p_y_result_train['train_y'], p_y_result_train['train_pred_y'])
        # Probabilities of class in val data
        probs_train = p_model.predict_proba(p_model_data['train_x'])
        # In case of a nan, replace it with zero (to prevent errors)
        probs_train = np.nan_to_num(probs_train)

        # Accuracy rate
        acc_train = accuracy_score(list(p_model_data['train_y']), p_y_train_d)
        # False Positive Rate, True Positive Rate, Thresholds
        fpr_train, tpr_train, thresholds = roc_curve(list(p_model_data['train_y']),
                                                            probs_train[:, 1], pos_label=1)
        # Area Under the Curve (ROC) for train data
        auc_train = roc_auc_score(list(p_model_data['train_y']), probs_train[:, 1]) + 1e-5

        # Logloss (Binary cross-entropy function)
        logloss_train = log_loss(p_model_data['train_y'], p_y_train_d) + 1e-5
    
        if 'val_x' not in data_keys and 'val_y' not in data_keys:
            cm_val = cm_train
            probs_val = probs_train
            acc_val = acc_train
            fpr_val = fpr_train
            tpr_val = tpr_train
            auc_val = auc_train
            logloss_val = logloss_train
            p_y_result_val = p_y_result_train

    # -- val SET ALSO: In the case of the presence of a val set, do calculations accordingly
    if 'val_x' in data_keys and 'val_y' in data_keys:

        # Fitted val values
        p_y_val_d = p_model.predict(p_model_data['val_x'])
        p_y_result_val = pd.DataFrame({'val_y': p_model_data['val_y'], 'val_pred_y': p_y_val_d})
        cm_val = confusion_matrix(p_y_result_val['val_y'], p_y_result_val['val_pred_y'])
        # Probabilities of class in val data
        probs_val = p_model.predict_proba(p_model_data['val_x'])
        # In case of a nan, replace it with zero (to prevent errors)
        probs_val = np.nan_to_num(probs_val)

        # Accuracy rate
        acc_val = accuracy_score(list(p_model_data['val_y']), p_y_val_d)
        # False Positive Rate, True Positive Rate, Thresholds
        fpr_val, tpr_val, thresholds_val = roc_curve(list(p_model_data['val_y']),
                                                             probs_val[:, 1], pos_label=1)
        # Area Under the Curve (ROC) for train data
        auc_val = roc_auc_score(list(p_model_data['val_y']), probs_val[:, 1]) + 1e-5

        # Logloss (Binary cross-entropy function)
        logloss_val = log_loss(p_model_data['val_y'], p_y_val_d) + 1e-5

        if 'train_x' not in data_keys and 'train_y' not in data_keys:
            cm_train = cm_val
            probs_train = probs_val
            acc_train = acc_val
            fpr_train = fpr_val
            tpr_train = tpr_val
            auc_train = auc_val
            logloss_train = logloss_val
            p_y_result_train = p_y_result_val
    
    else:
        print('error in sk_model_metrics, keys in p_model_data not valid')

    # -- ----------------------------------------------------------------------------------------------------

    # calculate relevant metrics
    pro_metrics = {'acc-train': acc_train,
                   'acc-val': acc_val,
                   'acc-diff': abs(acc_train - acc_val), 
                   'acc-mean': (acc_train + acc_val)/2 + 1e-5, 
                   'acc-weighted': (acc_train*0.80 + acc_val*0.20)/2 + 1e-5, 
                   'acc-inv-weighted': (acc_train*0.20 + acc_val*0.80)/2 + 1e-5,

                   'auc-train': auc_train,
                   'auc-val': auc_val,
                   'auc-diff': abs(auc_train - auc_val), 
                   'auc-mean': (auc_train + auc_val)/2 + 1e-5,
                   'auc-weighted': (auc_train*0.80 + auc_val*0.20)/2 + 1e-5,
                   'auc-inv-weighted': (auc_train*0.20 + auc_val*0.80)/2 + 1e-5,

                   'logloss-train': auc_train,
                   'logloss-val': auc_val,
                   'logloss-diff': abs(logloss_train - logloss_val), 
                   'logloss-mean': (logloss_train + logloss_val)/2 + 1e-5,
                   'logloss-weighted': (logloss_train*0.80 + logloss_val*0.20)/2 + 1e-5,
                   'logloss-inv-weighted': (logloss_train*0.20 + logloss_val*0.80)/2 + 1e-5}

    # -- ----------------------------------------------------------------------------------------------------

    # Return all the results for the model
    r_model_metrics = {'model': p_model, 'pro-metrics': pro_metrics, 
                       'results': {'data': {'train': p_y_result_train, 'val': p_y_result_val},
                                   'matrix': {'train': cm_train, 'val': cm_val}},
                       'metrics': {'train': {'tpr': tpr_train, 'fpr': fpr_train, 'probs': probs_train},
                                   'val': {'tpr': tpr_val, 'fpr': fpr_val, 'probs': probs_val}}}

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
    en_model.fit(p_data['train_x'].copy(), p_data['train_y'].copy())

   # performance metrics of the model
    metrics_en_model = sk_model_metrics(p_model=en_model, p_model_data=p_data.copy())

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
    svm_model.fit(p_data['train_x'].copy(), p_data['train_y'].copy())

    # performance metrics of the model
    metrics_svm_model = sk_model_metrics(p_model=svm_model, p_model_data=p_data.copy())

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

    class Modified_KerasClassifier(KerasClassifier):

        """
        TensorFlow Keras API neural network classifier.

        Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
        issue using BytesIO and HDF5 in order to enable pickle dumps.

        Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-522115115
        and https://github.com/keras-team/keras/issues/4274#issuecomment-519226139

        """

        def __getstate__(self):
            state = self.__dict__
            if "model" in state:
                model = state["model"]
                model_hdf5_bio = io.BytesIO()
                with h5py.File(model_hdf5_bio, mode="w") as file:
                    model.save(file)
                state["model"] = model_hdf5_bio
                state_copy = copy.deepcopy(state)
                state["model"] = model
                return state_copy
            else:
                return state

        def __setstate__(self, state):
            if "model" in state:
                model_hdf5_bio = state["model"]
                with h5py.File(model_hdf5_bio, mode="r") as file:
                    state["model"] = tf.keras.models.load_model(file)
            self.__dict__ = state

    # number of inputs in the neural net
    n_inputs = len(list(p_data['train_x'].columns))

    # function to build base model that is used with the KerasClassifier sklearn wrapper for tf.keras
    def build_model(hidden_layers, hidden_neurons, activation, dropout, reg_1, reg_2,
                    learning_rate, momentum, input_shape=[n_inputs]):

        # create base model
        model = models.Sequential()
        # add input layer
        model.add(layers.InputLayer(input_shape=input_shape, name='input_layer'))
        
        # dynamic construction of neural net
        for layer in range(hidden_layers):

            # Add hidden layers with hyperparameters
            model.add(layers.Dense(hidden_neurons, activation=activation,
                                   name='hidden_layer_' + str(layer),
                                   kernel_regularizer=regularizers.l1_l2(reg_2[0], reg_2[1]),
                                   bias_regularizer=regularizers.l1_l2(reg_2[0], reg_2[1]),
                                   activity_regularizer=regularizers.l1_l2(reg_1[0], reg_1[1])))
            # Add dropout layer
            model.add(layers.Dropout(dropout))

        # output layer
        model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
        
        # optimizer for the training
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
        
        # compile model 
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
                      'kullback_leibler_divergence'])

        return model

    # re-mapping of hyperparameters in order to use them inside the wrapper
    param_distribs = {'hidden_layers': p_params['hidden_layers'],
                      'hidden_neurons': p_params['hidden_neurons'],
                      'activation': p_params['activation'], 'dropout': p_params['dropout'],
                      'reg_1': p_params['reg_1'], 'reg_2': p_params['reg_2'],
                      'learning_rate': p_params['learning_rate'], 'momentum': p_params['momentum']}

    # build model with modified wrapper
    keras_class = Modified_KerasClassifier(build_fn=build_model, **param_distribs)
    keras_class.model = build_model(**param_distribs)

    # fit model with corresponding training scheme
    history = keras_class.fit(p_data['train_x'], p_data['train_y'],
                              epochs=200, batch_size=8, verbose=0, shuffle=False,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN(),

                              tf.keras.callbacks.ReduceLROnPlateau(monitor='kullback_leibler_divergence', 
                              factor=0.2, patience=10, verbose=0, mode='min', min_delta=0.01, 
                              cooldown=0, min_lr=0.00001),
                              
                              tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                              min_delta=0.01, patience=20, verbose=0, mode='max', baseline=0.40, restore_best_weights=False)])
   
    # preparation of metrics history data to analyze later
    metrics_history = {str(i): history.history[i]
                       for i in ['loss', 'accuracy', 'kullback_leibler_divergence']}

    # output of the function is sent to the model metrics calculations for tensorflow
    metrics_mlp_model = tf_model_metrics(p_model=keras_class, p_model_data=p_data.copy(),
                                         p_history=metrics_history)

    return metrics_mlp_model


# --------------------------------------------------------------- FUNCTION: Genetic Algorithm Evaluation -- #
# --------------------------------------------------------------- ----------------------------------------- #

def genetic_algo_evaluate(p_individual, p_eval_data, p_model, p_fit_type):
    """
    To evaluate an individual used in the genetic optimization process

    Parameters
    ----------

    p_model: str
        with the model name: 'logistic-elasticnet', 'l1-svm', 'ann-mlp'

    p_fit_type: str
        type of fitness metric for the optimization process:
        'train': the train AUC is used
        'val': the val AUC is used
        'simple': a simple average is calculated between train and val AUC
        'weighted': a weighted average is calculated between train (80%) and val (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and val (80%) AUC

    """

    optimization_params = dt.models
    model_params = list(optimization_params[p_model]['params'].keys())

    # dummy initialization
    model = {}

    # chromosome construction
    chromosome = {model_params[param]: p_individual[param] for param in range(0, len(model_params))}

    if p_model == 'logistic-elasticnet':
        # model results
        model = logistic_net(p_data=p_eval_data, p_params=chromosome)
        # return the already calculated metric
        return model['pro-metrics'][p_fit_type]
        
    elif p_model == 'l1-svm':
        # model results
        model = l1_svm(p_data=p_eval_data, p_params=chromosome)
        # return the already calculated metric
        return model['pro-metrics'][p_fit_type]

    elif p_model == 'ann-mlp':
        # model results  
        model = ann_mlp(p_data=p_eval_data, p_params=chromosome)

        # return the already calculated metric
        return model['pro-metrics'][p_fit_type]
    
    else:
        print('error: genetic_algo_evaluate presented error')
    

# -------------------------------------------------------------------------- FUNCTION: Genetic Algorithm -- #
# ------------------------------------------------------- ------------------------------------------------- #

def genetic_algo_optimization(p_gen_data, p_model, p_opt_params, p_fit_type, p_minmax):
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
        'val': the val AUC is used
        'simple': a simple average is calculated between train and val AUC
        'weighted': a weighted average is calculated between train (80%) and val (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and val (80%) AUC

    p_minmax: str
        To control whether is a minimization or maximization problem
        'min' = minimizes
        'max' = maximizes

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

    # establish weight to specify maximization or minimization in the optimization process
    ga_type = -1.0 if p_minmax == 'min' else 1.0

    # -- ------------------------------------------------------- OLS con regularizacion tipo Elastic Net -- #
    # ----------------------------------------------------------------------------------------------------- #
    if p_model['label'] == 'logistic-elasticnet':

        # borrar clases previas si existen
        try:
            del creator.Fitness_en
            del creator.Individual_en
        except AttributeError:
            pass

        # inicializar ga
        creator.create("Fitness_en", base.Fitness, weights=(ga_type,))
        creator.create("Individual_en", list, fitness=creator.Fitness_en)
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
                                              p_eval_data=p_gen_data, p_model='logistic-elasticnet',
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
            del creator.Fitness_svm
            del creator.Individual_svm
        except AttributeError:
            pass

        # inicializar ga
        creator.create("Fitness_svm", base.Fitness, weights=(ga_type, ))
        creator.create("Individual_svm", list, fitness=creator.Fitness_svm)
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
                                              p_eval_data=p_gen_data, p_model='l1-svm',
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
            del creator.Fitness_mlp
            del creator.Individual_mlp
        except AttributeError:
            pass

        # inicializar ga
        creator.create("Fitness_mlp", base.Fitness, weights=(ga_type, ))
        creator.create("Individual_mlp", list, fitness=creator.Fitness_mlp)
        toolbox_mlp = base.Toolbox()

        # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
        toolbox_mlp.register("attr_hidden_layers", random.choice, p_model['params']['hidden_layers'])
        toolbox_mlp.register("attr_hidden_neurons", random.choice, p_model['params']['hidden_neurons'])
        toolbox_mlp.register("attr_activation", random.choice, p_model['params']['activation'])
        toolbox_mlp.register("attr_reg_1", random.choice, p_model['params']['reg_1'])
        toolbox_mlp.register("attr_reg_2", random.choice, p_model['params']['reg_2'])
        toolbox_mlp.register("attr_dropout", random.choice, p_model['params']['dropout'])
        toolbox_mlp.register("attr_learning_rate", random.choice, p_model['params']['learning_rate'])
        toolbox_mlp.register("attr_momentum", random.choice, p_model['params']['momentum'])

        # This is the order in which genes will be combined to create a chromosome
        toolbox_mlp.register("Individual_mlp", tools.initCycle, creator.Individual_mlp,
                             (toolbox_mlp.attr_hidden_layers,
                              toolbox_mlp.attr_hidden_neurons,
                              toolbox_mlp.attr_activation,
                              toolbox_mlp.attr_reg_1,
                              toolbox_mlp.attr_reg_2,
                              toolbox_mlp.attr_dropout,
                              toolbox_mlp.attr_learning_rate,
                              toolbox_mlp.attr_momentum), n=1)

        # population definition
        toolbox_mlp.register("population", tools.initRepeat, list, toolbox_mlp.Individual_mlp)

        # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
        def mutate_mlp(individual):

            # select which parameter to mutate
            gene = random.randint(0, len(p_model['params']) - 1)

            if gene == 0:
                individual[0] = random.choice(p_model['params']['hidden_layers'])
            elif gene == 1:
                individual[1] = random.choice(p_model['params']['hidden_neurons'])
            elif gene == 2:
                individual[2] = random.choice(p_model['params']['activation'])
            elif gene == 3:
                individual[3] = random.choice(p_model['params']['reg_1'])
            elif gene == 4:
                individual[4] = random.choice(p_model['params']['reg_2'])
            elif gene == 5:
                individual[5] = random.choice(p_model['params']['dropout'])
            elif gene == 6:
                individual[6] = random.choice(p_model['params']['learning_rate'])
            elif gene == 7:
                individual[7] = random.choice(p_model['params']['momentum'])

            return individual,

        # ------------------------------------------------------------ funcion de evaluacion para LS SVM -- #
        def evaluate_mlp(eva_individual):

            model_fit = genetic_algo_evaluate(p_individual=eva_individual,
                                              p_eval_data=p_gen_data, p_model='ann-mlp',
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

    return 'error, invalid model selection'


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
        parameters = {'hidden_layers': p_optim_data[0], 'hidden_neurons': p_optim_data[1],
                      'activation': p_optim_data[2], 'reg_1': p_optim_data[3], 'reg_2': p_optim_data[4],
                      'dropout': p_optim_data[5], 'learning_rate': p_optim_data[6],
                      'momentum': p_optim_data[7]}

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

def fold_process(p_data_folds, p_models, p_embargo, p_inner_split,
                 p_trans_function, p_trans_order, p_fit_type):
    """
    Global evaluations for specified data folds for specified models

    Parameters
    ----------
    p_data_folds: dict
        with all the folds of data
        
        p_data_folds = folds

    p_models: list
        with the name of the models

        p_models = ['logistic-elasticnet', 'l1-svm', 'ann-mlp']
        p_models = ['ann-mlp']

    p_fit_type: str
        type of fitness metric for the optimization process:
            'metrics-train'
            'metric-val'
            'metric-diff'
            'metric-mean'
            'metric-weighted'
            'metric-inv-weighted'
        
        p_fit_type = 'auc-diff'
    
    p_transform: str
        type of transformation to perform to the data
        'scale': x/max(x)
        'standard': [x - mean(x)] / sd(x)
        'robust': x - median(x) / iqr(x)

        p_transform = 'robust'

    p_scaling: str
        Order to perform the data transformation
        'pre-features': previous to the feature engineering (features will not be transformed)
        'post-features': after the feature engineering (features will be transformed)       

        p_scaling = 'post-features'
    
    p_inner_split: float
        Proportion of the val split in the data as a representation of a inner-split in a k-fold process
        if 0 then no val split is performed and all the data is treated as train.

        p_inner_split = '20'

    Returns
    -------
    memory_palace: dict

    """

    # -- Eembargo calculations

    if p_embargo == 'fix':
        # Fixed memory derived from features calculations
        memory = dt.features_params['lags_diffs']
        p_data_folds, embargo_dates = folds_embargo(p_folds=p_data_folds, p_mode='fix', p_memory=memory)
    elif p_embargo == 'memory':
        # Derived from max value from both PACF and ACF functions applied to first difference of ts data
        p_data_folds, embargo_dates = folds_embargo(p_folds=p_data_folds, p_mode='memory', p_memory=None)
    elif p_embargo == 'False':
        # Without embargo
        p_data_folds, embargo_dates = p_data_folds, ['no embargo']
    else: 
        print('error in p_embargo, invalid input')
    
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
    name_log = iteration + '_' + p_fit_type + '_' + p_trans_function + '_' + \
               p_trans_order + '_' + p_inner_split

    # Base route to save file
    route = 'files/logs/'

    # Create logfile
    logger = setup_logger('log%s' %name_log, route + '%s.txt' %name_log)

    logger.debug('                                                            ')
    logger.debug(' ********************************************************************************')
    logger.debug('                           T-FOLD SIZE: ' + msg + '                              ')
    logger.debug(' ********************************************************************************\n')

    # cycle to iterate all periods
    for period in list(p_data_folds.keys()):
        
        # debugging
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
        if p_trans_order == 'pre-features':
            
            # Original data
            data_folds = data_scaler(p_data=p_data_folds[period], p_trans=p_trans_function)
        
            # Feature engineering (Autoregressive)
            linear_data = linear_features(p_data=data_folds, p_memory=dt.features_params['lags_diffs'])
            
            # Symbolic features generation with genetic programming
            m_features = genetic_programed_features(p_data=linear_data, p_split=p_inner_split)

            # print to have it in the log
            df_log = pd.DataFrame(m_features['sym_data']['details'])
            df_log.columns = ['gen', 'avg_len', 'avg_fit', 'best_len', 'best_fit', 'best_oob', 'gen_time']
            logger.debug('\n\n{}\n'.format(df_log))

        # OPTION 2: Scaling original data after feature engineering
        elif p_trans_order == 'post-features':
            
            # Original data
            data_folds = p_data_folds[period].copy()

            # Feature engineering (Autoregressive)
            linear_data = linear_features(p_data=data_folds, p_memory=4)
            
            # Symbolic features generation with genetic programming
            m_features = genetic_programed_features(p_data=linear_data, p_split=p_inner_split)

            # print to have it in the log
            df_log_2 = pd.DataFrame(m_features['sym_data']['details'])
            df_log_2.columns = ['gen', 'avg_len', 'avg_fit', 'best_len', 'best_fit', 'best_oob', 'gen_time']
            logger.debug('\n\n{}\n'.format(df_log_2))

            # Data scaling in train
            for data in list(m_features['model_data'].keys()):
                # debugging
                # data = list(m_features['model_data'].keys())[1]

                # just scale the features, not the target, of inner data-sets
                if data[-1] == 'x':
                    m_features['model_data'][data] = data_scaler(p_data=m_features['model_data'][data],
                                                                 p_trans=p_trans_function)
        
        else:
            print('error: p_scaling value not valid')

        # features data profile 
        ft_metrics = {}
        for data in list(m_features['model_data'].keys()):
            if data[-1] == 'y':
                data_type = 'target' 
            else:
                data_type = 'ts'
            ft_metrics.update({data: data_profile(p_data=m_features['model_data'][data],
                                                  p_type=data_type, p_mult=10000)})
        
        # save calculated metrics
        memory_palace[period]['metrics'] = {'data_metrics': dt_metrics, 'feature_metrics': ft_metrics}

        logger.debug('----------------- Hyperparameter Optimization on the Current Fold ---------------')
        logger.debug('------------------- --------------------------------------- ---------------------\n')

        logger.debug('---- Optimization Fitness: ' + p_fit_type)
        logger.debug('---- Data Scaling Order: ' + p_trans_order)
        logger.debug('---- Data Transformation: ' + p_trans_function)
        logger.debug('---- val set inner-split: ' + p_inner_split + '\n')

        logger.info("Feature Engineering in Fold done in = " + str(datetime.now() - init) + '\n')

        # Save data of features used in the evaluation in memory_palace (only once per fold)
        memory_palace[period]['features'] = m_features['model_data']

        # Save equations of features used in the evaluation in memory_palace (only once per fold)
        memory_palace[period]['sym_features'] = m_features['sym_data']

        # cycle to iterate all models
        for model in p_models:
            # debugging
            # model = p_models[0]

            # Optimization
            
            logger.debug('---------------------------------------------------------------------------------')
            logger.debug('model: ' + model)
            logger.debug('---------------------------------------------------------------------------------\n')

            # -- model optimization and evaluation for every element in the Hall of Fame for every period
               
            # verification of type of objective to optimize
            # default to minimize in order to have an option for logloss
            ob_type = 'min'
            # maximize for auc and acc related metrics
            if p_fit_type[0:3] == 'auc' or p_fit_type[0:3] == 'acc':
                ob_type = 'max'
            # if it is a difference between any of acc, auc and logloss, then choose to minimize
            elif p_fit_type[-4:] == 'diff':
                ob_type = 'min'

            # optimization process NEEDS TO INCLUDE MODEL OBJECT OR WEIGHTS FOR MLP for reproducibility
            hof_model = genetic_algo_optimization(p_gen_data=m_features['model_data'],
                                                  p_model=dt.models[model], p_fit_type=p_fit_type,
                                                  p_opt_params=dt.optimization_params, p_minmax=ob_type)

            # log the result of genetic algorithm
            logger.info('\n\n{}\n'.format(hof_model['logs']))

            # evaluation process
            for i in range(0, len(list(hof_model['hof']))):
                # i = range(0, len(list(hof_model['hof'])))[0]
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

    # Base route to save file
    route = 'files/pickle_rick/'

    # File name to save the data
    file_name = route + period[0] + '_' + p_fit_type + '_' + p_trans_function + '_' + \
                p_trans_order + '_' + p_inner_split + '.dat'

    # (pending) Extract tensorflow model and parameters separately
    
    # objects to be saved
    pickle_rick = {'data': dt.ohlc_data, 't_folds': period, 'embargo_dates': embargo_dates,
                   'memory_palace': memory_palace}

    # print ending message
    logger.debug('---------------------------------------------------------------------------------')
    logger.debug('--- FOLD PROCESS SUCCESSFULLY COMPLETED ---')
    logger.debug('---------------------------------------------------------------------------------\n')
    
    # pickle format function
    dt.data_pickle(p_data_objects=pickle_rick, p_data_file=file_name, p_data_action='save')

    # print ending message
    logger.debug('---------------------------------------------------------------------------------')
    logger.debug('--- FILE SAVED: ' + file_name)
    logger.debug('---------------------------------------------------------------------------------')

    return memory_palace


# ------------------------------------------------------------------------------ Model Global Evaluation -- #
# --------------------------------------------------------------------------------------------------------- #

def global_evaluation(p_case, p_global_data, p_features, p_model):
    """
    Evaluation of models with global data and features for particular selected cases of parameters

    Parameters
    ----------

    p_hof
    p_data
    p_features
    p_model



    Returns
    -------
    global_auc_cases: dict
        with the evaluations

    """

    # entire hof parameters (orderer in descent order by value obtained of the optimization metric)
    hof_params = p_case['p_hof']['hof'].copy()
    hof_models = p_case['e_hof'].copy()

    # -------------------------------------------------------------------------------------- GLOBAL DATA -- #
    
    # get all the linear features 
    g_linear_data = linear_features(p_data=p_global_data, p_memory=dt.features_params['lags_diffs'])
    g_y_target = g_linear_data['co_d'].copy()
    g_linear_data = g_linear_data.drop(['co_d'], axis=1, inplace=False)

    # use equations to generate symbolic features
    g_sym_data = p_features['sym_features']['model'].transform(g_linear_data)
    g_global_data = pd.DataFrame(np.hstack((g_linear_data, g_sym_data)))

    # data format
    global_data = {}
    global_data['val_x'] = g_global_data
    global_data['val_y'] = g_y_target

    # --------------------------------------------------------------- ITERATIVE GLOBAL EVALUATION OF HOF -- #
    
    # store global results in a list (to keep same order as hof)
    global_results = []
    
    # iterative evaluation
    for i in range(0, len(hof_params)):
        individual_params = hof_params[i]
        individual_model = hof_models[i]
    
        if p_model == 'logistic-elasticnet':
            parameters = {'ratio': individual_params[0], 'c': individual_params[1]}
            global_results.append({'global_data': global_data, 'global_parameters': parameters,
                                   'model': sk_model_metrics(p_model=individual_model['model'],
                                                             p_model_data=global_data)})

        elif p_model == 'l1-svm':
            parameters = {'c': individual_params[0], 'kernel': individual_params[1],
                          'gamma': individual_params[2]}
            global_results.append({'global_data': global_data, 'global_parameters': parameters,
                                   'model': sk_model_metrics(p_model=individual_model['model'], 
                                                             p_model_data=global_data)})

        elif p_model == 'ann-mlp':
            parameters = {'hidden_layers': individual_params[0], 'hidden_neurons': individual_params[1],
                          'activation': individual_params[2], 'alpha': individual_params[3],
                          'learning_rate_init': individual_params[4]}
            
            tf_history = individual_model['metrics']['history']
            tf_model = tf.keras.models.model_from_json(individual_model['model'])

            global_results.append({'global_data': global_data, 'global_parameters': parameters,
                                   'model': tf_model_metrics(p_model=tf_model, 
                                                             p_model_data=global_data,
                                                             p_history=tf_history)})

        else: 
            print('error in model selection during (global_evaluation)')

    return global_results


# -------------------------------------------------------------------------- Model AUC Min and Max Cases -- #
# --------------------------------------------------------------------------------------------------------- #

def model_cases(p_models, p_global_cases, p_data_folds, p_cases_type):
    """
    AUC min and max cases for the models

    Parameters
    ----------
    p_models: list
        with the models name
        
        p_models = ['logistic-elasticnet', 'ann-mlp', 'l1-svm']

    p_global_cases: dict
        With all the info for the global cases

        p_global_cases = memory_palace

    p_data_folds: dict
        with all the historical data info in folds

        p_data_folds = folds

    p_cases_type: str
        'train': the train AUC is used
        'simple': a simple average is calculated between train and val AUC
        'weighted': a weighted average is calculated between train (80%) and val (20%) AUC
        'inv-weighted': an inverse weighted average is calculated between train (20%) and val (80%) AUC

        p_cases_type = 'logloss-mean'

    Returns
    -------
    auc_cases:dict
        with all the info of the min and the max case for every model

    """

    # diccionario para almacenar resultados de busqueda
    met_cases = {j: {i: {'data': {}, 'period':''}
                     for i in ['met_min', 'met_max', 'met_mode', 'hof_metrics']} for j in p_models}

    # catch mode on model params
    met_mode = {}

    # search for every model
    for model in p_models:

        # debugging
        # model = p_models[1]

        # dummy initializations
        met_min = float('inf')
        met_max = -float('inf')
        met_max_params = {}
        met_min_params = {}
        met_mode[model] = {}
        
        # search for every fold
        for period in p_data_folds:

            # debugging
            # period = list(p_data_folds.keys())[0]

            if p_cases_type not in list(p_global_cases[period][model]['e_hof'][0]['pro-metrics'].keys()):
                print('error: selected p_cases_type is not in pro-metrics')
                return 0

            # add period key to data info
            met_cases[model]['hof_metrics']['data'][period] = {}
           
            # -- CASE 0 (MODE)
            # get the number of repeated individuals in the whole HoF
            for p in p_global_cases[period][model]['p_hof']['hof']:
                # p = p_global_cases[period][model]['p_hof']['hof'][1]

                # in case of finding a repeated parameter set
                if str(tuple(p)) in list(met_mode[model].keys()):
                    met_mode[model][str(tuple(p))]['repetitions'] += 1
                    met_mode[model][str(tuple(p))]['periods'].append(period)
                
                # no repeated parameter set is found
                else:
                    # base dict to store findings
                    met_mode[model].update({str(tuple(p)): {'params': tuple(p), 'repetitions': 0,
                                                                                'periods': [period]}})

            # Values for all metrics for all evaluated HoF
            hof = [metric['pro-metrics'][p_cases_type] for metric in p_global_cases[period][model]['e_hof']]
            
            # -- CASE 1 (MIN)
            # min metric in evaluated HoF
            fold_i_met_min = hof.index(min(hof))
            fold_v_met_min = hof[fold_i_met_min]
            fold_p_met_min = period
            
            if fold_v_met_min < met_min:
                met_min = fold_v_met_min
                period_min = fold_p_met_min
                i_min = fold_i_met_min

            # -- CASE 2 (MAX)
            # max metric in evaluated HoF
            fold_i_met_max = hof.index(max(hof))
            fold_v_met_max = hof[fold_i_met_max]
            fold_p_met_max = period

            if fold_v_met_max > met_max:
                met_max = fold_v_met_max
                period_max = fold_p_met_max
                i_max = fold_i_met_max

            # Get features used for every case, therefore, for min and max metric cases
            features = {'features': p_global_cases[period]['features'],
                        'sym_features': p_global_cases[period]['sym_features']}
            met_cases[model]['hof_metrics']['data'][period]['features'] = features
        
        # update data with min case
        met_cases[model]['met_min']['data'] = p_global_cases[period_min][model]['e_hof'][i_min]
        met_cases[model]['met_min']['period'] = period_min
        met_min_params = p_global_cases[period][model]['p_hof']['hof'][i_min]
        met_cases[model]['met_min']['params'] = met_min_params
        met_cases[model]['met_min'][p_cases_type] = met_min
        
        # update data with max case
        met_cases[model]['met_max']['data'] = p_global_cases[period_max][model]['e_hof'][i_max]
        met_cases[model]['met_max']['period'] = period_max
        met_max_params = p_global_cases[period][model]['p_hof']['hof'][i_max]
        met_cases[model]['met_max']['params'] = met_max_params
        met_cases[model]['met_max'][p_cases_type] = met_max
        
        # mode(s) data and metrics
        met_cases[model]['met_mode']['data'] = met_mode[model]
        met_cases[model]['met_mode']['modes'] = []
        met_cases[model]['met_mode']['repetitions'] = []
        met_cases[model]['met_mode']['period'] = {}

        # find mode or modes of parameters
        all_tuples = list(met_mode[model].keys())
        mode_value = max([met_mode[model][i]['repetitions'] for i in all_tuples])
        
        # at least 1 repetition existed
        if mode_value > 0:
            for i in all_tuples:
                # all repeated tupples 
                if met_mode[model][i]['repetitions'] == mode_value:
                    # transfer key values (tupple with parameters)
                    met_cases[model]['met_mode']['modes'].append(str(tuple(met_mode[model][i]['params'])))
                    # number of repetitions of the mode or modes
                    met_cases[model]['met_mode']['repetitions'].append(mode_value)
                    # periods of ocurrence
                    key_period = str(tuple(met_mode[model][i]['params']))
                    met_cases[model]['met_mode']['period'].update({key_period: met_mode[model][i]['periods']})

    return met_cases
