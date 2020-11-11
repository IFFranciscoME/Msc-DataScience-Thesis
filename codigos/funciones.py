
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: datos.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler  # estandarizacion de variables


# --------------------------------------------------------------------------------- preparacion de datos -- #
# --------------------------------------------------------------------------------- -------------------- -- #

def f_transformacion(p_datos, p_transformacion):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_transformacion: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)

    p_datos: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    if p_transformacion == 'Standard':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        # armar objeto de salida
        p_datos[list(p_datos.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_transformacion == 'Robust':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        # armar objeto de salida
        p_datos[list(p_datos.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_transformacion == 'Scale':

        # estandarizacion de todas las variables independientes
        lista = p_datos[list(p_datos.columns[1:])]

        p_datos[list(p_datos.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_datos


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
        data['ma_vol_' + str(n + 2)] = data['volume'].rolling(n + 2).mean()

        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 2)] = data['ol'].rolling(n + 2).mean()

        # promedio movil de ventana n
        data['ma_ho_' + str(n + 2)] = data['ho'].rolling(n + 2).mean()

        # promedio movil de ventana n
        data['ma_hl_' + str(n + 2)] = data['hl'].rolling(n + 2).mean()

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

    # para hacer calculos
    n = p_nmax

    for n in range(p_nmax):

        # hadamard product of previously generated features
        list_hadamard = [p_data['lag_vol_' + str(n + 1)], p_data['lag_ol_' + str(n + 1)],
                         p_data['lag_ho_' + str(n + 1)], p_data['lag_hl_' + str(n + 1)]]

        for x in list_hadamard:
            p_data['h_' + 'lag_vol_' + str(n+1) + '_' + 'ma_vol_' + str(n+2)] = x*p_data['ma_vol_' + str(n+2)]
            p_data['h_' + 'lag_vol_' + str(n+1) + '_' + 'ma_ol_' + str(n+2)] = x*p_data['ma_ol_' + str(n+2)]
            p_data['h_' + 'lag_vol_' + str(n+1) + '_' + 'ma_ho_' + str(n+2)] = x*p_data['ma_ho_' + str(n+2)]
            p_data['h_' + 'lag_vol_' + str(n+1) + '_' + 'ma_hl_' + str(n+2)] = x*p_data['ma_hl_' + str(n+2)]

    return p_data
