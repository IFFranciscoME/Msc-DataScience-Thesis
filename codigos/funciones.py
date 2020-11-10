
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
from sklearn.preprocessing import StandardScaler       # estandarizacion de variables


# --------------------------------------------------------------------------------- preparacion de datos -- #
# --------------------------------------------------------------------------------- -------------------- -- #

def f_estandarizar(p_datos):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada

    Parameters
    ----------
    p_datos: pd.DataFrame
        Con datos numericos de entrada

    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados

    """

    # estandarizacion de todas las variables independientes
    lista = p_datos[list(p_datos.columns[1:])]

    # armar objeto de salida
    p_datos[list(p_datos.columns[1:])] = StandardScaler().fit_transform(lista)

    return p_datos
