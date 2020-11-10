
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: principal.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import codigos.funciones as fn
from codigos.datos import price_data

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[0]]

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
datos = data.iloc[0:400, ]
datos = fn.f_estandarizar(p_datos=datos)

# ------------------------------------------------------------------------------- visualizacion de datos -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# grafica OHLC

# ----------------------------------------------------------------------- analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos

# --------------------------------------------------------------- ingenieria de variables autoregresivas -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# LaTeX: 3 formulas de ejemplo
# Visualizacion: head del DataFrame

# ------------------------------------------------------------------- ingenieria de variables simbolicas -- #
# ------------------------------------------------------------------- ---------------------------------- -- #

# Lista de operaciones simbolicas
# LaTeX: 3 formulas de ejemplo
# Visualizacion: head del DataFrame

# ----------------------------------------------------------------- importancia y seleccion de variables -- #
# ----------------------------------------------------------------- ------------------------------------ -- #

# metodo 1 para importancia de variables
# metodo 1 para seleccion de variables

# ----------------------------------------------------------------------------------- conjuntos de datos -- #
# ----------------------------------------------------------------------------------- ------------------ -- #

# Tipo 1: 1 solo data set con 80% Entrenamiento y 20% Prueba
# Tipo 2: 1 data set por mes con 80% Entrenamiento y 20% Prueba

# ---------------------------------------------------------------------------- ajuste de modelo OLS-L1L2 -- #
# ---------------------------------------------------------------------------- ------------------------- -- #

# ajuste de modelo
# Visualizacion: Mostrar metricas de performance

# ------------------------------------------------------------------------------ ajuste de modelo LS-SVM -- #
# ------------------------------------------------------------------------------ ----------------------- -- #

# ajuste de modelo
# Visualizacion: Mostrar metricas de performance

# --------------------------------------------------------------------------------- ajuste de modelo MLP -- #
# --------------------------------------------------------------------------------- -------------------- -- #

# ajuste de modelo
# Visualizacion: Mostrar metricas de performance
