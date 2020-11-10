
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: principal.py : python script with the main functionality                                    -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import codigos.funciones as fn
from codigos.datos import price_data
import codigos.visualizaciones as vs

# primeras pruebas sera con 1 solo periodo
data = price_data[list(price_data.keys())[9]]

# ---------------------------------------------------------------------------------- datos para proyecto -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

# datos iniciales para hacer pruebas
datos = data

# ------------------------------------------------------------------------------- visualizacion de datos -- #
# ------------------------------------------------------------------------------- ---------------------- -- #

# grafica OHLC
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [datos['timestamp'].head(1), datos['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc = vs.g_ohlc(p_ohlc=datos, p_theme=p_theme, p_dims=p_dims, p_vlines=p_vlines, p_labels=p_labels)

# mostrar grafica
# ohlc.show()

# ----------------------------------------------------------------------- analisis exploratorio de datos -- #
# ----------------------------------------------------------------------- ------------------------------ -- #

# tabla de descripcion de datos
datos.describe()

# --------------------------------------------------------------- ingenieria de variables autoregresivas -- #
# --------------------------------------------------------------- -------------------------------------- -- #

# funcion para generar variables autoregresivas
datos_arf = fn.f_autoregressive_features(p_data=datos, p_nmax=30)

# Visualizacion: head del DataFrame
datos_arf.head(5)

# LaTeX: 3 formulas de ejemplo
# Pendiente

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
