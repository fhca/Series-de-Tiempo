__author__ = 'fhca'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from precio_dolar_util import *

######### LECTURA DE DATOS
dolar = pd.read_excel("tipoCambio.xls", skiprows=8, index_col=0, parse_dates=True, header=None)[2][1:]

######### Limpieza de datos
dolar.replace(to_replace='N/E', value=np.nan, inplace=True)
dolar.fillna(method='bfill', inplace=True)

######### Extracción de valores conocidos (separamos la parte a predecir)
original = dolar['2018-01-01':'2019-03-08']
datos_a_predecir = dolar['2019-03-09':]
### No se vale modificar antes de esta linea!!!

######### Descomposición estacional
freq = 31
descomposición = seasonal_decompose(original, freq=freq)

tendencia = descomposición.trend
estacionalidad = descomposición.seasonal
residuales = descomposición.resid

# Limpiamos los NaNs
original.dropna(inplace=True)
tendencia.dropna(inplace=True)
estacionalidad.dropna(inplace=True)
residuales.dropna(inplace=True)

# Graficamos las series
grafica(original, tendencia, estacionalidad, residuales)


######### Predicciones
# recta que asumimos como la continuación de la tendencia
# Demasiado simple, pues tomamos dos puntos cualesquiera cercanos al final
# y esta es la recta que los une.
def f(x):
    return -0.010896875 * x + 19.694882

def f1(x):
    return 0.006058344 * x + 19.06257537

x_extra = pd.date_range(start='2019-03-09', end='2019-04-08')
y_extra = [f1(x) for x in range(43, 43+31)]

resultados = pd.DataFrame(y_extra, index=x_extra, columns=["tendencia"])

# retrocedo 100 dias y de ahí empiezo a contar 31 días para obtener el rango
# de valores 'x' de donde sacaremos los valores 'y' a agregar
ciclo_anterior = pd.date_range(datetime(2019, 3, 9) - timedelta(days=freq), periods=31)

# extrayendo de estacionalidad el ciclo anterior (devuelve un 'Series')
resultados["estacionalidad"] = np.array(estacionalidad[ciclo_anterior])
resultados["residuales"] = genera_residuales(31)

resultados["PREDICCIÓN"] = resultados["tendencia"] + resultados["estacionalidad"] + resultados["residuales"]
resultados["DATOS REALES"] = datos_a_predecir

plt.plot(resultados["PREDICCIÓN"])
plt.plot(resultados["DATOS REALES"])

# Cálculo del error cuadrático promedio
resultados["diferencia2"] = np.square(resultados["PREDICCIÓN"] - resultados["DATOS REALES"])

plt.plot(resultados["diferencia2"])

print(resultados)
print("Error cuadrático promedio = ", resultados["diferencia2"].mean())

plt.show()

#plt.plot(tendencia["2019-01-25":])

#tendencia["2019-01-25":].to_csv("tendencia_desde_01_25.csv")
plt.show()
