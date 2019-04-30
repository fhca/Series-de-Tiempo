__author__ = 'fhca'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta


def test_stationarity(timeseries):
    # mMedidas móviles
    rol = timeseries.rolling(window=12, center=False)
    rolmean = rol.mean()
    rolstd = rol.std()

    # graficar medidas móviles:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Media móvil')
    std = plt.plot(rolstd, color='black', label='Desviación móvil')
    plt.legend(loc='best')
    plt.title('Media y Desviación Estándard móviles')
    plt.show(block=False)

    # Realiza prueba de Dickey-Fuller:
    print('Resultados de la prueba de Dickey-Fuller:')
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4],
                         index=['Estadística de la prueba', 'valor-p', '#retrasos usados', 'Número de observaciones'])
    for key, value in dftest[4].items():
        dfoutput['Valor crítico (%s)' % key] = value
    print(dfoutput)

# dolar = \
#    pd.read_csv("https://www.exchange-rates.org/HistoryExchangeRatesReportDownload.aspx?base_iso_code=USD&iso_code=MXN",
#                index_col=0)['Rate']

dolar = pd.read_excel("tipoCambio.xls", skiprows=8, index_col=0, parse_dates=True, header=None)[2][1:]

dolar.replace(to_replace='N/E', value=np.nan, inplace=True)
dolar.fillna(method='bfill', inplace=True)

"""
plt.figure()
ax = plt.axes()
# plt.plot(dolar)
rol = dolar.rolling(365)  # crea 'objeto' de estadisticas móviles para el dataframe, ventana=365
plt.plot(rol.mean())    # muestra media movil del df
plt.plot(rol.std())     # muestra desviación estándar móvil
plt.xticks(rotation=90)
xmin, xmax = ax.get_xlim()
ax.set_xticks(np.round(np.linspace(xmin, xmax, 20)))
plt.show()
 
# SERIE ORIGINAL
# test_stationarity(dolar)
 
# SACANDOLE EL LOG base 10
dolar_log = np.log(dolar)
# test_stationarity(dolar_log)
 
rol = dolar_log.rolling(window=12, center=False)  # calcula las ventanas
mm = rol.mean()
dm = rol.std()
dolar_log_menosmediamovil = dolar_log - mm
dolar_log_menosmediamovil.dropna(inplace=True)
#test_stationarity(dolar_log_menosmediamovil)
 
dolar_log_normalizado = dolar_log_menosmediamovil / dm
dolar_log_normalizado.dropna(inplace=True)
test_stationarity(dolar_log_normalizado)
 
 
promedio_movil_ponderado_exponencial = dolar.ewm(halflife=12, min_periods=0, adjust=True, ignore_na=False).mean()
#plt.plot(dolar)
#plt.plot(promedio_movil_ponderado_exponencial, color='red')
test_stationarity(dolar - promedio_movil_ponderado_exponencial)
plt.show()
"""

"""
ESTACIONARIEDAD (stationarity): La serie (con cierto de grado de confianza) es estacionaria si su media y desviación 
estandar móviles son cercanas a cero en todos sus puntos. (son practicamente unas horizontales sobre la recta y=0)
 
ESTACIONALIDAD (seasonal): Se refiere a los cambios producidos por las "estaciones" (como en las estaciones del año: 
primavera, verano, otoño, invierno), o períodos cíclicos (temporadas) que puede tener una serie.
"""

from statsmodels.tsa.seasonal import seasonal_decompose


def grafica(original, tendencia, estacionalidad, residuales):
    ticksize = 6
    plt.subplot(411)
    plt.plot(original, label='Original')
    plt.legend(loc='best')
    plt.xticks(size=ticksize)
    plt.subplot(412)
    plt.plot(tendencia, label='Tendencia')
    plt.legend(loc='best')
    plt.xticks(size=ticksize)
    plt.subplot(413)
    plt.plot(estacionalidad, label='Estacionalidad')
    plt.legend(loc='best')
    plt.xticks(size=ticksize)
    plt.subplot(414)
    plt.plot(residuales, label='Residuales')
    plt.legend(loc='best')
    plt.xticks(size=ticksize)
    plt.tight_layout()
    plt.show()


"""# tomando menos datos
dolar=dolar['2019-03-09':]
 
descomposición = seasonal_decompose(dolar, freq=7)
 
tendencia = descomposición.trend
estacionalidad = descomposición.seasonal
residuales = descomposición.resid
 
grafica(dolar, tendencia, estacionalidad, residuales)
 
"""

# Prueba para Samuel... ganas $0.03 por cada dolar si se compra en viernes y se vende al martes siguiente
# plt.plot(residuales['2019-01':])

# plt.xticks(size=7)
# plt.show()

# VERIFICANDO QUE      O = T + E + R
# plt.plot(dolar)
# plt.plot(tendencia + estacionalidad + residuales, color='red')
# plt.show()

# VERIFICANDO QUE LOS RESIDUALES SON UNA SERIE ESTACIONARIA
# res = residuales.dropna()
# test_stationarity(res)


### Predicciones
original = dolar['2018-01-01':'2019-03-08']
datos_reales = dolar['2019-03-09':]
### No se vale modificar antes de esta linea!!!


freq = 100

descomposición = seasonal_decompose(original, freq=freq)

tendencia = descomposición.trend
estacionalidad = descomposición.seasonal
residuales = descomposición.resid

grafica(original, tendencia, estacionalidad, residuales)

# print(len(datos_reales))
# print(len(original), len(tendencia), len(estacionalidad), len(residuales))


"""
tend = tendencia['2019-01':].dropna()
plt.plot(tend)
plt.plot([datetime(2019, 1, 1), datetime(2019, 1, 17)], [19.694882, 19.520532], color='red')
print(tendencia['2019-01':])
"""

"""
 AGREGANDO MAS DATOS A LA TENDENCIA
 
 Viendo que la gráfica de la tendencia nos dá una curva, pero los últimos datos tienen
 cierto parecido a una recta, ASUMIREMOS que los datos a predecir SON UNA RECTA que 
 continua esta.
 
 La recta la construiremos con aquella que pasa por dos puntos con la ec. 
 y-y1=(y2-y1)/(x2-x1)*(x-x1)
 
 Para (x1, y1) = (0, 19.694882)    (x2, y2) = (16, 19.520532)
 
 Donde el día 0 corresponde a la fecha 2019-01-01
 y donde el dia 16 corresponde a la fecha 2019-01-17
 
 La ec. resultante es  y = -0.010896875 * x + 19.694882 y la ponemos en una función
 para calcular rápidamente los 31 valores que queremos extrapolar.
 
 Pero hay 50 datos NaN desde 2019-01-18, así la extrapolación tendrá que incluirlos.
"""


def f(x):
    return -0.010896875 * x + 19.694882


nextra = 50 + 31
x_extra = pd.date_range(start='2019-01-18', periods=nextra)
y_extra = [f(x) for x in range(17, 17 + nextra)]
extra = pd.DataFrame(y_extra, index=x_extra)
# print(extra)

tendencia.dropna(inplace=True)
# plt.plot(tendencia)
# plt.plot(extra)


tendencia2 = tendencia.append(extra)
# plt.plot(tendencia2)


"""
  La estacionalidad contiene un patron repetitivo que tendríamos que reproducir
  tantas fechas como queramos reproducir
      
"""

# 31 dias a agregar (fechas para predecir)
x_extra = pd.date_range(start='2019-03-09', end='2019-04-08')

# retrocedo 100 dias y de ahí empiezo a contar 31 días para obtener el rango
# de valores 'x' de donde sacaremos los valores 'y' a agregar
ciclo_anterior = pd.date_range(datetime(2019, 3, 9) - timedelta(days=100), periods=31)

# extrayendo de estacionalidad el ciclo anterior (devuelve un 'Series')
y_extra = estacionalidad[ciclo_anterior]

# construimos la tabla con los valores 'y', convertidos a array y con indice x_extra
y_extra = pd.DataFrame(np.array(y_extra), index=x_extra)

estacionalidad2 = estacionalidad.append(y_extra)
# TODO: añadir la predicción para los residuales (utilizando caminante aleatorio)

plt.plot(estacionalidad2)

# estacionalidad2 = estacionalidad.append()



plt.show()
