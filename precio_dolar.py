__author__ = 'fhca'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


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
ESTACIONARIEDAD (stationarity): La serie (con cierto de grado de confianza) es estacionaria si su media y desviación estandar móviles 
son cercanas a cero en todos sus puntos. (son practicamente unas horizontales sobre la recta y=0)

ESTACIONALIDAD (seasonal): Se refiere a los cambios producidos por las "estaciones" (como en las estaciones del año: primavera, 
verano, otoño, invierno), o períodos cíclicos (temporadas) que puede tener una serie.
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
#plt.plot(residuales['2019-01':])

#plt.xticks(size=7)
#plt.show()

# VERIFICANDO QUE      O = T + E + R
#plt.plot(dolar)
#plt.plot(tendencia + estacionalidad + residuales, color='red')
#plt.show()

# VERIFICANDO QUE LOS RESIDUALES SON UNA SERIE ESTACIONARIA
#res = residuales.dropna()
#test_stationarity(res)



### Predicciones
original = dolar['2018-01-01':'2019-03-08']
datos_reales = dolar['2019-03-09':]
descomposición = seasonal_decompose(original, freq=100)

tendencia = descomposición.trend
estacionalidad = descomposición.seasonal
residuales = descomposición.resid

grafica(original, tendencia, estacionalidad, residuales)
