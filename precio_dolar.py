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

dolar = pd.read_excel("tipoCambio.xls", skiprows=8, index_col=0, parse_dates=True, header=None)[1][1:]

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
"""

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
