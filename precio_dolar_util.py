__author__ = 'fhca'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_stationarity(timeseries, ventana=12, nombre="ORIGINAL"):
    from statsmodels.tsa.stattools import adfuller

    """
    :param timeseries: Enum
    :parama ventana: Int
    :return:

    Dada una timeseries y un tamaño de ventana, la grafica y calcula los valores de la prueba
    Dickey - Fuller, para determinar que tan estacionaria es.

    El valor de la estadística de la prueba tenemos que localizarlo DEBAJO de alguno de los
    valores críticos. Por ejemplo si es menor que el de 10%, eso significará que la serie
    original tiene un 90% de confianza de ser estacionaria.
    """

    # mMedidas móviles
    rol = timeseries.rolling(window=ventana, center=False)
    rolmean = rol.mean()
    rolstd = rol.std()

    # graficar medidas móviles:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Media móvil')
    std = plt.plot(rolstd, color='black', label='Desviación móvil')
    plt.legend(loc='best')
    plt.title(f'Serie: {nombre} - Media y Desviación Estándard móviles')
    plt.show(block=False)

    # Realiza prueba de Dickey-Fuller:
    print(f'Serie: {nombre} - Resultados de la prueba de Dickey-Fuller:')
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4],
                         index=['Estadística de la prueba', 'valor-p', '#retrasos usados', 'Número de observaciones'])
    for key, value in dftest[4].items():
        dfoutput['Valor crítico (%s)' % key] = value
    print(dfoutput)


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


def genera_residuales(n):
    "Genera n números al azar con distribución uniforme y devuelve su acumulado."
    r = np.random.rand(n) - .5
    ra = r.cumsum()
    return ra
