__author__ = 'fhca'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pasajeros = pd.read_csv("AirPassengers.csv", index_col=0, parse_dates=True)


def media_movil(a, v):
    """Calcula las medias móviles del arreglo a con ventanas
    de tamaño v."""
    n = len(a)
    return np.array([np.sum(a[i:i + v]) / v for i in range(n - v + 1)])


def mediam(serie, v):
    """Calcula la media movil de la serie (un dataframe) con tamaño de ventana v.
    Devuelve dataframe."""
    mm = media_movil(serie, v)
    mm = pd.DataFrame(mm)
    lm = len(mm)
    mm.index = serie.index[:lm]
    mm.columns = serie.columns
    return mm

ventana = 4  #valor entero

mm = mediam(pasajeros, ventana)
azuln = (pasajeros - pasajeros.mean()) / pasajeros.std()
azuln.dropna(inplace=True)
naranjan = mediam(azuln, ventana)

plt.figure()
#plt.plot(pasajeros)
#plt.plot(mm)
plt.plot(azuln)
plt.plot(naranjan)
verde = azuln - naranjan
plt.plot(verde)

rojan = (verde - verde.mean()) / verde.std()
plt.plot(rojan)


print("media de pasajeros=", pasajeros.mean())
print("desviación de pasajeros=", pasajeros.std())
print("media de la resta=", azuln.mean())
print("desviación de la resta=", azuln.std())

print("media de la verde=", verde.mean())
print("desviación de la verde=", verde.std())

print("media de la roja=", rojan.mean())
print("desviación de la roja=", rojan.std())

plt.show()
