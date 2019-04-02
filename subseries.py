__author__ = 'fhca'

import numpy as np
import pandas as pd


def lee_serie_con_numpy(archivo):
    """Abre el archivo de texto plano (ASCII),
    que debe contener un dato por linea, utilizando
    únicamente Numpy. Devuelve un arreglo de numpy.
    """
    with open(archivo) as f:
        a = np.array([float(e) for e in f.readlines()])
    return a


def lee_serie_con_pandas(archivo):
    """Abre el archivo de texto plano (ASCII),
    que debe contener un dato por linea, utilizando
    Pandas. Devuelve un arreglo de numpy.

    pd.read_csv devuelve un 'DataFrame'
      (el archivo no tiene encabezado)
    escojemos la primera columna, la [0] y eso
      devuelve un 'Series'
    Convertimos el Series a arreglo de Numpy con np.array
    """
    a = np.array(pd.read_csv(archivo, header=None)[0])
    return a


a = lee_serie_con_pandas("LIVIG40.txt")
n = len(a)  # n = a.size  ó  n = a.shape[0]
print(a)
v = 4

def media_movil(a, v):
    """Calcula las medias móviles del arreglo a con ventanas
    de tamaño v."""
    return np.array([np.sum(a[i:i + v]) / v for i in range(n - v + 1)])


import matplotlib.pyplot as plt
plt.figure()
"gráfica de la serie"
plt.plot(a)
mm = media_movil(a, v)
xm = list(range(v-1, n))
"gráfica de las medias móviles"
plt.plot(xm, mm)
"gráfica de las medias recortadas"
plt.plot(xm[::10], mm[::10])
plt.show()

