__author__ = 'fhca'

"""
Calcula la mejor recta que se ajuste a una colección de datos dada...
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datos = pd.read_csv("tendencia_desde_01_25.csv", index_col=0, parse_dates=True)
plt.plot(datos, ".")


def f1(x):
    return 0.006058344 * x + 19.06257537

plt.plot(datos.index, f1(np.arange(len(datos.index))))

plt.show()
