__author__ = 'fhca'

import numpy as np
from scipy import stats

# np.random.seed(0)
n = 151
a = np.random.randint(10, 1000, n)  # randint genera numeros enteros aleatorios
print(a)
a_copia = a.copy()
media = a.mean()
print("Media:", media)
mediana = np.median(a)
print("Mediana:", mediana)
print("Moda:", stats.mode(a))

"Cálculos "
print("Otra media:", a.sum() / n)
a.sort()
print(a)
print("Otra mediana:", a[int(n / 2)])

d = dict()
for x in a:
    d[x] = d.get(x, 0) + 1
t = np.array(list(d.items()), dtype=[('llave', int), ('valor', int)])
t_ordenada = np.sort(t, order='valor')
"Puede salir diferente, debido a que el valor que puede haber mas de un valor que mas aparece."
moda = t_ordenada[-1]['llave']
print("Otra Moda:", moda)

print("Desviación estándar:", a.std())

"La raíz del promedio de las diferencias de cada dato con la media elevadas al cuadrado."
desviacion = np.sqrt(np.sum((a - media) ** 2) / n)
print("Otra desviación estándar:", desviacion)

varianza = a.var()
print("Varianza:", varianza)
print("Otra Varianza:", desviacion ** 2)

"""graficas"""

import matplotlib.pyplot as plt

plt.figure()
l = len(a_copia)
x = list(range(l))
plt.plot(x, a_copia)
plt.plot(x, [media] * l, color="orange")
plt.plot(x, [mediana] * l, color="green")
plt.plot(x, [moda] * l, "*", color="red")
plt.plot(x, [desviacion] * l, "-.", color="purple")
#plt.plot(x, [varianza] * l)
plt.show()
