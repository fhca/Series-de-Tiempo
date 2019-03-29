__author__ = 'fhca'

import pandas as pd

asc1 = pd.read_csv("https://raw.githubusercontent.com/fhca/Series-de-Tiempo/master/ascombe/asc1.txt", sep='\t',
                   header=None)
asc2 = pd.read_csv("https://raw.githubusercontent.com/fhca/Series-de-Tiempo/master/ascombe/asc2.txt", sep='\t',
                   header=None)
asc3 = pd.read_csv("https://raw.githubusercontent.com/fhca/Series-de-Tiempo/master/ascombe/asc3.txt", sep='\t',
                   header=None)
asc4 = pd.read_csv("https://raw.githubusercontent.com/fhca/Series-de-Tiempo/master/ascombe/asc4.txt", sep='\t',
                   header=None)
"""
print(asc1)
print(asc2)
print(asc3)
print(asc4)
"""
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(221, title='asc1')
plt.scatter(asc1[0], asc1[1])
plt.subplot(222, title='asc2')
plt.scatter(asc2[0], asc2[1])
plt.subplot(223, title='asc3')
plt.scatter(asc3[0], asc3[1])
plt.subplot(224, title='asc4')
plt.scatter(asc4[0], asc4[1])
plt.show()

r = pd.DataFrame({"media x": [asc1[0].mean(), asc2[0].mean(), asc3[0].mean(), asc4[0].mean()],
                  "media y": [asc1[1].mean(), asc2[1].mean(), asc3[1].mean(), asc4[1].mean()],
                  "varianza x": [asc1[0].var(), asc2[0].var(), asc3[0].var(), asc4[0].var()],
                  "varianza y": [asc1[1].var(), asc2[1].var(), asc3[1].var(), asc4[1].var()],
                  }, index=["asc1", "asc2", "asc3", "asc4"])

print(r)

s = pd.DataFrame(index=["asc1", "asc2", "asc3", "asc4"])
s["media x"] = [asc1[0].mean(), asc2[0].mean(), asc3[0].mean(), asc4[0].mean()]
s["media y"] = [asc1[1].mean(), asc2[1].mean(), asc3[1].mean(), asc4[1].mean()]
s["varianza x"] = [asc1[0].var(), asc2[0].var(), asc3[0].var(), asc4[0].var()]
s["varianza y"] = [asc1[1].var(), asc2[1].var(), asc3[1].var(), asc4[1].var()]

print(s)

t = pd.DataFrame([asc1[0], asc1[1], asc2[0], asc2[1], asc3[0], asc3[1], asc4[0], asc4[1]]).T
t.columns=["asc1 x", "asc1 y", "asc2 x", "asc2 y", "asc3 x", "asc3 y", "asc4 x", "asc4 y"]
# print(t)
print("MEDIAS:")
print(t.mean())
print("VARIANZAS:")
print(t.var())
print("DESVIACION:")
print(t.std())
