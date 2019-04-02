__author__ = 'fhca'

from math import log
import numpy as np

with open('LIVIGREPOSO_C3.txt', 'r') as f:
    data = [float(i) for i in f.read().split()]

N = len(data)
eps = 0.001
lyapunovs = [[] for i in range(N)]

for i in range(N):
    for j in range(i + 1, N):
        if np.abs(data[i] - data[j]) < eps:
            for k in range(min(N - i, N - j)):
                lyapunovs[k].append(log(1e-10+np.abs(data[i + k] - data[j + k])))

with open('lyapunov.txt', 'w') as f:
    for i in range(len(lyapunovs)):
        if len(lyapunovs[i]):
            string = str((i, sum(lyapunovs[i]) / len(lyapunovs[i])))
            f.write(string + '\n')
