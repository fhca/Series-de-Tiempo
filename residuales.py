__author__ = 'fhca'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def genera_residuales(n):
    r = np.random.rand(n)*2-1
    ra = r.cumsum()
    plt.figure()
    plt.plot(ra)
    plt.show()

genera_residuales(31)