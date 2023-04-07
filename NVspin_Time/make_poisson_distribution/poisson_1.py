import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp

# Probability density of the Poisson distribution
def pois_dist(n, lamb):
    pd = (lamb ** n) * exp(-lamb) / factorial(n)
    return pd


x = np.arange(40)
pd1 = np.array([(pois_dist(n, 10)) for n in range(40)])
plt.ylim(0, 0.15)
plt.text(33.5, 0.14, 'lamb = 1')
plt.bar(x, pd1, color='lightcoral')
plt.show()