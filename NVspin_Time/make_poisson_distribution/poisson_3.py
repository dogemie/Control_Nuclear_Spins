from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(seed=100)

for x in range(100):
    rand_pois = np.random.poisson(lam=100, size=1)/100
    print(rand_pois)