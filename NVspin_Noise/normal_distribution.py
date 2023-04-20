from scipy.stats import lognorm
import numpy as np

mu, sigma = 0, 0.3 # mean and standard deviation of the normal distribution
s = sigma # shape parameter of the log-normal distribution
for x in range(10):
    random_numbers = lognorm(s, scale=np.exp(mu)).rvs(1)
    print(random_numbers)