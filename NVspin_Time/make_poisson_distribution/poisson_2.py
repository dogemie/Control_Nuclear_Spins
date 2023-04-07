# from numpy import random
# x_list = random.poisson(lam=1, size=1000)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as stats

# sns.distplot(x_list, kde=False, color='lightcoral')
# plt.show()

from scipy.stats import poisson

y1 = poisson(10).pmf(500)
print(y1)