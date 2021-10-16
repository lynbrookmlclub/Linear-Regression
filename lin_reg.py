"""
install: 
    - sealion
    - matplotlib
    - numpy 
"""


import numpy as np

# Y = 2x + 1

np.random.seed(420)

X = np.random.randn(100,1) # 100 random samples

y = 2 * X + 1

import matplotlib.pyplot as plt

#plt.scatter(X, y, color = "green")
#plt.show()

# use a framework to find the line y = mx + b that makes X into Y

from sealion.regression import LinearRegression

lin_reg = LinearRegression()


lin_reg.fit(X, y.flatten()) # run the guessing game 

# what were the m and b values that were learnt? The

print(lin_reg.weights, lin_reg.bias)


plt.plot(X, y, label = "actual data")
plt.plot(X, lin_reg.predict(X), label = "predicted data")
plt.legend()
plt.show()

# pip install sealion






