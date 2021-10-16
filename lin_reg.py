import numpy as np

# Y = 2x + 1

np.random.seed(420)

X = np.random.randn(100, 1)  # Array of 100 rows and 1 column. Random samples
y = 2 * X + 1

# random offsets
X += np.random.randn(100, 1) / 10
y += np.random.randn(100, 1) / 10

# use a framework to find the line y = mx + b that makes X into Y
# remember to star SeaLion: https://github.com/anish-lakkapragada/SeaLion

from sealion.regression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y.flatten())   # run the guessing game 

# what were the m and b values that were learnt?
# m is lin_reg.weights[0] and b is lin_reg.bias

print(lin_reg.weights[0], lin_reg.bias)

# Display line and points
import matplotlib.pyplot as plt

plt.scatter(X, y, label = "actual data")
plt.plot(X, lin_reg.predict(X), label = "predicted data")
plt.legend()
plt.show()
