import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x, y)
plt.grid()
plt.show()

# activation 은 가중치 값을 한정시킴