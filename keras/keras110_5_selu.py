import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

alpha=0.2

def elu(x, alpha):
    x = np.copy(x)
    x[x<0]=alpha*(np.exp(x[x<0])-1)
    return x


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    #return scale * tf.where(x >= 0.0, x, alpha * tf.exp(x) - alpha)
    return scale * elu(x, alpha)


a = 0.2
x = np.arange(-5, 5, 0.1)
y = selu(x)  

plt.plot(x, y)
plt.grid()
plt.show()