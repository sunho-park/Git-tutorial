import numpy as np
import matplotlib.pyplot as plt

alpha=0.2

def leakyrelu(z, alpha):
    return np.maximum(alpha * z, z)

a = 0.2
x = np.arange(-5, 5, 0.1)
y = leakyrelu(x, a)  

# relu elu selu 찾기 relu leakyrrelu elu selu

plt.plot(x, y)
plt.grid()
plt.show()