import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x)

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)  # 0 이하의 데이터가 소멸해버림

# y = Leaky
# relu elu selu 찾기 relu leakyrrelu elu selu


plt.plot(x, y)
plt.grid()
plt.show()