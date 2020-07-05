import numpy as np
import matplotlib.pyplot as plt

alpha=0.2
'''
# 오류 코드...The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# def elu(z, alpha):
#     return z if z >= 0 else alpha*(np.exp(z)-1)

# a = 0.2
# x = np.arange(-5, 5, 0.1)
# y = elu(x)  
'''
def elu(x, alpha):
    x = np.copy(x)
    x[x<0]=alpha*(np.exp(x[x<0])-1)
    return x

a = 0.2
x = np.arange(-5, 5, 0.1)
y = elu(x, a)  

# 리스트 컴프리헨션
# a = 0.2
# x = np.arange(-5,5,0.1)
# y = [x if x>0 else a*(np.exp(x)-1) for x in x]

plt.plot(x, y)
plt.grid()
plt.show()