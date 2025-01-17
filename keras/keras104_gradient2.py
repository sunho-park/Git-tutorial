import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
y = f(x)

gradient = lambda x : 2*x - 4

x0 = 0.0
# MaxIter = 10
MaxIter = 10

# learning_rate = 0.25
# learning_rate = 0.5 
learning_rate = 0.75

print("step/tx/tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(MaxIter):
    x1 = x0 - learning_rate*gradient(x0) 
    x0 = x1
    
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

#  # k- 검은색 선 sk 최소값에 점 찍음
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
