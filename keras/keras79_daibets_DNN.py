from sklearn.datasets import load_diabetes
import numpy as np

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print('x : \n', x)
print('x.shape : ', x.shape) # (442, 10)
print('y : ', y)
print('y.shape : ', y.shape) #(442,)