import numpy as np
x = np.array(range(1, 11))
y = np.array([0, 2, 3, 4, 5, 0, 2, 3, 4, 5]) #스칼라 10개 벡터 1 디멘션 1 

from keras.utils import np_utils # one-hot 인코딩 1차원이 2차원이 된다.
y = np_utils.to_categorical(y)


print(x.shape) 
print(y.shape)   # (10, 6)
y=y[:, 1:6] 
print(y.shape)   # (10, 5)
print(y) 
