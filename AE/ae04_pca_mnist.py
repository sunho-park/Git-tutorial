# ae01_autoencoder.py copy

import numpy as np
import matplotlib.pyplot as plt

# from keras.datasets import mnist
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape : ", x_train.shape)  #(60000, 28, 28)
print("y_train.shape : ", y_train.shape) # (60000, ) 6만개 스칼라를 가진 벡터1개
print("x_test.shape : ", x_test.shape)    #(10000, 28, 28) 
print("y_test.shape : ", y_test.shape)   # (10000,)

# 데이터 전처리

# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print("y_train : \n", y_train)
# print("y_train.shape : ", y_train.shape) # (60000, 10) 

# 데이터 전처리 2. 정규화

x_train = x_train.reshape(60000, 784).astype('float32')/255                                                            
x_test = x_test.reshape(10000, 784).astype('float32')/255     

X = np.append(x_train, x_test, axis=0)
print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

best_n_components = np.argmax(cumsum >= 0.95) + 1 
print(best_n_components)




