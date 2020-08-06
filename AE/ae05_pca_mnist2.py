# keras56_mnist_DNN.py copy
# input_dim = 154 로 모델을 만드시오

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

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train : \n", y_train)
print("y_train.shape : ", y_train.shape) # (60000, 10) 

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

pca2 = PCA(n_components=154)
pca2.fit(X)
pca_X = pca2.transform(X)
print(pca_X)
print(pca_X.shape)

from sklearn.model_selection import train_test_split
pca_x_train, pca_x_test = train_test_split(pca_X, test_size=1/7)

print('pca_x_train.shape : ', pca_x_train.shape)
print('pca_x_test.shape : ', pca_x_test.shape)
print('y_train : ', y_train)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()

model.add(Dense(100, input_dim=(154)))
model.add(Dense(256, activation='relu')) 
model.add(Dense(512, activation='relu')) 
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(pca_x_train, y_train, epochs=50, batch_size=256)


loss, acc = model.evaluate(x_test, y_test)
decoded_imgs = model.predict(pca_x_test)

# matplotlib 사용
'''
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(pca_x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.show()'''
