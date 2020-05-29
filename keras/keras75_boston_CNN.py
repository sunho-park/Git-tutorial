from sklearn.datasets import load_boston
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
'''
data    : x 값
target  : y 값
'''
dataset = load_boston()
x  = dataset.data
y = dataset.target


print("x : ", x) 
print("x.shape : ", x.shape) # (506, 13)

'''
# 표준화
from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize

scaler = StandardScaler()      

scaler.fit(x)
x = scaler.transform(x) 
'''

# 차원에서 중요한 부분만 압축해서 n_component 에 따라 새로운 컬럼을 만든다.

# x = x.reshape(506, 13, 1, 1)
# print("x.shape : ", x.shape) # (506, 13)
# print("x : ", x)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x = pca.transform(x)
# print('pca.transform(x) : ', pca.transform(x))
print('x : ', x)
print("x.shape : ", x.shape) #(506, 2)

'''
import numpy as np
from sklearn.decomposition import PCA

x = np.array([[-1, -1, 3], [-2, -1, 3], [-3, -2, -1], [1, 1, -1], [2, 1, 1], [3, 2, 1]])
pca = PCA(n_components=1)
pca.fit(x)

print('pca.transform(x) : \n', pca.transform(x))
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

print("x_train : \n", x_train)                        # (455, 2)
print("x_train[0] : \n", x_train[0])
print("x_test : ", x_test)                            # (51, 2)
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("y_train : \n", y_train)
print("y_test : ", y_test)
print("y_train.shape : ", y_train.shape)          # (455, )
print("y_test.shape : ", y_test.shape)            # (51, )


x_train = x_train.reshape(455, 2, 1, 1)
x_test = x_test.reshape(51, 2, 1, 1)

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

# 모델구성


from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Dropout


model = Sequential()

model.add(Conv2D(32, (1, 1), activation='relu', input_shape = (2, 1, 1)))
model.add(Conv2D(64, (1, 1), activation='relu')) # 커널사이즈가 왜 (1,1)??
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(1, activation='softmax'))

model.summary()

# 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1, validation_split=0.1)

# 훈련 평가

loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)


# output = model.predict(x_test)

# print("output : \n", output)



