from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense

import numpy as np


#1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape) # (4, 2)
print(y_data.shape) # (4, )

#2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)

model=Sequential()
model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=500, verbose=1, batch_size=1)

#4. 평가 예측

x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
x_test = np.array(x_test)

loss_acc = model.evaluate(x_data, y_data, batch_size=1)
# acc = accuracy_score([0, 1, 1, 0], y_predict)

y_predict = model.predict(x_test)
print("loss, acc = ", loss_acc)


# print(x_test, "의 예측 결과 : \n", y_predict)
