from keras import Sequential
from keras.layers import Dense
import numpy as np

x_train=np.array([1,2,3,4,5])
y_train=np.array([1,2,3,4,5])

x_test=np.array([1,2,3,4,5])
y_test=np.array([1,2,3,4,5])

model=Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#케라스 모델 실행
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss = ", loss)
print("acc = ", acc)




