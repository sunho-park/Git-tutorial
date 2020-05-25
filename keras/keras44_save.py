# keras40을 카피해서 복붙

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


# 2. 모델
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4, 1)))
model.add(Dense(5))
model.add(Dense(10)) # y.shape = (6, 1)

model.summary()

model.save(".//model//save_keras44.h5")
# model.save(".\model\save_keras44.h5")
# model.save("./model/save_keras44.h5")   

print("저장 잘됬다.")

'''
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=50, mode='auto')
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss, mse = model.evaluate(x, y)

print("lose : ", loss)
print("mse : ", mse)


x_predict = np.array(range(15, 19)) 

print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 4, 1)

print("x_predict : \n",x_predict)
print("x_predict.shape", x_predict.shape)

y_predict = model.predict(x_predict)

print("y_predict : ", y_predict)'''

