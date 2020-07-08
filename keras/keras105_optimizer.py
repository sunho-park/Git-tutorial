# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
# 경사 하강법의 base를 두고 있다.
# optimizer = Adam(lr=0.001)             # loss :  [0.026419956237077713, 0.026419956237077713],    pred1 :  [[3.4618902]]
# optimizer = RMSprop(lr=0.001)          # loss :  [0.0002723459037952125, 0.0002723459037952125],  pred1 :  [[3.478336]]
#optimizer = SGD(lr=0.001)               # loss :  [0.05417201295495033, 0.05417201295495033],      pred1 :  [[3.3573825]]
# optimizer = Adadelta(lr=0.001)         # loss :  [9.587225914001465, 9.587225914001465],          pred1 :  [[-0.45733494]]
# optimizer = Adagrad(lr=0.001)          # loss :  [5.982320785522461, 5.982320785522461],          pred1 :  [[0.3570324]]
optimizer = Nadam(lr=0.001)              # loss :  [0.08329077810049057, 0.08329077810049057],      pred1 :  [[3.275275]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print('loss : ', loss)

pred1 = model.predict([3.5])
print('pred1 : ', pred1)




