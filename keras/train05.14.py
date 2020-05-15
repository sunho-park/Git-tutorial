import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)


from keras.layers import Dense
from keras.models import Sequential

model= Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))

mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse = ", mse)

y_predict = model.predict(x_test)
print(y_predict)