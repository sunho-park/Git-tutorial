import numpy as np
x = np.array([range(1,101), range(311, 411), range(100)])
y = np.array(range(711,811))

x = np.transpose(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)

print("loss = ", loss)
print("mse = ", mse)

y_predict = model.predict(x_test)
print("y_predict = ", y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE = ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 = ", r2)




