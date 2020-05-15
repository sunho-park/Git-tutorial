import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18])


from keras.layers import Sequential
from keras.models import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mes'] )
model.fit(x, y, epochs=100, batch_size=1)

loss, mse = model.evaluate(x_test, y_test)

print("loss = ", loss)
print("mse = ", mse)

y_predict=model.predict(x_test)

print("y_predict = ", y_predict)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


print("RMSE : ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
