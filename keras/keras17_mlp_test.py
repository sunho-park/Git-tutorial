#1. R2를 0,5 이하 2. 레이어 5개이상 인풋아웃풋포함 3. 노드의 개수 10개 이상 4. EPOCHS 30개 이상 5. BATCHI_SIZE = 8 이하 

#1. 데이터

import numpy as np
x=np.array([range(1, 101), range(311, 411), range(100)])
y=np.array(range(711, 811))

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   

#행과 열을 바꾸는 함수
x = np.transpose(x)
y = np.transpose(y)
#x=np.swapaxes(x,0,1)
#x=np.arange(300).reshape(100,3)


print("x : \n", x)
print("y : \n ", y)


print("x.shape : ", x.shape)
print("y.shape : ", y.shape)  

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,shuffle=False, test_size=0.855) # train x = (80,3) test (20,3)



# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

# Sequential 함수를 model 로 하겠다.

model = Sequential()
model.add(Dense(100, input_dim=3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=12, batch_size=8, validation_split=0.01, verbose=3)       # train (60.3), val(20.3),test(20,3)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=8) 
print("loss : ", loss)
print("mse = ", mse)


y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
      
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2) 
