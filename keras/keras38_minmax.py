from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [2000, 3000, 4000],[3000, 4000, 5000], [4000, 5000, 6000], [100, 200, 300]])            #(14, 3)
 
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])                                #(14,  )

x_predict = array([55, 65, 75])                                                                     # (3, )

print("x.shape : ", x.shape)   
print("y.shape : ", y.shape)   
print("x_predict.shape : ", x_predict.shape)   
print("================================================================")

x_predict = x_predict.reshape(1, 3)
print("x_predict.shape : ", x_predict.shape)   

##############################################################################################


from sklearn.preprocessing import MinMaxScaler, StandardScaler, robust_scale, normalize

scaler = MinMaxScaler()        #정규화
# scaler = StandardScaler()       #표준화

scaler.fit(x)
x = scaler.transform(x)  #fit 하고 tranform 해야함.
x_predict = scaler.transform(x_predict)

# 정규화를 하는 이유 http://hleecaster.com/ml-normalization-concept/

##############################################################################################


print("x : \n", x)
print("x_predict : \n", x_predict)
print("x_predict.shape : ", x_predict.shape)   
print("================================================================")


################### 틀 변환 ###########################################
# x = x.reshape(4, 3, 1) 
x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1, 3, 1)

print("x.shape[0] : ", x.shape[0])
print("x_predict.shape : ", x_predict.shape)   
print("x.shape", x.shape)
print("x : \n ", x)




# 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=True)) # (none, 3, 1) 10은 노드의 갯수
# model.add(LSTM(10, input_length=3, input_dim=1, return_sequences=True))   return_sequence = True 는 차원을 유지시켜준다. 3차원
model.add(LSTM(10, return_sequences=False)) # LSTM 의 출력 값은 2차원
model.add(Dense(5) )  #2차원
model.add(Dense(1))
model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5)

'''
print("x_predict", x_predict)
print("x_predict.shape", x_predict.shape)
x_predict = x_predict.reshape(1, 3, 1)

print("x_predict : ",x_predict)
print("x_predict.shape", x_predict.shape)
'''


y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)




