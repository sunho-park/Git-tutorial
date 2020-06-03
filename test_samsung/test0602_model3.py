# lstm 2개 구현

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)- size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

size = 6

# 1. 데이터 
# npy 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle='True')
hite = np.load('./data/hite.npy', allow_pickle=True)

#print(samsung.shape)    #(509, 1) --> ([1], [2], [3])   // (509, ) -> (1, 2, 3)
#print(hite.shape)       #(509, 5)

samsung = samsung.reshape(samsung.shape[0],)     # (509, )로 변환
print('samsung.shape : ',samsung.shape )         # (509, )

samsung = (split_x(samsung, size))
print('samsung.shape : ',samsung.shape )         # (504, 6, 1) //  # (504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)        # (504, 5)
print(y_sam.shape)        # (504, )

# 앙상블 input1과 행을 맞추려고 504로 자름
x_hit = hite[5:510, :]
print(x_hit.shape)        # (504, 5)

# 차원 변경
x_sam = x_sam.reshape(504, 5, 1)   #(504, 5, 1)
x_hit = x_hit.reshape(504, 5, 1)   #(504, 5, 1)

# 2. 모델 구성
input1 = Input(shape=(5, 1))
x1 = LSTM(100)(input1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)
x1 = Dropout(0.5)(x1)

input2 = Input(shape=(5, 1))
x2 = LSTM(100)(input2)
x2 = Dense(100)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.5)(x2)

merge = concatenate([x1, x2]) 
output = Dense(1)(merge)
# Concatenate은 Sequential형 concatenate은 함수형함수
# fusion = Concatenate()([x1, x2])
# output = Dense(1)(fusion)

model = Model(inputs=[input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_sam, x_hit], y_sam, epochs=1, batch_size=1)

# All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델 할때 주의 점 : 행숫자 까지를 맞춰줘야함.

# 4. 평가, 예측

loss, mse = model.evaluate([x_sam, x_hit], y_sam, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)

print(x_sam)
print(x_sam.shape)
x1_predict = x_sam[-1]
x2_predict = x_hit[-1]

print(x_sam[-1].shape) # (5, 1)
print(x_hit[-1].shape) # (5, 1)

x1_predict = x1_predict.reshape(1, 5, 1)
x2_predict = x2_predict.reshape(1, 5, 1)



y_predict = model.predict([x1_predict, x2_predict])

print('y_predict : ', y_predict)
