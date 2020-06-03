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

# print(samsung.shape)    (509, 1) --> ([1], [2], [3])   // (509, ) -> (1, 2, 3)
# print(hite.shape)       (509, 5)

samsung = samsung.reshape(samsung.shape[0],)     # (509, )로 변환
print(samsung.shape)                             # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape)                # (504, 6, 1) //  # (504, 6)

# print("samsung : ",samsung)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

# print('x_sam : ', x_sam)
# print('y_sam : ', y_sam)

print(x_sam.shape)        # (504, 5)
print(y_sam.shape)        # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)        # (504, 5)


# 2. 모델 구성
input1 = Input(shape=(5, ))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5, ))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_sam, x_hit], y_sam, epochs=5)

# All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델 할때 주의 점 : 행숫자 까지를 맞춰줘야함.

