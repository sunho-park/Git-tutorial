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

# print(samsung.shape)    (509, 1) --> ex: ([1], [2], [3])   // (509, ) -> ex: (1, 2, 3)
# print(hite.shape)       (509, 5)

samsung = samsung.reshape(samsung.shape[0],)     # (509, )로 변환
print(samsung.shape)                             # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape)                # (504, 6, 1) //  # (504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)        # (504, 5)
print(y_sam.shape)        # (504, )

# 앙상블 행맞추기 위해 504로 자름
x_hit = hite[5:510, :]
print(x_hit.shape)        # (504, 5)

# train, test 자르기 
xsam_train, xsam_test, ysam_train, ysam_test, xhite_train, xhite_test =train_test_split(x_sam, y_sam, x_hit, shuffle=False, test_size=0.3)

# 데이터 전처리 MinMaxscaler
scaler = MinMaxScaler()

scaler.fit(xsam_train)
xs_train = scaler.transform(xsam_train)
xs_test = scaler.transform(xsam_test)

scaler.fit(xhite_train)
xh_train = scaler.transform(xhite_train)
xh_test = scaler.transform(xhite_test)

# PCA
pca = PCA(n_components=3)

pca.fit(xh_train)
xhpca_train = pca.transform(xh_train)
xhpca_test = pca.transform(xh_test)

# 차원 변경
xs_train = xs_train.reshape(352, 5, 1)  
xs_test = xs_test.reshape(152, 5, 1)

print('xs_train.shape : ', xs_train.shape)     
print('xs_test.shape : ', xs_test.shape)      
print('xhpca_train.shape : ', xhpca_train.shape)   
print('xhpca_test.shape : ', xhpca_test.shape) 

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

input2 = Input(shape=(3, ))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([xs_train, xhpca_train], ysam_train, epochs=1, batch_size=1)

# All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델 할때 주의 점 : 행숫자 까지를 맞춰줘야함.

# 4. 평가, 예측

loss, mse = model.evaluate([xs_test, xhpca_test], ysam_test, batch_size=1)
print("loss : ", loss)
print("mse : ", mse)

# y_predict = model.predict([[xs_test[-1]], [xhpca_test[-1]]])

print(xs_test.shape)  # (504, 5, 1)
print(xhpca_test.shape)  # (504, 5)

print(xs_test[-1].shape) # (5, 1)
print(xhpca_test[-1].shape) # (3, )

x1_predict = xs_test[-1]
x2_predict = xhpca_test[-1]

print(x1_predict.shape)  #(5, 1)
print(x2_predict.shape)  #(3, )   

x1_predict = x1_predict.reshape(1, 5, 1,)
x2_predict = x2_predict.reshape(1, 3)

print(x1_predict.shape) 
print(x2_predict.shape)

y_predict = model.predict([x1_predict, x2_predict])


print('y_predict : ', y_predict)
