# hite에 pca 적용하여 차원줄이고 lstm 에 대입

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping

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

#print(hite.shape)       #(509, 5)

samsung = samsung.reshape(samsung.shape[0],)     # (509, 1)을 (509, )로 변환
#print('samsung.shape : ',samsung.shape )         # (509, )

samsung = (split_x(samsung, size))
print('samsung.shape : ', samsung.shape)         # (504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print('x_sam.shape : ', x_sam.shape)        # (504, 5)
print('y_sam.shape : ', y_sam.shape)        # (504, )

hite = hite[5:510, :]

# train, test 자르기 
xsam_train, xsam_test, ysam_train, ysam_test, xhite_train, xhite_test = train_test_split(x_sam, y_sam, hite, shuffle=False, test_size = 0.3) 

# 데이터 전처리 / 표준화 / 2차원으로 변경해야 표준화가능
scaler = StandardScaler()

# train만 fit 해주기 test, predict는 평가만한다/ train의 계산치로 train 범위 밖의 test를 평가
scaler.fit(xsam_train)
xs_train = scaler.transform(xsam_train)
xs_test = scaler.transform(xsam_test)

scaler.fit(xhite_train)
xh_train = scaler.transform(xhite_train)
xh_test = scaler.transform(xhite_test)

# PCA

pca = PCA(n_components=1)
# pca.fit(xsam_train)
# pca.transform(xsam_train)
# pca.transform(xsam_test)

pca.fit(xh_train)
xhpca_train = pca.transform(xh_train)
xhpca_test = pca.transform(xh_test)

# 차원 변경 
print('xs_train.shape : ', xs_train.shape)     # (352, 5)
print('xs_test.shape : ', xs_test.shape)      # (152, 5)

print('xhpca_train.shape : ', xhpca_train.shape)    # (352, 1)
print('xhpca_test.shape : ', xhpca_test.shape)     # (152, 1)

xs_train = xs_train.reshape(352, 5, 1)   
xs_test = xs_test.reshape(152, 5, 1)
xhpca_train = xhpca_train.reshape(352, 1, 1)   
xhpca_test = xhpca_test.reshape(152, 1, 1)


# 2. 모델 구성
input1 = Input(shape=(5, 1))
x1 = LSTM(100, activation='relu')(input1)
x1 = Dense(100, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(100, activation='relu')(x1)
x1 = Dense(100, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(100, activation='relu')(x1)
x1 = Dense(100, activation='relu')(x1)

input2 = Input(shape=(1, 1))
x2 = LSTM(100, activation='relu')(input2)
x2 = Dense(100, activation='relu')(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(100, activation='relu')(x2)
x2 = Dense(100, activation='relu')(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(100, activation='relu')(x2)
x2 = Dense(100, activation='relu')(x2)

merge = Concatenate()([x1, x2]) 
output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련

es = EarlyStopping(monitor='loss', patience=20)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([xs_train, xhpca_train], ysam_train, epochs=300, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

# All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델 할때 주의 점 : 행 숫자가 중요함 열은 안맞아도 됨!


# 4. 평가, 예측

loss, mse = model.evaluate([xs_test, xhpca_test], ysam_test, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)

print(xs_test.shape)           #(152, 5, 1)
print(xhpca_train.shape)         #(352, 1, 1)

x1_predict = xs_test[-1]
x2_predict = xhpca_test[-1]

print(xs_test[-1].shape)       # (5, 1)
print(xhpca_test[-1].shape)      # (1, 1)

# x1_predict = x1_predict.reshape(1, 5, 1)
# x2_predict = x2_predict.reshape(1, 1, 1)

y_predict = model.predict([[x1_predict], [x2_predict]])  # 2020년 6월 2일 삼성전자 주가 예측

print('2020년 6월 2일 삼성전자 주가 : ', y_predict)

print('y_predict.shape : ', y_predict.shape) # (1, 1)

print('ysam_train.shape : ', ysam_train.shape)  # (352, )
print('ysam_test.shape : ', ysam_test.shape)    # (152, )


# 또는 밑에 처럼 해도 가능

'''
# print(xs_test[-1].shape)
# print(xhite_test[-1].shape)

# y_predict = model.predict([[xs_test[-1]], [xhite_test[-1]]])
# print('y_predict : ', y_predict)
'''

ysam_predict = model.predict([xs_test, xhpca_test])

print('ysam_predict : ', ysam_predict)
print('ysam_predict.shape : ', ysam_predict.shape)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(ysam_test, ysam_predict): 
    return np.sqrt(mean_squared_error(ysam_test, ysam_predict))
      
print("RMSE : ", RMSE(ysam_test, ysam_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(ysam_test, ysam_predict)
print("r2 : ", r2)

