import numpy as np
import pandas as pd

df1 = pd.read_csv("./data/csv/하이트 주가.csv", header=0, encoding='cp949', index_col=0, sep=',')

# df1 = df1.dropna()


df1 = df1[:509]
df1 = df1.fillna(0)

print("하이트 주가 : ", df1)
print(df1.shape)
print(type(df1))


df2 = pd.read_csv("./data/csv/삼성전자 주가.csv", header=0, encoding='cp949', index_col=0, sep=',')

df2 = df2.dropna()


print("삼성전자 주가 : ", df2)
print(df2.shape)
print(type(df2))

# 수치형 데이터로 변환

# 하이트

for i in range(len(df1.index)):  
    for j in range(len(df1.iloc[i])):
        df1.iloc[i, j] = str(df1.iloc[i, j]).replace(',', '')        
        df1.iloc[i, j] = int(df1.iloc[i, j])

for i in range(len(df2.index)):# str -> int 변경
    df2.iloc[i,0] = str(df2.iloc[i, 0]).replace(',','')
    df2.iloc[i,0] = str(df2.iloc[i, 0])

# 오름차순


df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])


print('df1 : ', df1)
print('df2 : ', df2)

df1 = df1.values
df2 = df2.values

print(df1)
print(df2)
print(type(df1), type(df2))
print(df1.shape, df2.shape)


np.save('./data/csv/하이트 주가.npy', arr = df1)
np.save('./data/csv/삼성전자 주가.npy', arr = df2)


'''
hite = np.load('./data/csv/하이트 주가.npy', allow_pickle=True)
samsung = np.load('./data/csv/삼성전자 주가.npy', allow_pickle=True)

print('hite : ', hite)
print('samsung : ', samsung)

print('hite.shape : ', hite.shape)
print('samsung.shape : ', samsung.shape)

'''
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

hite = np.load('./data/csv/하이트 주가.npy', allow_pickle=True)
samsung = np.load('./data/csv/삼성전자 주가.npy', allow_pickle=True)


# print('hite : ', hite)
# print('samsung : ', samsung)

print('hite.shape : ', hite.shape)           # (509, 5)
print('samsung.shape : ', samsung.shape)     # (509, 1)

def split_xy1(data, ts):
    a, b = list(), list()
    for i in range(len(data)):
        e_n= i +ts
        if e_n > len(data) - 1:
            break
        t_x, t_y = data[i:e_n], data[e_n]
        a.append(t_x)
        b.append(t_y)
    return np.array(a), np.array(b)

x1, y1 = split_xy1(samsung, 5)

'''
print('x1 : ', x1)
print('y1 : ', y1)
print('x1.shape : ', x1.shape)  # 504, 5, 1
print('y1.shape : ', y1.shape)  # 504, 1
'''

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]

        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x2, y2 = split_xy5(hite, 5, 1)

print('x2 : ', x2) 
print('y2 : ', y2)
print('x1.shape : ', x1.shape)   # (504, 5, 1)
print('x2.shape : ', x2.shape)   # (504, 5, 5)
print('y2.shape : ', y2.shape)   # (504, 1)  

x1 = x1.reshape(504, 5)
x2 = x2.reshape(504, 25)
# 데이터 전처리

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(x1)
x1 = scaler1.transform(x1)


scaler2 = StandardScaler()
scaler2.fit(x2)
x2 = scaler2.transform(x2)


# 데이터 셋 나누기

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=1, test_size=0.3)

print(x2_train.shape)       # 352, 5, 5
print(x2_test.shape)        # 152, 5, 5
print(y2_train.shape)       # 352, 1
print(y2_test.shape)        # 152, 1
'''
# reshape

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]))

print('x1_train : ', x1_train.shape)         # 352, 5
print('x1_test.shape : ', x1_test.shape)     #  152, 5

print('x2_train : ', x2_train.shape) # (352, 25)
print('x2_test.shape : ', x2_test.shape)  # (152, 25)


# 데이터 전처리

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(x1_train)

x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

scaler2 = StandardScaler()
scaler2.fit(x2_train)

x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print(x2_train_scaled[0, :])
'''

# 모델 구성

from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(5, ))
dense1 = Dense(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(25, ))
dense2 = Dense(64)(input2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
dense2 = Dense(32)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
output3 = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output3)

# 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)
model.fit([x1_train, x2_train], y1_train, validation_split=0.2, verbose=1 , batch_size=1, epochs=300, callbacks=[es]) 


# 평가 / 예측
loss, mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)

print("loss : ", loss)
print("mse : ", mse)

print('x1_test :', x1_test)
print('x2_test : ', x2_test)

print('x1_test.shape : ', x1_test.shape)    # 152, 5
print('x2_test.shape : ', x2_test.shape)    # 152, 25      


y1_pred = model.predict([x1_test, x2_test])

print('y1_pred : ',y1_pred)

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])


print('x1.shape : ', x1.shape)     # 504, 5
print('x2.shape : ', x2.shape)     # 504, 25

print('x1[-1]', x1[-1])
print('x2[-1]', x2[-1])

y1_predict = model.predict([[x1[-1]], [x2[-1]]])
print('y1_predict : ', y1_predict)


#y1_predict :  [[51268.457]]
