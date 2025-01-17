import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

hite_df = pd.read_csv("./data/csv/하이트 주가.csv", header=0, encoding='cp949', index_col=0, sep=',')
samsung_df = pd.read_csv("./data/csv/삼성전자 주가.csv", header=0, encoding='cp949', index_col=0, sep=',')

hite_df = hite_df[:509]
hite_df = hite_df.fillna(0)

print(hite_df)
print("하이트 주가 : ", hite_df)

samsung_df = samsung_df[:509]
print("삼성 주가 : ", samsung_df)


#  string 데이터를 수치형 데이터로 변환

print(len(hite_df.index))               # 509
print(range(len(hite_df.index)))        # range(0, 509)
                                        # 0~508
print(len(hite_df.iloc[508]))             # 5
print(range(len(hite_df.iloc[508])))      #range(0,5)

# df.iloc["행번호 리스트", "열번호 리스트"]

for i in range(len(hite_df.index)):                # range(0, 509)
    for j in range(len(hite_df.iloc[i])):          # range(0, 5)  
        hite_df.iloc[i, j] = str(hite_df.iloc[i, j]).replace(',','')
        hite_df.iloc[i, j] = int(hite_df.iloc[i, j])


for i in range(len(samsung_df.index)):
    samsung_df.iloc[i,0] = str(samsung_df.iloc[i, 0]).replace(',','')
    samsung_df.iloc[i,0] = str(samsung_df.iloc[i, 0])

# 오름차순

hite_df = hite_df.sort_values(['일자'], ascending=[True] )
samsung_df=samsung_df.sort_values(['일자'], ascending=[True])

print('hite_df : ', hite_df)
print('hite_df : ', hite_df)


hite_df=hite_df.values
samsung_df=samsung_df.values

np.save('./data/csv/하이트 주가.npy', arr=hite_df)
np.save('./data/csv/삼성전자 주가.npy', arr=samsung_df)

hite = np.load('./data/csv/하이트 주가.npy', allow_pickle=True)
samsung = np.load('./data/csv/삼성전자 주가.npy', allow_pickle=True)


print('hite.shape : ', hite.shape)           # (509, 5)
print('samsung.shape : ', samsung.shape)     # (509, 1)

