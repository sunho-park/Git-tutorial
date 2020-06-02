import numpy as np
import pandas as pd

df1 = pd.read_csv("./data/kospi200/data/kospi200.csv", index_col=0,
                    header=0, encoding='cp949', sep=',')

print("kospi200 : ", df1)
print(df1.shape)
print(type(df1))

df2 = pd.read_csv("./data/kospi200/data/samsung.csv", index_col=0,
                    header=0, encoding='cp949', sep=',')

print("samsung : ", df2)
print(df2.shape)
print(type(df2))
# kospi 200의 거래량

for i in range(len(df1.index)):            # 거래량 str -> int 변경
    df1.iloc[i,4] = int(df1.iloc[i,4].replace(',',''))

# 삼성전자의 모든 데이터

                                           #모든 str -> int 변경
for i in range(len(df2.index)):  
    for j in range(len(df2.iloc[i])):
        df2.iloc[i, j] = int(df2.iloc[i, j].replace(',', ''))         


# 데이터를 오름차순

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])

print("kospi200 : ", df1)
print("samsung : ", df2)

df1 = df1.values
df2 = df2.values
print(df1)
print(df2)
print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./data/kospi200/data/kospi200.npy', arr=df1)
np.save('./data/kospi200/data/samsung.npy', arr=df2)





    

