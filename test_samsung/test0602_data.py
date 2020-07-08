import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv',  # index_col=None 첫번째 컬럼이 새롭게 생성되면서 밑에 행을 순서대로 0, 1, 2, 3, 4 만들어짐
                        index_col=0,             # index_col= 0 0번째 컬럼을 인덱스로 사용하겠다. 
                        header=0,                # header = None 첫번째 행부터 자료로 인식, header = 0 첫번째행은 데이터로 인식하지않는다
                        sep=',', 
                        encoding='CP949')        # CP949, utf8, utf16 이중에 넣어보고 안꺠지는 것으로 시행착오


# 앙상블 가중치가 50 : 50,다 빼버리고 시가만 넣는 것도 

hite = pd.read_csv('./data/csv/hite.csv', 
                    index_col=0, 
                    header=0, 
                    sep=',', 
                    encoding='CP949')

print(samsung.head())
print(hite.head())
print(samsung.shape)     # (700, 1)
print(hite.shape)        # (720, 5)


# Non 제거1
samsung = samsung.dropna(axis=0) # axis = 0행
print(samsung.shape)      #(509, 1)

hite = hite.fillna(method='bfill') # bfill : backfill 뒤에 있는 데이터로 채움. 차이가 얼마 없어서 
hite = hite.dropna(axis=0)         # ffill : frontfill? 앞에있는 데이터로 채움.
print(hite)                        # fillna(0) : 0 으로 채우겠다.


# None 제거 2
hite = hite[0:509]
hite.loc["2020-06-02", '고가':'거래량'] = ['10', '20', '30', '40']   # df.loc ["인덱스리스트", "컬럼리스트"]
#hite.iloc[0, 1:5] = [10, 20, 30, 40]                               # df.iloc ["행번호리스트", "열번호리스트"]
print(hite)


# None 제거 3
# 예측모델을 이용해서 고가, 저가, 종가, 거래량의 Nan 을 구함
############################################################################

# 삼성과 하이트의 정렬을 오름차순으로 변경. 내림차순 ascending=['False'], decending
samsung = samsung.sort_values(['일자'], ascending=['True'])  
hite = hite.sort_values(['일자'], ascending=['True'])  


# 콤마제거, 문자를 정수로 형변환   / nan 값에 ''없이 숫자만 넣으면 float 형으로 떠서 오류 발생
for i in range(len(samsung.index)):   # '37,000' -> 37000 / str -> int
    samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',', ''))
# print(samsung)
# print(type(samsung.iloc[0,0])) / <class 'int'>

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',' ,''))
print(hite)
print(type(hite.iloc[1,1]))

# print(samsung.shape)   (509, 1)
# print(hite.shape)      (509, 5) 

# numpy로 저장  / pandas.core.frame.DataFrame -> numpy.ndarray
samsung = samsung.values
hite = hite.values

# 저장

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)

