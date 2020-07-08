# 이상치 처리
# 0으로 처리
# nan처리후 보간법

import numpy as np
'''
# a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100], [10, 20, 30, 4000, 50, 60, 7000, 80, 90, 100]])
a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])


def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
        
# a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])
# a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100], [10, 20, 30, 4000, 50, 60, 7000, 80, 90, 100]])
# a = a.transpose()
print(a.shape) # (2, 10)

b = outliers(a)
print("이상치의 위치 : ", b)

'''
# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하시오.
# 파일명 : m36_outliers2.py
# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구하시오.
#------------------------------------------------------------------
# numpy
def outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return outliers

a2 = np.array([[1, 5000], [200, 8], [2, 4], [3, 7], [8, 2]])
print(a2.shape) # (5, 2) 
print(a2.shape[1]) # 2
print(a2)
# [[   1 5000]
#  [ 200    8]
#  [   2    4]
#  [   3    7]
#  [   8    2]]

b2 = outliers(a2)
print(b2)
# [(array([1], dtype=int64),), (array([0], dtype=int64),)]
'''
# -------------------------------------------------------------------
# pandas
import pandas as pd
def outliers(data_out):
        quartile_1 = data_out.quantile(.25)
        quartile_3 = data_out.quantile(.75)
        print("1사 분위 : ", quartile_1)                                       
        print("3사 분위 : ", quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out > upper_bound) | (data_out < lower_bound))
         
a3 = pd.DataFrame({'a' : [1, 3, 5, 200, 100, 8],
                    'b' : [300, 100, 6, 8, 2, 3]})
print(a3)
#      a    b
# 0    1  300
# 1    3  100
# 2    5    6
# 3  200    8
# 4  100    2
# 5    8    3

b3 = outliers(a3)
print(b3)
# 1사 분위 :  a    3.50
# b    3.75
# Name: 0.25, dtype: float64
# 3사 분위 :  a    77.0
# b    77.0
# Name: 0.75, dtype: float64
# (array([0, 3], dtype=int64), array([1, 0], dtype=int64))'''
