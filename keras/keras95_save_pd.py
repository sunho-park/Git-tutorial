import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", 
                    index_col=None,  
                    header=0, sep=',') # header = None 첫번째 행부터 자료로 인식, header = 0 첫번째행은 데이터로 인식하지않는다

print(datasets)

print(datasets.head()) # 위에서 다섯줄정도만 보임
print(datasets.tail()) # 아래에서 다섯줄

print("=====================================")
print(datasets.values)

aaa = datasets.values

print(type(aaa))

# 넘파이로 저장하시오
# (150, 4)
np.save('./data/csv/iris.npy', arr=datasets)