import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a= np.array(range(1,11))
print(a)
size = 5

def split_x(seq, size): # seq 전체 데이터, size
    aaa= []

    for i in range(len(seq) - size + 1):    #range(6)    # 0, 1, 2, 3, 4, 5
        print("i : ", i)
        subset = seq[i : (i+size)]
        print("subset : ", subset)
        aaa.append([item for item in subset])
        
   # print(type(aaa)) list
    return np.array(aaa)

dataset =split_x(a, size)
print(dataset)

print("========================================================")


