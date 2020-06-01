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


####################################################################
'''
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                #10         4
def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):  # 10
        end_number = i + time_steps           #            0 + 4, 1 + 4, 2 + 4, 3 + 4, 4 + 4, ~~~~, 9+4
        if end_number > len(dataset) -1:      #end number  4 > 9, 5 > 9, 6 > 9, 7 > 9, 8 > 9 
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number] # i = 0, x = [0, 1, 2, 3] y =[4]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 4)
print(x, "\n", y)
'''
###########################################################################
'''
dataset = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def split_xy2(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):                     #          10
        x_end_number = i +time_steps                  #          x_end_number = 0 + 4    /  1 + 4 / 2 + 4 / 3 + 4
        y_end_number = x_end_number + y_column        # 추가    # y_end_number = 4 + 2= 6 /  5 + 2= 7 / 6 + 2 = 8 / 7 + 2 = 9/
        
        if y_end_number > len(dataset):               # 수정    # 8 > 10   / 9 > 10
            break
        tmp_x = dataset[ i : x_end_number]                      # [0 : 4] -> [1, 2, 3, 4]
        tmp_y = dataset[x_end_number : y_end_number]  # 수정    # [4 : 6] -> [5, 6]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 4
y_column = 2
x, y = split_xy2(dataset, time_steps, y_column)
print(x, "\n", y)
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)
'''