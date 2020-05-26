from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(2, (3, 3), input_shape=(6, 6, 3)))    
# x = (10000, 10, 10, 1) 가로세로 2,2 로 자르겟다. 
# (10, 10, 1)(흑백), (10, 10 ,3)(칼라)
model.summary()
