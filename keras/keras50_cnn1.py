from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1)))    #(9, 9, 10) filter 10 이 (2, 2)로 자른것의 장수
# x = (10000, 10, 10, 1) 가로세로 2,2 로 자르겟다. 
# (10, 10, 1)(흑백), (10, 10 ,3)(칼라)
model.add(Conv2D(7, (3, 3)))                               #(7, 7, 7)  
model.add(Conv2D(5, (2, 2), padding='same'))               #(7, 7, 5)                     
model.add(Conv2D(5, (2, 2)))                               #(6, 6, 5)                     
#model.add(Conv2D(5, (2, 2), strides=2))                   #(3, 3, 5)                 
#model.add(Conv2D(5, (2, 2), strides=2, padding='same'))    #(3, 3, 5)                 
model.add(MaxPooling2D(pool_size=2))                        #(3, 3, 5)  pool_size : 수직, 수평 축소 비율을 지정합니다. (2, 2)이면 출력 영상 크기는 입력 영상 크기의 반으로 줄어듭니다.

model.add(Flatten())

model.add(Dense(1))      # before Flatten (None, 3, 3, 1) 

model.summary()




# https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
# https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/