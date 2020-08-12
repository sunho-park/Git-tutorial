from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
 
import os
import matplotlib.pyplot as plt
import numpy as np
import math

############################################################################

# 이미지들을 Numpy 형식으로 변환하기
import numpy as np
from PIL import Image              # Python Image Library
import glob, os, random 

outfile = "./myproject/resnet/face_photos.npz" # 저장할 파일 이름

max_photo = 200 # 사용할 장 수
photo_size = 32 # 이미지 크기

x = []          # 이미지 데이터
y = []          # 레이블 데이터

def main():
    # 디렉터리 읽어 들이기                    (경로, 레이블) 
    glob_files("./myproject/resnet/", 0) # Hamburger폴더에 있는 모든 jpg파일 0 레이블
    glob_files("./myproject/resnet/", 1)
    glob_files("./myproject/resnet/", 2)    
    glob_files("./myproject/resnet/", 3)    
    glob_files("./myproject/resnet/", 4)    
    glob_files("./myproject/resnet/", 5)    
    glob_files("./myproject/resnet/", 6)    
    
    np.savez(outfile, x=x, y=y)             # 파일로 저장하기
    print(outfile, len(x), "장 저장했습니다.")           # 600 장 저장했습니다.
  
def glob_files(path, label):            # path 내부의 이미지 읽어 들이기    
    files = glob.glob(path+ "/*.jpg")   # 폴더안의 파일들의 목록을 불러옴ex)glob('*.txt')
    random.shuffle(files)
    # for문으로 모든 파일 색공간과 크기 변경
    num = 0
    for f in files:                                 # 200개의 이미지 3번
        if num >= max_photo:break                   # 200개가 넘으면 멈춤
        num += 1
        # 이미지 파일 읽어 들이기 Image모듈 open함수 사용
        img = Image.open(f)
        img = img.convert("RGB")                     # 색공간 변환하기
        img = img.resize((photo_size, photo_size))   # (32, 32)로 크기 변경하기
        img = np.asarray(img)                        # modifying img itself
                                                     # array : modifying img a copy
        x.append(img)
        y.append(label)
# 엔트리 포인트 또는 메인이므로 main() 함수 실행    
if __name__ == '__main__':  
    main()

############################################################################

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join('./')
var_dir = os.path.join('./')

# number of classes
K = 1000
 
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
 
 
def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
     
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)          

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)           
    
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)                  
        
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
         
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x

def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)                  
          
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)           
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x

def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
                         
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='same')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
           
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x


x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)

x = GlobalAveragePooling2D()(x)

output_tensor = Dense(K, activation='softmax')(x)
 
resnet34 = Model(input_tensor, output_tensor)
resnet34.summary()


