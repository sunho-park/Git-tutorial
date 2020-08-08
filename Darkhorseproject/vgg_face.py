import keras
import tensorflow as tf
import cv2
# 현재 버전
print(tf.__version__)       # 2.3.0
print(keras.__version__)    # 2.3.1
print(cv2.__version__)      # 4.2.0

# 요구버전 
# 1.9.0
# 2.2.0
# 3.4.4

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()
#you can download the pretrained weights from the following link 
#https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
#or you can find the detailed documentation https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

from tensorflow.keras.models import model_from_json
model.load_weights('./Darkhorseproject/matrix/vgg_face_weights.h5')

#
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)   # 차원이 늘어남
    img = preprocess_input(img)         # 이미지를 -1~1 정규화
    return img

#
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return a / (np.sqrt(b) * np.sqrt(c))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.60

def verifyFace(img1, img2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image('./Darkhorseproject/matrix/%s' % (img1)))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image('./Darkhorseproject/matrix/%s' % (img2)))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine 유사도: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity > epsilon):
        print("같은 사람입니다.")
    else:
        print("다른 사람입니다.!")
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img('./Darkhorseproject/matrix/%s' % (img1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img('./Darkhorseproject/matrix/%s' % (img2)))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    print("-----------------------------------------")
    
#6
verifyFace("amanda.jpg", "amanda1.jpg")
verifyFace("amanda.jpg", "jenifer.jpg")
verifyFace("amanda1.jpg", "jenifer.jpg")
verifyFace("ann.jpg", "jenifer.jpg")
verifyFace("ann.jpg", "amanda1.jpg")
verifyFace("amanda.jpg", "ann.jpg")
#8
verifyFace("leo.jpg", "leo2.jpg")
verifyFace("leo.jpg", "tom.jpg")
verifyFace("leo2.jpg", "tom.jpg")
verifyFace("will.jpg", "wpr.jpg")
verifyFace("wpr.jpg", "wpr2.jpg")
verifyFace("rb.jpg", "rb2.jpg")
verifyFace("rb.jpg", "wpr.jpg")
verifyFace("rb.jpg", "tom.jpg")
#6
verifyFace("nicolas.jpg", "nicolas2.jpg")
verifyFace("nicolas.jpg", "tom.jpg")
verifyFace("nicolas.jpg", "rb.jpg")
verifyFace("nicolas.jpg", "wpr.jpg")
verifyFace("nicolas.jpg", "wpr2.jpg")
verifyFace("nicolas.jpg", "leo.jpg")
#5
verifyFace("nicolas2.jpg", "tom.jpg")
verifyFace("nicolas2.jpg", "rb.jpg")
verifyFace("nicolas2.jpg", "wpr.jpg")
verifyFace("nicolas2.jpg", "wpr2.jpg")
verifyFace("nicolas2.jpg", "leo.jpg")

# # 25 중에 4개 틀림

verifyFace("hb.jpg", "hb1.jpg")
verifyFace("hb.jpg", "jdg.jpg")
verifyFace("hb.jpg", "jh.jpg")
verifyFace("hb.jpg", "ssh.jpg")
verifyFace("hb.jpg", "lsw.jpg")
verifyFace("hb.jpg", "ljj.jpg")
#8
verifyFace("hb1.jpg", "jdg.jpg")
verifyFace("hb1.jpg", "jh.jpg")
verifyFace("hb1.jpg", "ssh.jpg")
verifyFace("hb1.jpg", "lsw.jpg")
verifyFace("hb1.jpg", "ljj.jpg")
verifyFace("hb1.jpg", "ssh1.jpg")
verifyFace("hb1.jpg", "jh1.jpg")
verifyFace("hb1.jpg", "lsw1.jpg")
#6
verifyFace("ksh.jpg", "ksh2.jpg")
verifyFace("ksh.jpg", "jdg.jpg")
verifyFace("ksh.jpg", "jh.jpg")
verifyFace("ksh.jpg", "ssh.jpg")
verifyFace("ksh.jpg", "lsw.jpg")
verifyFace("ksh.jpg", "ljj.jpg")
#5
verifyFace("ksh2.jpg", "ljj1.jpg")
verifyFace("ksh2.jpg", "ssh.jpg")
verifyFace("ksh2.jpg", "lsw1.jpg")
verifyFace("ksh2.jpg", "jdj.jpg")

# 25개중 11개

# Cosine 유사도:  0.39293617010116577
# Euclidean distance:  99.559845


verifyFace("jomask.jpg", "jomask1.jpg")
verifyFace("jo.jpg", "jomask.jpg")
verifyFace("jo.jpg", "jomask1.jpg")

verifyFace("lee.jpg", "leemask.jpg")
verifyFace("lee.jpg", "leemask1.jpg")
verifyFace("leemask1.jpg", "leemask.jpg")


# verifyFace("m.jpg", "m2.jpg")
# verifyFace("m.jpg", "m3.jpg")
# verifyFace("m.jpg", "neo.jpg")
# verifyFace("m.jpg", "test1.jpg")

# verifyFace("maskm.jpg", "maskm1.jpg")
# verifyFace("maskm.jpg", "maskm2.jpg")
# verifyFace("maskm1.jpg", "maskm2.jpg")


verifyFace("sin.jpg", "sin2.jpg")
verifyFace("sin.jpg", "sin3.jpg")
verifyFace("sin.jpg", "sin1.jpg")
verifyFace("sin.jpg", "sin4.jpg")
verifyFace("sin4.jpg", "sin5.jpg")


# test
verifyFace("11.jpg", "5.jpg")
verifyFace("22.jpg", "5.jpg")
verifyFace("33.jpg", "5.jpg")
verifyFace("44.jpg", "5.jpg")

verifyFace("11.jpg", "66.jpg")
verifyFace("22.jpg", "66.jpg")
verifyFace("33.jpg", "66.jpg")
verifyFace("44.jpg", "66.jpg")