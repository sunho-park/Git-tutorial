import sc4_cnn_model
import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

rows = 32 # 이미지의 세로 픽셀 수 
cols = 32 # 이미지의 가로 픽셀 수
color = 3 # 이미지의 색공간
in_shape = (rows, cols, color)
out_y = 3

LABELS = ["햄버거", "샐러드", "김밥"]
CALORIES = [550, 118, 441] 

# CNN 모델 읽어 들이기
model = sc4_cnn_model.get_model(in_shape, out_y)
model.load_weights('./myproject/photos-model-light.hdf5')

def input_photo(path):                        # path = 'test1.jpg'
    # 이미지 읽어들이기
    img = Image.open(path)
    img = img.convert("RGB")                  # 색공간 변환하기
    img = img.resize((cols, rows))            # (32, 32)로 크기 변환하기
  
    x = np.asarray(img)
    x = x.reshape(-1, rows, cols, color)/ 255  # x.shape : (1, 32, 32, 3)   

    # 예측하기
    predict = model.predict(x).reshape(3, )  # (1, 3)을 (3, ) [9.996532e-01 3.467400e-04 6.208235e-11]                                                                    
    index = predict.argmax()                 # index = [햄버거 0, 샐러드 1, 김밥 2] 중 가장 큰 값 인덱스번호 출력
    per = int(predict[index]*100)            # predict[index], predict[0]*100 후에 정수형으로 바꿔서 per값 출력 
    return (index, per)                      # predict[0]=9.996532e-01/predict[1]=3.467400e-04/predict[2]=6.208235e-11
    
def check_photo(path):
    index, per = input_photo(path) # 음식과 칼로리 판정하기 # LABELS[햄버거, 샐러드, 김밥], CALORIES = [550, 118, 441]
    print("이 사진은", LABELS[index], "로(으로), 칼로리는", CALORIES[index], "kcal입니다.")   
    print("가능성은", per, "%입니다.")
    print("------------------------")

if __name__=='__main__':
    check_photo('./myproject/test1.jpg')
    check_photo('./myproject/test2.jpg')
    check_photo('./myproject/test3.jpg')
    check_photo('./myproject/test4.jpg')
    check_photo('./myproject/test5.jpg')
    check_photo('./myproject/test6.jpg')
    check_photo('./myproject/test7.jpg')
    