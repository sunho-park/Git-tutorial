import sc4_cnn_model as cnn
import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# target_image = 'test_2.jpg'

rows = 32 # 이미지의 세로 픽셀 수 
cols = 32 # 이미지의 가로 픽셀 수
color = 3 # 이미지의 색공간
in_shape = (rows, cols, color)
out_y = 3

LABELS = ["햄버거", "샐러드", "김밥"]
CALORIES = [550, 118, 441] 

# CNN 모델 읽어 들이기
model = cnn.get_model(in_shape, out_y)
model.load_weights('./myproject/photos-model-light.hdf5')

def input_photo(path):                        # path = 'test1.jpg'
    # 이미지 읽어들이기
    img = Image.open(path)
    img = img.convert("RGB")                  # 색공간 변환하기
    img = img.resize((cols, rows))            # (32, 32)로 크기 변환하기
  
    # 데이터로 변환하기
    x = np.asarray(img)
    x = x.reshape(-1, rows, cols, color)/ 255 # 정규화 / x.shape(1, 32, 32, 3)   
  
    # 예측하기
    predict = model.predict([x])[0]               # (1, 3)을 스칼라로 바꿔주기 
    #print(model.predict([x]))                    # [[9.99989390e-01 1.00640455e-05 6.33899845e-07]]
    #print("predict : ", predict)                 # [9.99989390e-01 1.00640455e-05 1.00640455e-05]
    #print("predict.shape : ", predict.shape)     # (3, )

    index = predict.argmax()                      # index = [햄버거 0, 샐러드 1, 김밥 2] 중 가장 큰 값 인덱스번호 출력
    per = int(predict[index]*100)                 # predict[index], predict[0]*100 후에 정수형으로 바꿔서 정수부분만 출력 
    #print('predict[index] : ', predict[index])   # predict[0] = 9.99989390e-01 / predict[1] = 1.00640455e-05 /predict[2]=1.00640455e-05
    return (index, per)
    
def check_photo(path):
    index, per = input_photo(path)
    # 음식과 칼로리 판정하기                       # LABELS[햄버거, 샐러드, 김밥], CALORIES = [550, 118, 441]
    try:
        print("이 사진은", LABELS[index], "로(으로), 칼로리는", CALORIES[index], "kcal입니다.")   
        print("가능성은", per, "%입니다.")
    except:
        pass

print(check_photo('test1.jpg'))
print(check_photo('test2.jpg'))
print(check_photo('test3.jpg'))
print(check_photo('test4.jpg'))
print(check_photo('test5.jpg'))
print(check_photo('test6.jpg'))
print(check_photo('test7.jpg'))

'''
if __name__=='__main__':
    check_photo('test1.jpg')
    check_photo('test2.jpg')
    check_photo('test3.jpg')
    check_photo('test4.jpg')
    check_photo('test5.jpg')
    check_photo('test6.jpg')
    check_photo('test7.jpg')
'''