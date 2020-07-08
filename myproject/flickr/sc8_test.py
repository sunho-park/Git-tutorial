import sc4_cnn_model as cnn
import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

target_image = 'test_sushi.jpg'

im_rows = 32 # 이미지의 세로 픽셀 수 
im_cols = 32 # 이미지의 가로 픽셀 수
im_color = 3 # 이미지의 색공간
in_shape = (im_rows, im_cols, im_color)
nb_classes = 3

LABELS = ["초밥", "샐러드", "김밥"]
CALORIES = [588, 118, 441] # 김밥 250g 당 441 kal

# CNN 모델 읽어 들이기
model = cnn.get_model(in_shape, nb_classes)
model.load_weights('./flickr/photos-model-light.hdf5')

def check_photo(path):
    # 이미지 읽어들이기
    img = Image.open(path)
    img = img.convert("RGB") # 색공간 변환하기
    img = img.resize((im_cols, im_rows)) # 크기 변환하기
    plt.show(img)
    plt.show()
    # 데이터로 변환하기
    x = np.asarray(img)
    x = x.reshape(-1, im_rows, im_cols, im_color)/ 255

    # 예측하기
    pre = model.predict([x])[0]
    idx = pre.argmax()
    per = int(pre[idx]*100)
    return (idx, per)

def check_photo_str(path):
    idx, per = check_photo(path)
    # 응답하기
    print("이 사진은", LABELS[idx], "로(으로), 칼로리는", CALORIES[idx], "kcal입니다.")
    print("가능성은", per, "%입니다.")

if __name__=='__main__':
    check_photo_str('test_sushi.jpg')
    check_photo_str('test_salad.jpg')