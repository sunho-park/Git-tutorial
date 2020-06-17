import matplotlib.pyplot as plt
import cv2 # OpenCV 사용하기
import numpy as np

# 사진 데이터 읽어 들이기
photos = np.load('myproject/photos.npz')
x = photos['x']
img = x[12] # 사진 하나 선택하기

plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    # 회전하기
    center = (16, 16) # 회전 중심
    angle = i*5       # 각도 지정
    scale = 1.0       # 확대 비율 지정
    mtx = cv2.getRotationMatrix2D(center, angle, scale) # 필요한 행렬 얻기 (회전의 중심, 회전의 각도, 배율) ex( , 180, 2) 180도 설정 2배확대
    img2 = cv2.warpAffine(img, mtx, (32, 32))           # affin 변환 (변환하려는 이미지, 위에서 생성한 행렬, 사이즈)
    # 화면에 출력하기
    plt.imshow(img2)
plt.show()



