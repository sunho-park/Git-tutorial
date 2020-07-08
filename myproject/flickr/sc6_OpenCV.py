import matplotlib.pyplot as plt
import cv2 # OpenCV 사용하기
import numpy as np

# 사진 데이터 읽어 들이기
photos = np.load('flickr/photos.npz')
x = photos['x']
img = x[12] # 사진 하나 선택하기

plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    # 회전하기
    center = (16, 16) # 회전 중심
    angle = i*5       # 각도 지정
    scale = 1.0       # 확대 비율 지정
    mtx = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(img, mtx, (32, 32))
    # 화면에 출력하기
    plt.imshow(img2)
plt.show()



