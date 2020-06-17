import matplotlib.pyplot as plt
import cv2 # OpenCV 사용하기
import numpy as np

# 사진 데이터 읽어 들이기
photos = np.load('myproject/photos.npz')
x = photos['x']
img = x[12] # 사진 하나 선택하기

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i+1)
    # 회전하기
    center = (16, 16) # 회전 중심
    angle = i*5-30      # 각도 지정
    scale = 1.0       # 확대 비율 지정
    mtx = cv2.getRotationMatrix2D(center, angle, scale) # 필요한 행렬 얻기 (회전의 중심, 회전의 각도, 배율) ex( , 180, 2) 180도 설정 2배확대
    img2 = cv2.warpAffine(img, mtx, (32, 32))           # affin 변환 (변환하려는 이미지, 위에서 생성한 행렬, 사이즈)
    # 화면에 출력하기
    plt.imshow(img2)
plt.show()



'''
#######################################################################
## OpenCV 이용하여 학습전용 데이터수 늘리기 ##
x_new = []
for i, xi in enumerate(x[12]): # i : index, xi : value
    for ang in range(-30, 30, 5): # [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25] = 12개
        # 회전 시키기
        center = (16, 16)                                   # 회전 중심
        matrix = cv2.getRotationMatrix2D(center, ang, 1.0)  # affin에 필요한 행렬 만들기 (회전의 중심, 회전의 각도, 배율)
        xi2 = cv2.warpAffine(xi, matrix, (32, 32))          # affin 변환 (변환하려는 이미지, 위에서 생성한 행렬, 사이즈)
        x_new.append(xi2)
        # 좌우 반전  [12] * 2 =24
        xi3 = cv2.flip(xi2, i)                             # flip(이미지 데이터, 반전방향)
        x_new.append(xi3)
        plt.imshow(x_new)
    plt.show(x_new)

'''