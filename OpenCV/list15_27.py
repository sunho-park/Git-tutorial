import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

# 필터 정의 
filt = np.array([[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]], np.uint8) # uint 데이터형

# 팽창 처리
my_img = cv2.dilate(img, filt)
cv2.imshow("sample", my_img)
cv2.imwrite("list15_27_dilate.jpg", my_img)

# 침식 처리
my_img = cv2.erode(img, filt)
cv2.imshow("list15_27_erode.jpg", my_img)
