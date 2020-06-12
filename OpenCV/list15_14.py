import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

# 색공간을 변환합니다.

# my_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
my_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow("sample", my_img)

cv2.imwrite("list15_14_GRAY.jpg", my_img)