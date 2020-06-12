import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

# 이미지를 x축을 중심으로 반전시키세요

my_img = cv2.flip(img, 0)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_12.jpg", my_img)