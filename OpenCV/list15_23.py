
import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")


# 첫 번째 인수는 원본 이미지입니다.
# 두 번째 인수는 nxn(마스크 크기)의 n 값입니다.(n은 홀수)
# 세 번째 인수는 x축 방향의 편차(일반적으로 0으로 지정)입니다.
my_img = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_23(5, 5).jpg", my_img)

my_img = cv2.GaussianBlur(img, (21, 21), 0)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_23.jpg", my_img)