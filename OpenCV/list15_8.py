import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")
size = img.shape

# 이미지를 나타내는 행렬의 일부를 꺼내면 그것이 트리밍이 됩니다.
#  n등분하려면 가로와 세로 크기를 나눕니다.

my_img = img[: size[0]//2, size[1]//3]

# 여기에서는 원래의 배율을 유지하면서 폭과 높이를 각각 2배로 합니다. 크기를 지정할 때는 (폭, 높이) 순서라는 점을 유의하세요
my_img = cv2.resize(my_img, (my_img.shape[1]*2, my_img.shape[0]*2))

cv2.imshow("sample", my_img)

cv2.imwrite("list15_8.jpg", my_img)