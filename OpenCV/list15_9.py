import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

my_img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

cv2.imshow("sample", my_img)

cv2.imwrite("list15_9.jpg", my_img)