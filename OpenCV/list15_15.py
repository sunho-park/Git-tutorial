import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(len(img[i][j])):
            img[i][j][k] = 255 - img[i][j][k]

cv2.imshow("sample", img)

cv2.imwrite("list15_15.jpg", img)