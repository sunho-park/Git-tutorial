import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

retval, my_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_18.jpg", my_img)