
import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

mask = cv2.imread("./OpenCV/tokyogul.png", 0)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

retval, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
my_img = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_21.jpg", my_img)