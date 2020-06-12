import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")
# 노이즈 제거
my_img = cv2.fastNlMeansDenoisingColored(img)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_26.jpg", my_img)