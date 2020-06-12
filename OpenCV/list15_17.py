import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

# 첫 번째 인수는 처리하는 이미지 입니다.
# 두 번째 인수는 임곗값입니다.
# 세 번째 인수는 최댓값(max value)입니다.
# 네 번째 인수는 THRESH_BINARY, THRESH_BINARY_INV, THRESH_TOZERO, THRESH_TRUNC, THRESH_TOZERO_INV 중 하나입니다.

# THRESH_BINARY : 픽셀값이 임곗값을 초과하는 경우 해당 픽셀을 maxValue 로 하고, 그외의 경우에는 0(검은색)으로 합니다.
# THRESH_BINARY_INV : 픽셀값이 임곗값을 초과하는 경우 0으로 설정하고, 그 외의 경우에는 maxValue로 합니다.
# THRESH_TOZERO : 픽셀값이 임곗값을 초과하는 경우 변경하지 않고, 그 외의 경우에는 0으로 설정합니다.
# THRESH_TRUNC : 픽셀값이 임곗값을 초과하는 경우 임곗값으로 설정하고, 그 외의 경우에는 변경하지 않습니다.

# 임곗값을 75로, 최댓값을 255로 하여 THRESH_TOZERO를 적용합니다.
# 임곗값도 반환되므로 retval로 돌려받습니다.

retval, my_img = cv2.threshold(img, 75, 255, cv2.THRESH_TOZERO)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_17.jpg", my_img)