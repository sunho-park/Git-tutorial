# 미리 OpenCV 폴더에 15장의 샘플 mask.png 파일을 넣어두세요

import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")

# 두 번째 인수로 0을 지정하면 채널수가 1인 이미지로 변환해서 읽습니다.
mask = cv2.imread("./OpenCV/tokyogul.png", 0)

# 원래 이미지와 같은 크기로 리사이즈 합니다.
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# 세 번째 인수로 마스크용 이미지를 선택합니다.
my_img = cv2.bitwise_and(img, img, mask= mask)

cv2.imshow("sample", my_img)
cv2.imwrite("list15_20.jpg", my_img)
# cv2.imwrite("list15_20.jpg", mask)