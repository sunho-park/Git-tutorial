import numpy as np
import cv2

# img = cv2.imread("./OpenCV/sample.jpg")
# cv2.imshow("sample", img)

# 이미지 크기 결정
img_size = (512, 512)

# 이미지 정보를 가지는 행렬을 만듭니다.
# 빨간색 이미지이므로 각 요소가 [0, 0, 255]인 512 * 512

# 행렬이 전치되는 점에 주의합니다.
# 이미지 데이터의 각 요소는 0~255 값만 지정가능.
# 이를 명시하기 위해 dtype 옵션으로 데이터형을 지정합니다.

my_img = np.array([[[0, 0, 255] for _ in range(img_size[1])] for _ in range(img_size[0])], dtype="uint8")

cv2.imshow("sample", my_img)

cv2.imwrite("my_red_img.jpg", my_img)
