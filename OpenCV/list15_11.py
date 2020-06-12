import numpy as np
import cv2

img = cv2.imread("./OpenCV/sample.jpg")


# warpAffine() 함수 사용에 필요한 행렬을 만듭니다.
# 첫 번째 인수는 회전의 중심입니다(야기서는 이미지의 중심을 설정).
# 두 번째 인수는 회전 각도입니다(여기서는 180도 설정).
# 세 번째 인수는 배율입니다(여기서는 2배 확대로 설정).

mat = cv2.getRotationMatrix2D(tuple(np.array(img.shape[:2])/2), 180, 2.0)

# 아핀 변환을 합니다.
# 첫 번째 인수는 변환하려는 이미지입니다.
# 두 번째 인수는 위에서 생성한 행렬(mat)입니다.
# 세 번째 인수는 사이즈입니다.

my_img = cv2.warpAffine(img, mat, img.shape[:2])

cv2.imshow("sample", my_img)

cv2.imwrite("list15_11.jpg", my_img)
