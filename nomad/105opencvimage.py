import cv2
import numpy as np

print("OpenCV version")
print(cv2.__version__)

img = cv2.imread(r'C:\Users\bitcamp\Desktop\opencv_dnn_202005\nomadProgramerIcon.png')
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {} ".format(img.shape[2]))

(height, width) = img.shape[:2]
center = (width//2 , height // 2)

print(height, width)        # 653, 653

cv2.imshow("madProgramer", img)

move = np.float32([[1, 0, 100], [0, 1, 100]])
print(move) 
print(move.shape)   # (2, 3)
# [[  1.   0. 100.]
#  [  0.   1. 100.]]
moved = cv2.warpAffine(img, move, (width, height))

cv2.imshow("Moved down: +, up: - and right: +, left- ", moved)

rotate = cv2.getRotationMatrix2D(center, 90, 1.0)                              # 회전의 중심, 각도, 배율
rotated = cv2.warpAffine(img, rotate, (width, height))                           # 이미지, 위에서 생성한 행렬, 사이즈
# cv2.imshow("Rotated clockwise degrees", rotated)

ratio = 200.0 / width
dimension = (200, int(height*ratio))

resized = cv2.resize(img, dimension, interpolation = cv2.INTER_CUBIC)
cv2.imshow("Resized", resized)

flipped = cv2.flip(img, 1)
cv2.imshow("Flipped Horizontal 1, Vertical 0, both -1", flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()