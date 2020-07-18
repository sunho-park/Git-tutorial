import sys
import cv2

src = cv2.imread('./OpenCV/namecard2.jpg')

if src is None:
    print('image load failed')
    sys.exit()


# src = cv2.resize(src, (640, 370))
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print(th)
cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)
cv2.waitKey()
cv2.destroyAllWindows()

