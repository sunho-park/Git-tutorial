import sys
import numpy as np
import cv2
import pytesseract

def reorderPts(pts):
    idx = np.lexsort((pts[:, 1], pts[:, 0])) # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    pts = pts[idx] # x좌료로 정렬

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

filename = './OpenCV/namecard.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]

src = cv2.imread(filename)

dw, dh = 720, 400
srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
dstQuad = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]], np.float32)
dst = np.zeros((dh, dw), np.uint8)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)

    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue

    srcQuad = reorderPts(approx.reshape(4, 2).astype(np.float32))

    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(dst_rgb, lang='Hangul+eng'))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitkey()
cv2.destroyAllWindows()

