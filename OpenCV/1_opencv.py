import sys
import cv2

img = cv2.imread('./Opencv/cat.bmp') 

if img is None:
    print('Image load failed!')
    sys.exit()

    
cv2.namedWindow('image')    # wn
cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()
