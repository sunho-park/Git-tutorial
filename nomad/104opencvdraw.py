import cv2

print("OpenCV version")
print(cv2.__version__)

img = cv2.imread(r'C:\Users\bitcamp\Desktop\opencv_dnn_202005\nomadProgramerIcon.png')
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {} ".format(img.shape[2]))

# cv2.imshow("nomadProgramer", img)

(b, g, r) = img[0, 0]
print("pixel : blue{}, green{}, red{}".format(b, g, r))

dot = img[50:100, 50:100]

img[50:100, 50:100] = (0, 0, 255) # red

cv2.rectangle(img, (150, 50), (200, 100), (0, 255, 0), 5) 

cv2.circle(img, (275, 75), 25, (0, 255, 255), -1) # 중심, 반지름 yellow : (0, 255, 255)

cv2.line(img, (350, 100), (400, 100), (255, 0, 0), 5)

cv2.putText(img, 'creApple', (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

cv2.imshow("nomadProgramer - draw", img)

cv2.waitKey(0)
cv2.destroyAllWindows()