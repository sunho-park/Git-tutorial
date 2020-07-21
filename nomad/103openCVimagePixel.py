import cv2

print("OpenCV version")
print(cv2.__version__)

img = cv2.imread(r'C:\Users\bitcamp\Desktop\opencv_dnn_202005\nomadProgramerIcon.png')
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {} ".format(img.shape[2]))

cv2.imshow("nomadProgramer", img)

(b, g, r) = img[0, 0]
print("Pixel at (0, 0) - Red : {}, Green : {}, Blue : {} ".format(r, g, b))

dot = img[50:100, 50:100]
cv2.imshow("Dot", dot)
print(dot.shape) # 50, 50, 3 

img[50:100, 50:100] = (0, 0, 255)   # B G R
cv2.imshow("nomadProgramer - dotted", img)

cv2.waitKey(0)
cv2.destroyAllWindows()