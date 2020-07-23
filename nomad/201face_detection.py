import cv2
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # -- detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w //2, y + h//2)
        frame = cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 4)
        faceROI = frame_gray[y:y+h, x:x+h]

        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
        cv2.imshow('Capture - Face detection', frame)

print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread(r"C:\Users\bitcamp\Desktop\opencv_dnn_202005\image\marathon_01.jpg")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channel: {}".format(img.shape[2]))

(height, width) = img.shape[:2]
cv2.imshow("Original Image", img)

face_cascade_name = r'C:\Users\bitcamp\Desktop\frontalFace10\haarcascade_frontalface_alt.xml'
eyes_cascade_name = r'C:\Users\bitcamp\Desktop\frontalFace10\haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

cv2.imshow("Original Image", img)

# --1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('-- (!) Error loading face cascade')
    exit(0)

if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!) Error loading eyes cascade')
    exit(0)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()