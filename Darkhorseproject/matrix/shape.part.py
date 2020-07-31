import dlib
import numpy as np
from skimage import io
import cv2

predictor_path = './Darkhorseproject/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# img = io.imread(r"D:\study1\Darkhorseproject\matrix\neoface.jpg")

img = cv2.imread(r"D:\study1\Darkhorseproject\matrix\neoface.jpg")

dets = detector(img)

print(dets) # rectangles[[(353, 114) (532, 293)]]

#output face landmark points inside retangle
#shape is points datatype
#http://dlib.net/python/#dlib.point
for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
for b in range(68):
    vec[b][0] = shape.part(b).x
    vec[b][1] = shape.part(b).y

print(vec)
print(len(vec))