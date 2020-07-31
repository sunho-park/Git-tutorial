import numpy as np

import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./Darkhorseproject/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./Darkhorseproject/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        shapes.append(shape)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        
    
    return rects, shapes, shapes_np

img_bgr = cv2.imread('./Darkhorseproject/matrix/neo.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
faces = detector(img_rgb, 1)


print("{} faces are detected.", format(len(faces)))
print(faces)

cv2.imshow('face of neo', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()



