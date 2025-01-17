import numpy as np
import dlib
import cv2
import os
from PIL import Image
import glob, os
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

predictor_file = r'C:\Users\bitcamp\Desktop\opencv_dnn_202005\shape_predictor_68_face_landmarks.dat'

# image_file = [cv2.imread(file) for file in glob.glob("./resnet/khd/*.jpg")]
# image_file = r'D:\study1\Darkhorseproject\matrix\sin5.jpg'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (224, 224)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

# image = cv2.imread(image_file)
# image_origin = image.copy()

# (image_height, image_width) = image.shape[:2]
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# rects = detector(gray, 1)

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom() 
    return (startX, endX, startY, endY)    


from PIL import Image
from tqdm import tqdm
import glob
 
filename_list = glob.glob(r'D:\kface/395\*.jpg')
filename_list.sort()
 
fill_number = len(str(len(filename_list)))

for idx, filename in enumerate(tqdm(filename_list), 1):
    image = cv2.imread(filename)
    image_origin = image.copy()

    (image_height, image_width) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        (x, y, w, h) = getFaceDimension(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
        show_parts = points[EYES]

        right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")
        print(right_eye_center, left_eye_center)

        cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1)
        cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)
        
        cv2.circle(image, (left_eye_center[0,0], right_eye_center[0,1]), 5, (0, 255, 0), -1)
        
        cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
                (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 2)
        cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]),
            (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1)
        cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]),
            (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1)

        eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
        eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
        degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180

        eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
        aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
        scale = aligned_eye_distance / eye_distance

        eyes_center = ((left_eye_center[0,0] + right_eye_center[0,0]) // 2,
                (left_eye_center[0,1] + right_eye_center[0,1]) // 2)
        cv2.circle(image, eyes_center, 5, (255, 0, 0), -1)
                
        metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)
        cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
            flags=cv2.INTER_CUBIC)
        
        # cv2.imshow("warpAffine", warped)
        (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)
        croped = warped[startY:endY, startX:endX]
        output = cv2.resize(croped, OUTPUT_SIZE) 
        # cv2.imshow("output", output)
        savename = r'D:\kface\crop395/' + str(idx).zfill(fill_number) + '.jpg'
        # if not os.path.exists(savename):
        #     os.mkdir(savename)


        crop_image = cv2.imwrite(savename, output)

        # 저장경로
        # savedir = "./flickr/" + dir
        # if not os.path.exists(savedir):
        #     os.mkdir(savedir)


        for (i, point) in enumerate(show_parts):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

# cv2.imshow("Face Alignment", image)
cv2.waitKey(0)   
cv2.destroyAllWindows()


