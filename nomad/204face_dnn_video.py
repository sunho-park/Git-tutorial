import cv2
import numpy as np

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
min_confidence = 0.5
file_name = r'C:\Users\bitcamp\Desktop\opencv_dnn_202005\video\obama_01.mp4'

def detectAndDisplay(frame):
    # pass the blob through the model and obtain the detections
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # Resizing to a fixed 300x300 pixels and then normalizing it
    blob = cv2.dnn.blobFromImage(cv2.resize)