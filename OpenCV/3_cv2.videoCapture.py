import cv2

cap = cv2.VideoCapture('./Opencv/vtest.avi')

fps = round(cap.get(cv2.CAP_PROP_FPS))
delay = round(1000/fps)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150)    # edge 검출하는 영상만들기 넘파이형식으로 만들어짐

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)
    
    if cv2.waitKey(20) == 27:    # esc 
        break

cap.release()
cv2.destroyAllWindows()
    

