import matplotlib.pyplot as plt
import cv2

# 캐스 케이드 파일 지정해서 검출기 생성하기 ---(1)
cascade_file = "./Darkhorseproject/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 이미지를 읽어 들이고 그레이스케일로 변환하기 ---(2)
img = cv2.imread("./Darkhorseproject/girl.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 인식하기 ---(3)
face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))

# 결과 확인하기---(4)
if len(face_list) == 0:
    print("실패")
    quit()

# 인식한 부분 표시하기 ---*(5)
for (x, y, w, h) in face_list:
    print("얼굴의 좌표 = ", x, y, w, h)
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=20)

# 이미지 출력하기

cv2.imwrite("face-detect.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# https://blog.naver.com/yeob07/221693164432    # 얼굴 검출 및 실시간 모자이크

# http://alereimondo.no-ip.org/OpenCV/34        # cascade 파일 다운로드 좌표

# https://github.com/wikibook/python-ml-app-dev # 책내용 소스코드