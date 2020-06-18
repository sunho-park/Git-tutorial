# 이미지들을 Numpy 형식으로 변환하기
import numpy as np
from PIL import Image              # Python Image Library
import glob, os, random 

outfile = "./myproject/photos.npz" # 저장할 파일 이름

max_photo = 200 # 사용할 장 수
photo_size = 32 # 이미지 크기

x = []          # 이미지 데이터
y = []          # 레이블 데이터

def main():
    # 1. 디렉터리 읽어 들이기
    glob_files("./myproject/Hamburger", 0) # 경로 hamburger폴더 안에 있는 모든 jpg파일 0으로 레이블
    glob_files("./myproject/Salad", 1)
    glob_files("./myproject/Gimbap", 2)    
    
    # 파일로 저장하기
    np.savez(outfile, x=x, y=y)
    print(outfile, len(x), "장 저장했습니다.")

#  path 내부의 이미지 읽어 들이기
def glob_files(path, label):            
    files = glob.glob(path+ "/*.jpg")   # glob 폴더안의 jpg파일들의 목록을 불러옴 ex) glob.glob('*.txt')
    random.shuffle(files)
    # 파일 처리하기
    num = 0
    for f in files:                                 # 200개의 이미지 3번 반복
        if num >= max_photo:break                   # 200개가 넘으면 멈춤
        num += 1
        #이미지 파일 읽어 들이기 Image모듈 open함수 사용
        img = Image.open(f)
        img = img.convert("RGB")                     # 색공간 변환하기
        img = img.resize((photo_size, photo_size))   # (32, 32)로 크기 변경하기
        img = np.asarray(img)                        # modifying img itself
                                                     # array : modifying img a copy
        x.append(img)
        y.append(label)
        
if __name__ == '__main__':  # 엔트리 포인트 또는 메인이므로 main() 함수 실행 // if문 없으면 아무것도 출력하지 않음
    main()

    

