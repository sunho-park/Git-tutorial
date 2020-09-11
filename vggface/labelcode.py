# 이미지들을 Numpy 형식으로 변환하기
import numpy as np
from PIL import Image              # Python Image Library
import glob, os, random 

outfile = "./vggface/kface100.npz" # 저장할 파일 이름

# max_photo = 200 # 사용할 장 수
photo_size = 224 # 이미지 크기

x = []          # 이미지 데이터
y = []          # 레이블 데이터

def main():
    # 디렉터리 읽어 들이기             
    for i in os.listdir('D:\kface100'):
        glob_files("D:\kface100/"+str(i), i) 
          
        
                               # 파일로 저장하기
        print(outfile, len(x), "장 저장했습니다.")            # 600 장 저장했습니다.
    np.savez(outfile, x=x, y=y)  

def glob_files(path, label):            # path 내부의 이미지 읽어 들이기    
    files = glob.glob(path+ "/*.jpg")   # 폴더안의 파일들의 목록을 불러옴ex)glob('*.txt')
    # for문으로 모든 파일 색공간과 크기 변경
    num = 0
    for f in files:                                 # 200개의 이미지 3번
        # if num >= max_photo:break                   # 200개가 넘으면 멈춤
        num += 1
        # 이미지 파일 읽어 들이기 Image모듈 open함수 사용
        img = Image.open(f)
        img = img.convert("RGB")                     # 색공간 변환하기
        img = img.resize((photo_size, photo_size))   # (32, 32)로 크기 변경하기
        img = np.asarray(img)                        # modifying img itself
                                                     # array : modifying img a copy
        x.append(img)
        y.append(label)
# 엔트리 포인트 또는 메인이므로 main() 함수 실행    
if __name__ == '__main__':  
    main()

    

print(len(x))
# print(x)