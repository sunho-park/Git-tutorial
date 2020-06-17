import matplotlib.pyplot as plt
import numpy as np

# 사진 데이터 읽어들이기
photos = np.load('myproject/photos.npz')
x = photos['x']
y = photos['y']

# 시작 인덱스 --( 1)
idx = 0

#pyplot로 출력하기
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(y[i+idx])
    plt.imshow(x[i+idx])
plt.show()