import numpy as np
import h5py
 
a=np.random.random(size=(100,20))

with h5py.File('./vggface/kfacelabel.hdf5', 'w') as f:
 
    f.create_dataset('image', (40000, 32, 32, 3), dtype='float32')    # 1000개의 32x32 RGB 이미지를 담는 데이터 공간을 생성한다. 
    f.create_dataset('label', (400,), dtype='float32')                # 1000개의 float을 담는 데이터 공간을 생성한다. 
    image_set = f['image']    # 실 데이터 공간에 접근할 변수를 할당한다. 
    label_set = f['label']



with h5py.File('./vggface/kfacelabel.hdf5', 'r') as f: # read
    data = f['image'][:]                                    # 'dataset_1'불러오기
    data2 = f['label'][:]
print(data)
print(data.shape)

print(data2)
print(data2.shape)

# 하나하나 저장
f = h5py.File('./vggface/kfacelabel.hdf5', 'w')    # hdf5 file생성
f.create_dataset('image', (100, 20), dtype='float64')       # 'image'라는 (100, 20)의 빈 공간 생성 
dataset = f['image']                                        # 'image'저장소 불러오기

for i in range(len(a)):
    data = a[i]
    dataset[i] = data

print(dataset[10].shape)        
print(dataset.shape)
print(dataset)        
