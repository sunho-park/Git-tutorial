import numpy as np
import h5py
 
with h5py.File('./vggface/data_hdf5', 'w') as f:
 
    f.create_dataset('image', (1000, 32, 32, 3), dtype='float32')    # 1000개의 32x32 RGB 이미지를 담는 데이터 공간을 생성한다. 
    f.create_dataset('label', (1000,), dtype='float32')              # 1000개의 float을 담는 데이터 공간을 생성한다. 
    image_set = f['image']    # 실 데이터 공간에 접근할 변수를 할당한다. 
    label_set = f['label']