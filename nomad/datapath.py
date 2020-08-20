import os
from PIL import Image
from tqdm import tqdm
import glob


dataset_pathsss = ("D:\kface")
# output_pathss = ("D:\kface_detection")
for dataset_pathss in os.listdir(dataset_pathsss):
    dataset_paths = os.path.join(dataset_pathsss, dataset_pathss + '/')
    print(dataset_paths)
    file_name = glob.glob(dataset_paths + '*.jpg')
    print(file_name)

    # dataset_paths = ['{}'.format(dataset_paths)]
    # print(dataset_paths)

 
# filename_list = glob.glob(r'D:\kface\0\*.jpg')
# filename_list.sort()
 

# data_path = ("D:\kface")
# for i in os.listdir(data_path): 
    
#     dataset_path = os.path.join(data_path, i + '/')
#     filename = glob.glob(data_path + i )

    
#     filename.sort()
#     print(filename)


dataset_pathsss = ("D:\kface")
# output_pathss = ("D:\kface_detection")
for i in os.listdir(dataset_pathsss):
    dataset_paths = os.path.join(dataset_pathsss, i + '/')
    print(dataset_paths)
    file_name = glob.glob(dataset_paths + '*.jpg')
    print(file_name)
