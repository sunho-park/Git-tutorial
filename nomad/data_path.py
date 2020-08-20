import os
data_path = ("D:\kface")
for i in os.listdir(data_path):
    dir = os.path.join(data_path, i + '/')
    print(dir)