# import glob


# for filename in glob.iglob(r'D:\kface\얼굴데이터셋100\**\*.jpg', recursive=True):
#     print(filename)


import os
import shutil

list_dir = ("D:\kface_dataset\HIGH_Resolution")
status = ['S001','S002','S005']
light = ['L1', 'L2', 'L3', 'L4', 'L5', 'L8', 'L9', 'L10', 'L12', 'L13', 'L14', 'L19', 'L20', 'L22','L23']
emo = ['E01']
img = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
       'C16', 'C17', 'C18', 'C19', 'C20']


for i in os.listdir(list_dir):
    os.mkdir('D:\kface/' + i)
    ls = os.path.join(list_dir, i)
    cnt = 0
    for i1 in status:
        ls2 = os.path.join(ls, i1)

        for i2 in light:
            ls3 = os.path.join(ls2, i2)

            for i3 in emo:
                ls4 = os.path.join(ls3, i3)

                for i4 in img:
                    ls5 = os.path.join(ls4, i4)
                    cnt += 1
                    shutil.copy(ls5 + '.jpg', 'D:\kface/' + i + "/" + '{}'.format(cnt) + '.jpg')