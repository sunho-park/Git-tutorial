import os


# savedir = "D:\kface\crop_" + str(idx).zfill(fill_number) 


dir = ("D:\kface")
# for i in os.listdir(dir):
#     os.mkdir('D:\kface/' + i + '_crop')    


# dir = ("D:\kface"+ i + '.crop')
for i in os.listdir(dir):
    # print(os.listdir(dir))
    if i + '.crop' in i:
        os.rmdir('D:\kface/' + i + '.crop')    
