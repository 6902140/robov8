
from ultralytics import YOLO



import os
import shutil

path_xml = "C:/Users/Zhuiri Xiao/Desktop/3.17.image/colotr"
filelist = os.listdir(path_xml)

path3 = "C:/Users/Zhuiri Xiao/Desktop/3.17.image/labels/"


for files in filelist:
    filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
    filename0 = os.path.splitext(files)[0]  #读取文件名
    # print(filename1)
    # m = filename1 == '.txt'
    # print(m)
    # if filename1 == '.txt' :
    #     full_path = os.path.join(path_xml, files)
    #     despath = path2 + filename0 +'.txt' #.txt为你的文件类型，即后缀名，读者自行修改
    #     shutil.move(full_path, despath)

    # if filename1 == '.jpg':
    #     full_path = os.path.join(path_xml, files)
    #     despath = path1 + filename0 + '.jpg'  #.jpg为你的文件类型，即后缀名，读者自行修改
    #     shutil.move(full_path, despath)

    if filename1 == '.xml':
        full_path = os.path.join(path_xml, files)
        despath = path3 + filename0 + '.xml'  # .xml为你的文件类型，即后缀名，读者自行修改
        shutil.move(full_path, despath)

