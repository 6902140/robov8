import os
import os.path
import xml.etree.ElementTree as ET
import glob

class_names = ['CA001','CA002','CA003','CA004',
               'CB001','CB002','CB003','CB004-1','CB004-2',
               'CC001','CC002','CC003','CC004',
               'CD001','CD002','CD003','CD004',
               'desktop-1'
 ] # 类别名，依次写下来

class_names2=[]



dirpath = r'C:/Users/Zhuiri Xiao/Desktop/dxr/labels'  # 原来存放xml文件的目录
newdir = r'C:/Users/Zhuiri Xiao/Desktop/dxr/yolo_labels'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)
i=0
for fp in os.listdir(dirpath):
    i+=1
    print(fp)
    root = ET.parse(os.path.join(dirpath, fp)).getroot()
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    print(filename)
    for child in root.findall('object'):  # 找到图片中的所有框
        name = child.find('name').text  # 找到类别名
        
        
        class_num = class_names.index(name)  #

        sub = child.find('bndbox')  # 找到框的标注值并进行读取
        xmin = float(sub[0].text)
        ymin = float(sub[1].text)
        xmax = float(sub[2].text)
        ymax = float(sub[3].text)
        try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
        except ZeroDivisionError:
            print(filename, '的 width有问题')

        with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
            f.write(' '.join([str(class_num), str(x_center), str(y_center), str(w), str(h) + '\n']))




