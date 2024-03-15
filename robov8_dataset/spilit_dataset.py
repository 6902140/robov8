# -*- coding:utf-8 -*
import os
import random
import os
import shutil
def data_split(full_list, ratio):

    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


train_p="./train"
val_p="./val"
imgs_p="images"
labels_p="labels"

#创建训练集
if not os.path.exists(train_p):#指定要创建的目录
    os.mkdir(train_p)
tp1=os.path.join(train_p,imgs_p)
tp2=os.path.join(train_p,labels_p)
print(tp1,tp2)
if not os.path.exists(tp1):#指定要创建的目录
    os.mkdir(tp1)
if not os.path.exists(tp2):  # 指定要创建的目录
    os.mkdir(tp2)

#创建测试集文件夹
if not os.path.exists(val_p):#指定要创建的目录
    os.mkdir(val_p)
vp1=os.path.join(val_p,imgs_p)
vp2=os.path.join(val_p,labels_p)
print(vp1,vp2)
if not os.path.exists(vp1):#指定要创建的目录
    os.mkdir(vp1)
if not os.path.exists(vp2):  # 指定要创建的目录
    os.mkdir(vp2)

#数据集路径
images_dir="./images"
labels_dir="./labels"
#划分数据集，设置数据集数量占比
proportion_ = 0.9 #训练集占比

total_file = os.listdir(images_dir)

num = len(total_file)  # 统计所有的标注文件
list_=[]
for i in range(0,num):
    list_.append(i)

list1,list2=data_split(list_,proportion_)

for i in range(0,num):
    file=total_file[i]
    print(i,' - ',total_file[i])
    name=file.split('.')[0]
    if i in list1:
        jpg_1 = os.path.join(images_dir, file)
        jpg_2 = os.path.join(train_p, imgs_p, file)
        txt_1 = os.path.join(labels_dir, name + '.txt')
        txt_2 = os.path.join(train_p, labels_p, name + '.txt')
        if os.path.exists(jpg_1):
            shutil.copyfile(jpg_1, jpg_2)
        if os.path.exists(txt_1):
            shutil.copyfile(txt_1, txt_2)
        # if os.path.exists(txt_1) and os.path.exists(jpg_1):
        #     shutil.copyfile(jpg_1, jpg_2)
        #     shutil.copyfile(txt_1, txt_2)
        # elif os.path.exists(txt_1):
        #     print(txt_1)
        # else:
        #     print(jpg_1)

    elif i in list2:
        jpg_1 = os.path.join(images_dir, file)
        jpg_2 = os.path.join(val_p, imgs_p, file)
        txt_1 = os.path.join(labels_dir, name + '.txt')
        txt_2 = os.path.join(val_p, labels_p, name + '.txt')
        if os.path.exists(jpg_1):
            shutil.copyfile(jpg_1, jpg_2)
        if os.path.exists(txt_1):
            shutil.copyfile(txt_1, txt_2)

print("数据集划分完成： 总数量：",num," 训练集数量：",len(list1)," 验证集数量：",len(list2))

