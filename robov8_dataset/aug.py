import os
import cv2
import numpy as np
import random

def augment(img):
    img=img.astype(np.float32)
    fa=0.4*random.random()+0.8
    fb=0.4*random.random()+0.8
    fc=0.4*random.random()+0.8
    fa2=random.randint(-30,30)
    fb2=random.randint(-30,30)
    fc2=random.randint(-30,30)
    b, g, r = cv2.split(img)
    b=cv2.multiply(b,fa)
    g=cv2.multiply(g,fb)
    r=cv2.multiply(r,fc)
    b=cv2.add(b,fa2)
    g=cv2.add(g,fb2)
    r=cv2.add(r,fc2)
    b=np.clip(b, 0, 255).astype(np.uint8)
    g=np.clip(g, 0, 255).astype(np.uint8)
    r=np.clip(r, 0, 255).astype(np.uint8)
    img = cv2.merge([b, g, r])
    return img


image_dir='/home/moon/Desktop/files/datasets/final1-60-120/images'
augmented_dir='/home/moon/Desktop/files/datasets/final1-60-120/aug_'
total_file=os.listdir(image_dir)
num=len(total_file)
for i in range(0,num):
    file=total_file[i]
    img = cv2.imread(os.path.join(image_dir,file))
    base_name, extension = os.path.splitext(file)  
    for j in range(0,1):
        img=augment(img)
        cv2.imwrite(os.path.join(augmented_dir,"{}_{}{}".format(base_name,j,extension)),img)