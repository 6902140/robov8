import os
import random
import cv2

def add_gaussian_noise(image_path):
    image = cv2.imread(image_path)
    noisy_image = image.copy()

    # 高斯噪声的标准差
    std_dev = 100*random.random()
    strength=(1-2*random).random()*400
    # 生成随机数，判断是否加噪声
    if random.random() < 1:
        # 生成高斯噪声
        noise = cv2.randn(noisy_image, strength, std_dev)
        # 添加噪声
        noisy_image = cv2.add(image, noise)

    # 获取文件名和扩展名
    file_name, ext = os.path.splitext(image_path)

    # 添加后缀并保存文件
    output_path = file_name + '_gauss1' + ext
    cv2.imwrite(output_path, noisy_image)

# 遍历文件夹下的所有文件
folder_path = '你的文件夹路径'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 添加高斯噪声并保存文件
        add_gaussian_noise(file_path)