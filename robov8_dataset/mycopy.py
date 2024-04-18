import os
import shutil

def copy_and_rename_txt_files(source_dir, destination_dir):
    # 创建目标目录
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.txt'):
                # 构建源文件的完整路径
                source_file = os.path.join(root, filename)
                # 构建目标文件的完整路径，并重命名
                base_name, extension = os.path.splitext(filename)
                dest_filename = base_name + '_0' + extension
                destination_file = os.path.join(destination_dir, dest_filename)
                # 复制文件
                shutil.copyfile(source_file, destination_file)

# 指定源目录和目标目录
source_directory = 'C:/Users/Zhuiri Xiao/Desktop/dxr/labels'
destination_directory = 'C:/Users/Zhuiri Xiao/Desktop/dxr/labels_'

# 调用函数
copy_and_rename_txt_files(source_directory, destination_directory)
