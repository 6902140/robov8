import os

# 获取当前文件目录
current_directory = os.getcwd()

# 遍历当前目录下所有以'lyf'开头的txt文件
for filename in os.listdir(current_directory):
    if filename.startswith('lyf') and filename.endswith('.txt'):
        file_path = os.path.join(current_directory, filename)
        
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                # 检查每一行是否以'18'开头
                if line.strip().startswith('18'):
                    # 如果是，则将'18'改写为'17'并写入文件
                    file.write('17' + line.strip()[2:] + '\n')
                else:
                    # 否则直接写入文件
                    file.write(line)
