# robov8

author: **西安交通大学-你说的队**

尝试将往届代码进行迁移，从yolov6代码迁移到yolov8

#### to do list

(1)检查数据集

(2)不同参数训练模型


#### 项目文件组织格式
```python
.
├── launch.py # 启动图形界面的脚本
├── predefined_classes.txt # 定义的物品种类
├── result_r # 暂存识别结果
├── robocup # 对yoloV8进行封装
├── robov8_dataset # 数据集目录
├── robo.yaml # 一些配置
├── runs 
├── tests 
├── tools 
├── train_history # 历史训练记录
├── train_model.py # 训练模型的脚本
├── ultralytics # 修改后的ultralytics库
└── weights # 模型权重文件

```


#### 模型特点
```python
.
├── Akua-att.pt # 增加注意力机制
├── Akua-pro.pt # 无注意力机制的最好版本（5.1）
├── Akua.pt
├── Akua-v0.1.pt
├── Akua-v0.2.pt
└── Akua-v0.3.pt

```
