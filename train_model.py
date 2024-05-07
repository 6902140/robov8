# from ultralytics import YOLO

# # 从预训练模型开始训练
# # model = YOLO('./weights/Akua-v0.3.pt')  # build from YAML and transfer weights

# # 从自定义的网络结构开始训练
# model = YOLO('yolov8s-ghost-p6.yaml')
# model.train(data='./robo.yaml', epochs=3000, imgsz=1024,cls=0.75,dfl=1.8,cos_lr=True,cache=True,label_smoothing=0.02,patience=100,batch=32, workers=16,device="cuda",mosaic=0,erasing=0,optimizer="AdamW",momentum=0.9,lr0=0.00125,perspective=0.00005,shear=1,dropout=0.05,degrees=1)



import sys

sys.path.append("./ultralytics")

from ultralytics import YOLO
# 从预训练模型开始训练
model = YOLO('./yolov5n-att.yaml')  # build from YAML and transfer weights

# 从自定义的网络结构开始训练
# model = YOLO('yolov8s-ghost-p6.yaml')
model.train(data='./robo.yaml', deterministic=False,rect=True,epochs=2000, imgsz=640,cls=0.75,dfl=1.8,cos_lr=True,cache=True,label_smoothing=0.1,patience=100,batch=128, workers=16,device="cuda",mosaic=0,erasing=0,optimizer="AdamW",momentum=0.9,lr0=0.00025,perspective=0.00005,shear=1,dropout=0.05,degrees=1)


# model = YOLO("./runs/detect/train/weights/last.pt")
# # 中断训练的权重文件中的last.pt
# results = model.train(resume=True)
