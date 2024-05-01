from ultralytics import YOLO

# 从预训练模型开始训练
model = YOLO('./weights/Akua-v0.3.pt')  # build from YAML and transfer weights

# 从自定义的网络结构开始训练
# model = YOLO('yolov8n-att.yaml').load("./weights/Akua-att.pt")
model.train(data='./robo.yaml', epochs=3000, imgsz=1024,cls=0.75,dfl=1.8,cos_lr=True,cache=True,label_smoothing=0.02,patience=100,batch=32, workers=16,device="cuda",mosaic=0,erasing=0,,optimizer="AdamW",momentum=0.9,lr0=0.00125,perspective=0.00005,shear=1,dropout=0.05,degrees=1)
