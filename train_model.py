from ultralytics import YOLO

#load a pre-trained model
# model = YOLO('./weights/yolov8n.pt')  # build from YAML and transfer weights
# Train the model

model = YOLO('yolov8n-att.yaml').load("./weights/Akua-att.pt")
model.train(data='./robo.yaml', epochs=3000, imgsz=1024,cls=0.65,label_smoothing=0.03,patience=100,batch=32, workers=12,device="cuda",mosaic=0,erasing=0,optimizer="AdamW",momentum=0.9,lr0=0.00125,perspective=0.00005,shear=1,dropout=0.05,degrees=1)
# model.export(format='')