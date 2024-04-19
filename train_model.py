from ultralytics import YOLO

#load a pre-trained model
model = YOLO('./weights/yolov8n.pt')  # build from YAML and transfer weights
# Train the model
model.train(data='./robo.yaml', epochs=3000, imgsz=1024,batch=32, workers=12,device="cuda",mosaic=0,erasing=0,optimizer="AdamW",momentum=0.9,lr0=0.00125,perspective=0.00012,shear=20,dropout=0.1,degrees=5)
# model.export(format='')