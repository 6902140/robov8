from ultralytics import YOLO



# Load a model
#load a pre-trained model
model = YOLO('./weights/yolov8s.pt')  # build from YAML and transfer weights
# Train the model
model.train(data='./robo.yaml', epochs=100, imgsz=1536,batch=8, workers=0,device="cuda")
# model.export(format='')