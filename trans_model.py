from ultralytics import YOLO
 
model = YOLO("./weights/best.pt")  # load a pretrained model
model.export(format = "onnx",opset=15,dynamic=True)  # export the model to onnx format