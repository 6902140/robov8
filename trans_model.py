from ultralytics import YOLO
 
model = YOLO("./weights/Akua-bogon-att.pt")  # load a pretrained model
model.export(format = "onnx")  # export the model to onnx format