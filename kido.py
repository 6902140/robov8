from pyorbbecsdk import Pipeline,Config,OBSensorType,OBAlignMode,FrameSet,OBFormat,OBPropertyID
from ultralytics import YOLO
from PyQt5 import QtWidgets,QtGui
import numpy as np
from utils import frame_to_bgr_image
import threading
import sys
import time

from robocup.paru import Paru

RESULT_PATH = './result'
SERVER_IP = '127.0.0.1'
SERVER_PORT = 6666

# frame 宽高
FRAME_W = 640
FRAME_H = 480

class UI(QtWidgets.QWidget):
    def __init__(self):
        super(UI,self).__init__()
        print("正在初始化界面")
        self.setFixedSize(770,540)
        pixmap = QtGui.QPixmap(320,240)
        pixmap.fill(QtGui.QColor(0))
        pixmap2 = QtGui.QPixmap(320,240)
        pixmap2.fill(QtGui.QColor(0))
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(pixmap)
        self.label2 = QtWidgets.QLabel()
        self.label2.setPixmap(pixmap2)
        self.table_result = QtWidgets.QTableWidget(0,4)
        self.table_result.setFixedSize(402,240)
        self.table_result.setHorizontalHeaderLabels(['Goal_ID', 'Goal_A', 'Goal_B','Goal_C'])
        self.button_begin = QtWidgets.QPushButton()
        self.button_begin.setFixedSize(410,40)
        self.button_begin.setText("开始")
        self.button_begin.clicked.connect(self.on_button_begin_clicked)
        self.button_detect = QtWidgets.QPushButton()
        self.button_detect.setFixedSize(410,40)
        self.button_detect.setText("检测")
        self.button_detect.clicked.connect(self.on_button_detect_clicked)
        self.radio1 = QtWidgets.QRadioButton('静态测量') 
        self.radio1.setChecked(True)
        self.radio1.toggled.connect(self.on_radio_toggled1)
        self.radio2 = QtWidgets.QRadioButton('动态测量')
        self.radio2.setChecked(False)
        self.radio2.toggled.connect(self.on_radio_toggled2)
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.radio1)
        self.vbox.addWidget(self.radio2)
        self.label_status = QtWidgets.QLabel()
        self.label_status.setText("就绪")
        self.gLayout = QtWidgets.QGridLayout()
        self.gLayout.addWidget(self.label,0,0,4,1)
        self.gLayout.addWidget(self.label2,4,0,4,1)
        self.gLayout.addWidget(self.table_result,0,1,4,1)
        self.gLayout.addLayout(self.vbox,4,1,2,1)
        self.gLayout.addWidget(self.button_begin,6,1,1,1)
        self.gLayout.addWidget(self.button_detect,7,1,1,1)
        self.gLayout.addWidget(self.label_status,8,0,1,2)
        self.setLayout(self.gLayout)
        self.res_dict=dict()
        print("正在加载模型")
        self.model=Paru("./weights/Athena.pt","./robo.yaml")

        print("模型预热中...")  
        self.model.detect_image(np.zeros(shape=(FRAME_H, FRAME_W, 3), dtype=np.uint8), draw_img=False)

        print("正在连接相机")
        self.hasStarted=False
        self.mode=1
        self.camera_running=False
        self.camera_thread=threading.Thread(target=self.camera_thread_run)
        self.currentFrame=None
        self.pipeline=Pipeline()
        self.device=self.pipeline.get_device()
        self.config=Config()

        self.camera_thread.start()

    def on_radio_toggled1(self, checked):  
        if checked:  
            self.radio2.setChecked(False)
            self.mode=1
        if not self.hasStarted:
            return
        self.camera_running=False
        self.pipeline.stop()
        self.init_camera()
    
    def on_radio_toggled2(self, checked):  
        if checked:  
            self.radio1.setChecked(False)
            self.mode=2
        if not self.hasStarted:
            return
        self.camera_running=False
        self.pipeline.stop()   
        self.init_camera()
    
    def on_button_begin_clicked(self):
        if self.hasStarted:
            return
        self.hasStarted=True
        self.init_camera()

    def on_button_detect_clicked(self):
        self.label_status.setText("正在识别目标")
        results,img=self.model.detect_image(self.currentFrame)
        result=results[0]
        boxes=result.boxes
        boxes_num=len(boxes.cls)
        temp_dict=dict()
        isDesk=False
        desk_xyxy=[]
        for idx in range(boxes_num):
            i=int(boxes.cls[idx])
            if i==17:
                print("成功找到桌面！")
                isDesk=True
                for j in range(4):
                    desk_xyxy.append(float(boxes.xyxyn[idx][j]))
                             
                break
        if not isDesk:
            desk_xyxy=[0.33,0.67,0.33,0.67]


        for idx in range(boxes_num):
            i=int(boxes.cls[idx])
            nameOfBox=self.model.class_names[i]
            x_centr=(float(boxes.xyxyn[idx][0])+float(boxes.xyxyn[idx][2]))/2
            y_centr=(float(boxes.xyxyn[idx][1])+float(boxes.xyxyn[idx][3]))/2
            if x_centr<=desk_xyxy[0] or y_centr<=desk_xyxy[1] or x_centr>=desk_xyxy[2] or y_centr>=desk_xyxy[3]:
                print("识别到{},[{},{}],但是未在合法范围之内".format(nameOfBox,x_centr,y_centr))
                continue
            if nameOfBox not in temp_dict.keys():
                temp_dict[nameOfBox]=1
            else:
                temp_dict[nameOfBox]+=1
            pass
        
        for key in temp_dict.keys():
            if key not in self.res_dict.keys():
                self.res_dict[key]=temp_dict[key]
            else:
                if temp_dict[key]>self.res_dict[key]:
                    self.res_dict[key]=temp_dict[key]
        print(self.res_dict)
        
        
        img=img[0].astype(np.uint8)
        showImage = QtGui.QImage(img,img.shape[1],img.shape[0],QtGui.QImage.Format_RGB888)
        self.label2.setPixmap(QtGui.QPixmap.fromImage(showImage).scaled(320,240))
        self.label_status.setText("识别完成")

    def init_camera(self):
        self.label_status.setText("正在初始化相机")
        if self.mode==1:
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
            self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, False)
        else:
            self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 10)
            self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, True)
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(2048,0,OBFormat.RGB,15)
            self.config.enable_stream(color_profile)
        except Exception as e:
            print(e)
            return
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            return
        self.camera_running=True
        self.label_status.setText("运行中")

    def camera_thread_run(self):
        while True:
            if not self.camera_running:
                continue
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            self.currentFrame=color_image
            showImage = QtGui.QImage(self.currentFrame.astype(np.uint8),self.currentFrame.shape[1],self.currentFrame.shape[0],QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage).scaled(320,240))

   

def main():
    app = QtWidgets.QApplication(sys.argv)
    mywin = UI()
    mywin.setWindowTitle('冬日阳光-3D视觉')
    mywin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()