# 纯纯的帕鲁代码


import PIL
import cv2
import numpy as np
import torch
# from tracker.sort import Sort

from ultralytics import YOLO
import yaml
from ultralytics.utils.ops import clean_str

def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


class Paru(object):
    def __init__(self, weights,dataset):
        self.device = torch.device('cuda:0')
        self.model=YOLO(model=weights)
        self.class_names =load_yaml(dataset)['names']
    def detect_image(self, source, draw_img=True):
        """
        args:
        
            Args:
                source: type:Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] 
                表示想要预测的帧或图片

        returns:

            resp: (List[ultralytics.engine.results.Results])表示预测出来的结果


        """

        # 强制全部转化为list，为了适配以前代码
        if isinstance(source,list):
            pass
        else:
            image_list=[source]



        
        results=self.model.track(image_list,conf=0.5,tracker='botsort.yaml') # conf 设置置信度下限

        desk_index = 17
        detected_imgs=[]

        desk_box_xyxy=None
        invalid_index=[]
        for result in results:
            class_list_per_frame=result.boxes.cls

            for i in range(0,len(class_list_per_frame)):
                if class_list_per_frame[i] == desk_index:
                    desk_box_xyxy = result.boxes.xyxy[i]
                    # 获取得到桌面的范围
                    break
            
            if desk_box_xyxy==None:
                print("warning：未检测到桌面！")
                return [],source,[]

            else:
                for i in range(0,len(class_list_per_frame)):
                    x=(result.boxes.xyxy[i][0]+result.boxes.xyxy[i][2])/2
                    y=(result.boxes.xyxy[i][1]+result.boxes.xyxy[i][3])/2
                    if (x < desk_box_xyxy[0] or x > desk_box_xyxy[2] or y > desk_box_xyxy[3] or y < desk_box_xyxy[1]):
                        invalid_index.append(i)
                pass

            # boxlist=result.boxes
            # j=0
            # for item in boxlist.cls:
            #     if item == desk_index:
            #         desk_box_xyxy = boxlist.xyxy[j]
            #         break
            #     j+=1
            
            # if desk_box==None:
            #     print("warning：未检测到桌面！")
            #     return [],source
            
            # i=0
            # for xyxy_rectangle in boxlist.xyxy:
            #     x=(xyxy_rectangle[0]+xyxy_rectangle[2])/2
            #     y=(xyxy_rectangle[1]+xyxy_rectangle[3])/2
            #     if (x < desk_box_xyxy[0] or x > desk_box_xyxy[2] or y > desk_box_xyxy[1] or y < desk_box_xyxy[3]):
            #         index.append(i)
            #     i+=1

            

            detected_imgs.append(result.plot())
            resized_image = cv2.resize(result.plot(), (400, 300))
            cv2.imshow("test",resized_image)
            cv2.waitKey(0)
           
            print(results)
            print(results[0].boxes)
        return results,detected_imgs,invalid_index


# just for testing purposes
if __name__ == '__main__':

    myParu=Paru("../weights/3.17.1.best_x.pt","../robo.yaml")
    myParu.detect_image("./test_paru.jpg")
    
    pass