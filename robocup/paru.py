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

        results=self.model(image_list,conf=0.5,tracker='botsort.yaml') # conf 设置置信度下限
        result=results[0]
        detected_imgs=[]

        detected_imgs.append(result.plot())
        resized_image = cv2.resize(result.plot(), (400, 300))
        cv2.imshow("test",resized_image)
        cv2.waitKey(0)
        print(results)
        print(results[0].boxes)
        return results,detected_imgs


# just for testing purposes
if __name__ == '__main__':
    myParu=Paru("../weights/Apoll.pt","../robo.yaml")
    myParu.detect_image("./test.jpg")
    pass