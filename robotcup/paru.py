import PIL
import cv2
import numpy as np
import torch
# from tracker.sort import Sort

from ultralytics import YOLO



class Paru(object):
    def __init__(self, weights):
        self.device = torch.device('cuda:0')
        self.model=YOLO(model=weights,task="detect")
        # self.class_names = tools.load_yaml(dataset)['names']
    def detect_image(self, source, draw_img=True):
        """
        args:
        
            Args:
                source: type:Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] 
                表示想要预测的帧或图片

        returns:

            resp: (List[ultralytics.engine.results.Results])表示预测出来的结果


        """

        # 强制全部转化为list
        if isinstance(source,list):
            pass
        else:
            image_list=[source]
        idx=0
        results=self.model(image_list)
        for result in results:
            idx+=1
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            # result.show()  # display to screen
            result.save(filename=f'result_{str(idx)}.jpg')  # save to disk
        return results



if __name__ == '__main__':

    myParu=Paru("./yolov8s.pt")
    myParu.detect_image("./test_paru.jpg")
    
    pass