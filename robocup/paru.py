# 纯纯的帕鲁代码


import PIL
import cv2
import numpy as np
import torch
# from tracker.sort import Sort
# from . import tools
import tools
from ultralytics import YOLO



class Paru(object):
    def __init__(self, weights,dataset):
        self.device = torch.device('cuda:0')
        self.model=YOLO(model=weights)
        self.class_names = tools.load_yaml(dataset)['names']
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

        detected_imgs=[]

        for result in results:
            new_boxes = []
            new_classes = []
            new_scores = []

            for box, cls in zip(result.boxes, result.classes):
                if cls == 17:
                    desk_box = box
                    break
            #
            # for box, cls in zip(result.boxes, result.classes):
            #     x = (box[0] + box[2]) / 2
            #     y = (box[1] + box[3]) / 2
            #     if x < box[0] or x > box[2] or y < box[3] or y > box[1]:
            #         continue

            for i in range(len(result.boxes)):
                box = result.boxes[i]
                cls = result.classes[i]
                prob = result.probs[i]

                x = (box[0] + box[2]) / 2
                y = (box[1] + box[3]) / 2
                if x < box[0] or x > box[2] or y < box[3] or y > box[1]:
                    continue
                new_boxes.append(box)
                new_classes.append(cls)
                new_scores.append(prob)

            results.pred = np.column_stack((new_boxes, new_scores, new_classes))
            
            # boxes = result.boxes  # Boxes object for bounding box outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # result.show()  # display to screen
            detected_imgs.append(result.plot())


            # print("boxs:{}".format(boxes))
            # print("masks:{}".format(masks))
            # print("keypoints:{}".format(keypoints))
            # print("probs:{}".format(probs))
            # cv2.imshow("test",result.plot())
            # cv2.waitKey(0)

            # result.save(filename=f'result_{str(idx)}.jpg')  # save to disk
        return results,detected_imgs


# just for testing purposes
if __name__ == '__main__':

    myParu=Paru("../weights/yolov8s.pt","../formats/coco.yaml")
    myParu.detect_image("./test_paru.jpg")
    
    pass