import PIL
import numpy as np
import torch
from tracker.sort import Sort

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer


from ultralytics.engine import model
from ultralytics.utils import yaml_load
import tools
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Boxes
from ultralytics.utils.ops import masks2segments

def process_image(img_src, img_size, stride):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.float()  # uint8 to fp32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image

class YOLO(object):
    def __init__(self, weights, dataset):
        self.device = torch.device('cuda:0')
        self.model=model(model=weights,task="detect")
        
        self.class_names = tools.load_yaml(dataset)['names']
        # self.predictor=BasePredictor()
        # self.predictor.model=self.model
        # self.tracker = Sort(max_age=10)
    def detect_image(self, image, draw_img=True):
        """
        args:
        
            Args:
                image type:Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] 
                表示想要预测的帧或图片

        returns:

            resp: (List[ultralytics.engine.results.Results])表示预测出来的结果


        """

        results=self.model(image,stream=True)
        boxes=[]
        probs=[]
        names=[]
        for r in results:
            # boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            boxes.append(r.boxes)
            # probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
            probs.append(r.probs)   
            
            names.append(r.names)

        return results

    # def model_switch(self, model):
    #     ''' Model switch to deploy status '''
    #     from yolov6.layers.common import RepVGGBlock
    #     for layer in model.modules():
    #         if isinstance(layer, RepVGGBlock):
    #             layer.switch_to_deploy()

    # def reset_tracker(self):
    #     self.tracker = Sort(max_age=10)

    # def detect_image(self, image, draw_img=True):
    #     '''
    #     Detect objects from RGB image

    #     image must be a ndarray
    #     '''
    #     img = process_image(image, [640, 640], self.stride).to(self.device)[None]
    #     pred_result = self.model(img)
    #     det = non_max_suppression(pred_result, conf_thres=0.45, iou_thres=0.45)

    #     tracked_det = self.tracker.update(torch.cat(det).cpu().numpy())
    #     # gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #     labels = []
    #     if len(tracked_det):
    #         # det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], image.shape).round()
    #         for *xyxy, conf, cls_id, obj_id in tracked_det:
    #             cls_id = int(cls_id)
    #             obj_id = int(obj_id)
    #             label = f'{self.class_names[cls_id]} {obj_id} {conf:.2f}'
    #             labels.append(label)
    #             if draw_img:
    #                 Inferer.plot_box_and_label(image, max(round(sum(image.shape) / 2 * 0.003), 2),
    #                                            xyxy, label, color=Inferer.generate_colors(cls_id, True))
    #     # PIL.Image.fromarray(image)
    #     return image, labels, tracked_det
