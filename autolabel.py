from lxml.etree import Element, tostring, parse
from lxml.etree import SubElement as subElement
import torch


from ultralytics import YOLO


class Yolov8Detect():
    def __init__(self, weights, img_size=960, conf_thresh=0.2, iou_thresh=0.5, batchsize=16):
        cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.detect_model = YOLO(weights)
        # print(weights)
        self.detect_model.to(self.device)

        #####
        # 0.x
        # self.img_size = img_size
        # self.batch_size = batchsize
        # self.conf_thresh = conf_thresh
        # self.iou_thresh = iou_thresh
        # self.names =['ganjin','guahua','liewen','heidian','liangshao','jingshangfen','gouya','juzui','gehen','lvgai','yabutie']
        self.names = ['CA001','CA002','CA003','CA004','CB001','CB002','CB003','CB004-1', 'CB004-2',"CC001","CC002",
                      "CC003","CC004","CD001","CD002","CD003","CD004","desktop-1","desktop-2"]
        
    def inferences(self, inputs):
        # result1 = self.detect_model.predict()
        results = self.detect_model(inputs)
        for result in results:
            label_text = []
            boxes = result.boxes
            for box in boxes:
                cat_num = int(box.cls.cpu())
                cate=self.names[cat_num]
                label_text.append([cate, box.xyxy])
        image_name = inputs.split('/')[-1]
        save_path = inputs.replace('jpg', 'xml')
        xml_construct(save_path, inputs.split('/')[-2], image_name, inputs, width=1920, height=1200, label_text=label_text)
        print('save_path:', save_path)
                
      

def xml_construct(save_path,folder,filename,path,label_text,width=800,height=600,depth = 3,segmented=0):
    default_text = 'default'
    node_root = Element('annotation')  # 根节点
 
    node_folder = subElement(node_root, 'folder')  # 在节点下添加名为'folder'的子节点
    node_folder.text = folder  # 设定节点的文字
 
    node_filename = subElement(node_root, 'filename')
    node_filename.text = filename
 
    node_path = subElement(node_root, 'path')
    node_path.text = path
 
    node_size = subElement(node_root, 'size')
    node_size_width = subElement(node_size, 'width')
    node_size_width.text = '%s' % int(width)
    node_size_height = subElement(node_size, 'height')
    node_size_height.text = '%s' % int(height)
    node_size_depth = subElement(node_size, 'depth')
    node_size_depth.text = '%s' % int(depth)
 
    node_segmented = subElement(node_root, 'segmented')
    node_segmented.text = '%s' % int(segmented)
    
    for label in label_text:
        
        node_size = subElement(node_root, 'object')
        node_size_width = subElement(node_size, 'name')
        node_size_width.text = '%s' % (label[0])
        node_size_height = subElement(node_size, 'pose')
        node_size_height.text = 'Unspecified' 
        node_size_depth = subElement(node_size, 'truncated')
        node_size_depth.text = '0' 
        node_size_depth = subElement(node_size, 'difficult')
        node_size_depth.text = '0' 
        
        node_bndbox = subElement(node_size, 'bndbox')
        node_bndbox_xmin = subElement(node_bndbox, 'xmin')
        node_bndbox_xmin.text = '%s' % int(label[1][0][0])
        node_bndbox_ymin = subElement(node_bndbox, 'ymin')
        node_bndbox_ymin.text = '%s' % int(label[1][0][1])
        node_bndbox_xmax = subElement(node_bndbox, 'xmax')
        node_bndbox_xmax.text = '%s' % int(label[1][0][2])
        node_bndbox_ymax = subElement(node_bndbox, 'ymax')
        node_bndbox_ymax.text = '%s' % int(label[1][0][3])    
    
 
    xml = tostring(node_root, pretty_print=True) #将上面设定的一串节点信息导出
    with open(save_path,'wb') as f: #将节点信息写入到文件路径save_path中
        f.write(xml)

    return
 

if __name__ == '__main__':
    model_path = './weights/3.17.1.best_x.pt'
    model = Yolov8Detect(model_path)
    import glob 
    image_path = glob.glob('C:/Users/Zhuiri Xiao/Desktop/data317/color_images/*.jpg')
    for img_path in image_path[:]:
        model.inferences(img_path)

