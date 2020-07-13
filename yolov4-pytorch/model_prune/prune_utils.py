import xml.etree.ElementTree as ET
import numpy as np
from torchstat import stat
from nets.CSPdarknet import *
# D:\datasets\voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000005.jpg 263,211,324,339,8 165,264,253,372,8 241,194,295,299,8
def get_image_annotation(label_path):
    label_list = []
    with open(label_path) as f:
        for line in f.readlines():
            splited_by_space = line.split(' ')
            image_path = splited_by_space[0]
            image_label_dict = {}
            for item in splited_by_space[1:]:
                image_label_dict[image_path] = item
                label_list.append(image_label_dict)
                image_label_dict ={}
    return label_list

def get_net_stat(model):
    stat(model,(3,608,608))

if __name__ == '__main__':
    x = get_image_annotation('D:\\PyCharmSpace\\kan\\kansgitrepo\\yolov4-pytorch\\2007_train.txt')
    print(x)
    net = darknet53(False)
    get_net_stat(net)





