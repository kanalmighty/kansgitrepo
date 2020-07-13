# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import sys
from easydict import EasyDict
def get_cfg():
    Cfg = EasyDict()
    if sys.platform == 'linux':
        Cfg.partial_train_batch = 8
        Cfg.whole_train_batch = 2
        Cfg.init_epoch = 0
        Cfg.freeze_epoch = 0
        Cfg.unfreeze_epoch = 50
        Cfg.width = 608
        Cfg.height = 608
        Cfg.train_label_path = os.path.join('/content/drive/My Drive/','2007_trainval.txt')
        Cfg.test_label_path = os.path.join('/content/drive/My Drive/', '2007_test.txt')
        Cfg.image_path = 'D:\\datasets\\voc\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages'
        Cfg.model_path = os.path.join('/content/drive/My Drive/', 'yolo.pth')
        Cfg.prune_model_path = os.path.join('/content/drive/My Drive/', 'prune.pth')
        Cfg.model_data_path = '/content/cloned-repo/yolov4-pytorch/model_data/'
    else:
        Cfg.partial_train_batch = 4
        Cfg.whole_train_batch = 2
        Cfg.init_epoch = 0
        Cfg.freeze_epoch = 0
        Cfg.unfreeze_epoch = 50
        Cfg.width = 416
        Cfg.height = 416
        Cfg.train_label_path = 'D:\\PyCharmSpace\\kansgitrepo\\yolov4-pytorch\\2007_trainval.txt'
        Cfg.test_label_path = 'D:\\PyCharmSpace\\kansgitrepo\\yolov4-pytorch\\2007_test.txt'
        Cfg.label_path = 'D:\\PyCharmSpace\\kansgitrepo\\yolov4-pytorch\\2007_train.txt'
        Cfg.image_path = 'D:\\datasets\\voc\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages'
        Cfg.model_path = os.path.join('D:\\datasets\\saved_model', 'yolo.pth')
        Cfg.prune_model_path = os.path.join('D:\\datasets\\saved_model', 'prune.pth')
        Cfg.model_data_path = 'D:\\PyCharmSpace\\kansgitrepo\\yolov4-pytorch\\model_data'

    return Cfg
