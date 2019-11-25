import torch.utils.data as data
import datetime
import time
import pdb
from PIL import Image
import torchvision.transforms as transforms
import cv2
import shutil
import os
import torch
import numpy as np
from sys import exit
from pathlib import Path
import requests
import pandas as pd
import urllib.request
from options.configer import Configer

import configparser

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

#过滤非图片类型的文件
def filter_image_file(filename):
    for extension in IMG_EXTENSIONS:
        if filename.endswith(extension):
            return filename
        else:
            return False

#传入路径，返回图片路径的list
def get_image_set(dir):
    images_path_list = []
    for root,_,filenames in os.walk(dir):
        for filename in filenames:
            image_file_path = os.path.join(root, filename)
            if filter_image_file(image_file_path):
                varifeid_image_path = filter_image_file(image_file_path)
                images_path_list.append(varifeid_image_path)
    return images_path_list

#传入单张图片路径，返回图片
def get_image(image_path):
    if not Path(image_path).exists():
        raise IOError('not such file of ' + image_path)
    return Image.open(image_path)


def get_transforms(opt):
    transform_list = []
    if opt.normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif opt.centercropsize:
        transform_list.append(transforms.CenterCrop(opt.centercropsize))
    elif opt.resize:
        transform_list.append(transforms.Resize(opt.resize))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def get_auto_augments(auto_augment_object):
    transform_list = []
    transform_list.append(auto_augment_object)
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def make_directory(path):
    dataset_path = Path(path)
    if not dataset_path.exists():
        os.mkdir(dataset_path)


def download_dataset(url):
    pwd = os.getcwd()
    dataset_path = Path('./datasets')
    if not dataset_path.exists():
        os.mkdir(dataset_path)
    path_array = url.split('/')
    file_name = path_array[-1]
    file_path = os.path.join(pwd, dataset_path, file_name)
    if Path(file_path).exists():
        os.remove(file_path)
    print('downloading started at %str' % time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    urllib.request.urlretrieve(url, file_path, reporthook)
    print('target file has been successfully downloaded in %s' % file_path)


#urlretrieve的回调函数
def reporthook(blocks_read, block_size, total_size):
    if not blocks_read:
        print("Connection opened")
    if total_size < 0:
        print('Read %d blocks' % blocks_read)
    else:
        if (blocks_read*block_size/(1024.0**2) > 500) and (blocks_read*block_size/(1024.0**2) % 500 == 0):
            print('downloading: %d MB at %s, totalsize: %d MB' % (blocks_read*block_size/(1024.0 ** 2) ,time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), total_size/(1024.0**2)))


def record_data():
    pwd = os.getcwd()
    date_string = time.strftime("%Y%m%d", time.localtime())
    time_sting = time.strftime("%H%M%S", time.localtime())
    root_path = Path('/content/drive/My Drive/daily_report' + date_string)
    if not root_path.exists():
        try:
            os.mkdir(root_path)
        except IOError:
            print("please lunch google drive first")
            exit(0)
    with open(root_path+time_sting+'.log') as log:
        log.write('')


#read csv and return as ndarray
def read_csv(csv_dir):
    label_dataframe = pd.read_csv(csv_dir)
    # 把dataframe转换为ndarray
    label_ndarray = label_dataframe.iloc[:, 1:].as_matrix()
    return label_ndarray



#输入文件名称,返回该文件的完整路径
def get_file_path(filename):
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('filename')[0], filename)
    return file_path


#input 4 number ,output a dictionary of metrics
def get_evaluation_metrics(tp, tn, fp, fn):
    metrics_dict = {}
    if tn + fp == 0:
        metrics_dict['recall'] = None
    else:
        metrics_dict['recall'] = tp/(tp + fp)
    if tp + fn == 0:
        metrics_dict['precision'] = None
    else:
        metrics_dict['precision'] = tp / (tp + fn)
    if tn + fp == 0:
        metrics_dict['false_postive_rate'] = None
    else:
        metrics_dict['false_postive_rate'] = fp/(tn + fp)
    return metrics_dict


#所有参数都fix,把测试数据集分为测试和验证，目前仅适用于collab
def split_test_data():
    configr = Configer().get_configer()
    test_file_name = pd.read_csv('/content/drive/My Drive/isic2019test/ISIC_2019_Test_GroundTruth_Collab.csv',
                                 usecols=['image'], header=0).values.squeeze(1)
    des_file_root = Path(configr['testImagePath'])
    src_file_root = Path(configr['rowImagePath'])
    if not des_file_root.exists():
        os.mkdir(des_file_root)
    for file_name in test_file_name:
        src_file_path = os.path.join(src_file_root, file_name + '.jpg')
        des_file_path = os.path.join(des_file_root, file_name + '.jpg')
        try:
            shutil.copy(src_file_path, des_file_path)
        except IOError:
            print('copy file error!')


#rename a list of images to a path with a speciyied name

def rename_image_list(image_list, target_path):
    if not isinstance(image_list, list):
        raise TypeError("input must be python list")
    make_directory(target_path)
    image_list_renamed = encode_image_name(image_list)
    from tqdm import tqdm
    for image in tqdm(image_list_renamed):
        target_file_name = os.path.join(target_path, image_list_renamed)
        os.rename(image, target_file_name)

def rename_image(image, target_path):
    target_file_name = os.path.join(target_path, image)
    os.rename(image, target_file_name)

def get_onehot_by_class(class_list, specified_class):
    if not isinstance(class_list, list):
        raise TypeError('aurgument #1 must be to a list')
    onehot_dict = {}
    for label_class in class_list:
        onehot_dict[label_class] = 0
    onehot_dict[specified_class] = 1
    return onehot_dict




#rename a list of files to the names derived from their indices
def encode_image_name(total_number,index=0):
    #获取文件总长度
    length = len(str(total_number))
    #获取序号长度
    idx_length = len(str(index))
    #计算补几个零
    place_holder = '0'
    for _ in range(length - idx_length):
        place_holder += '0'
    file_name = place_holder + str(index)
    return file_name

# def encode_image_name(file_list, index=0):
#     file_list_encoded = []
#     if not isinstance(file_list, list):
#         raise TypeError("input must be python list")
#     length = len(file_list)
#     string_lenth = str(length)
#     for idx, image in file_list:
#         #获取文件名后缀
#         suffix = image.split('.')[1]
#         #获取序号长度
#         idx_length = len(str(idx))
#         #计算补几个零
#         place_holder = '0'
#         for _ in (length - idx_length):
#             place_holder += '0'
#         file_name = place_holder + str(idx+index)
#         file_list_encoded.append(file_name + '.' + suffix)
#     return file_list_encode





if __name__ == '__main__':
    list  = get_onehot_by_class(['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'],'NV')
    print(list)
