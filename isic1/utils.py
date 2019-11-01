import torch.utils.data as data
import datetime
import time
import pdb
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
import torch
from sys import exit
from pathlib import Path
import requests
import pandas as pd
import urllib.request
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
        print('file'+str(len(filenames)))
        for filename in filenames:
            image_file_path = os.path.join(root, filename)
            if filter_image_file(image_file_path):
                varifeid_image_path = filter_image_file(image_file_path)
            images_path_list.append(varifeid_image_path)
    print('filtered' + str(len(images_path_list)))
    return images_path_list

#传入图片路径数组，获取图片对象数组
def get_image(image_path):
    if not Path(image_path).exists():
        raise IOError('not such file of' + image_path)
    return Image.open(image_path)


def get_transforms(opt):
    transform_list = []
    if opt.Normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if opt.centercropsize:
        transform_list.append(transforms.CenterCrop(opt.centercropsize))
    if opt.resize:
        transform_list.append(transforms.Resize(opt.resize))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

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

#把CSV转换为ndarray返回出去
def read_csv(csv_dir):
    label_dataframe = pd.read_csv(csv_dir)
    # 把dataframe转换为ndarray
    label_ndarray = label_dataframe.iloc[:, 1:].as_matrix()
    label_tensor = torch.from_numpy(label_ndarray)
    return label_ndarray



if __name__ == '__main__':
    # download_dataset('https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Test_Input.zip')
    print(time.strftime("%Y%m%d", time.localtime()))
