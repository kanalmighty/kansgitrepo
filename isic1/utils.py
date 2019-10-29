import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import requests
import urllib.request
import pandas as pd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

#check if it's a file with a valid suffix
def is_image_file(filename):
    return any(filename.endwith(extension) for extension in IMG_EXTENSIONS)

#传入路径，返回图片数组
def get_image_set(dir):
    images = []
    for root,_,filenames in os.walk(dir):
        for filename in filenames:
            image_file = os.path.join(root, filename)
            images.append(image_file)
    return images

#传入图片路径数组，获取图片对象数组
def get_images(image_paths):
    if not isinstance(image_paths, list):
        raise("the aurgument is not a list")
    else:
        images = []
        for image_path in image_paths:
            images.append(cv2.imread(image_path))
    return images


def get_transforms(opt):
    transform_list = []
    if opt.Normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)))
        transform_list.append(transforms.ToTensor)
    return transforms.Compose(transform_list)

def download_dataset(url):
    pwd = os.getcwd()
    path_array = url.split('/')
    file_name = path_array[-1]
    file_path = os.path.join(pwd, file_name)
    urllib.request.urlretrieve(url, file_path, reporthook)
    print('target file has been successfully downloaded in %s' % file_path)


def reporthook(blocks_read, block_size, total_size):
    if not blocks_read:
        print("Connection opened")
    if total_size < 0:
        print('Read %d blocks'  % blocks_read)
    else:
        print('downloading: %d MB, totalsize: %d MB' % (blocks_read*block_size/(1024.0**2), total_size/(1024.0**2)))



if __name__ == '__main__':
    download_dataset('https://s3.amazonaws.com/isic-challenge-2019/ISIC_2019_Training_GroundTruth.csv')

