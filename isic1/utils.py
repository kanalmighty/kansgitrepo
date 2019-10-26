import torch.utils.data as data
import cv2
import os
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
