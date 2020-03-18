import time
from PIL import Image
import sys
import torchvision.transforms as transforms
sys.path.append('/content/cloned-repo/isic1')
import shutil
import os
import PIL
from sys import exit
from pathlib import Path
import numpy as np
from data.autoaugment import AutoAugment
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score, classification_report
import json
import urllib.request
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import glob
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

#过滤非图片类型的文件
def filter_image_file(filename):
    for extension in IMG_EXTENSIONS:
        if filename.endswith(extension):
            return filename
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

#input dir,output Image
def get_image(image_path):
    if not Path(image_path).exists():
        raise IOError('not such file of ' + image_path)
    return Image.open(image_path)


def get_transforms(opt):
    transform_list = []
    # if opt.mode == 'train':
    #     if opt.centerCropSize:
    #         transform_list.append(transforms.CenterCrop(opt.centerCropSize))
        # if opt.autoAugments:
        #     ag = AutoAugment(opt.autoAugments)
        #     transform_list.append(ag)
    if opt.resize:
        transform_list.append(transforms.Resize(opt.resize))
    # 多种组合变换有一定的先后顺序，处理PILImage的变换方法（大多数方法）
    # 都需要放在ToTensor方法之前，而处理tensor的方法（比如Normalize方法）就要放在ToTensor方法之后。
    transform_list.append(transforms.ToTensor())
    # if opt.mode == 'train':
    #     if opt.normalize:
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # input must be a tensor
    return transforms.Compose(transform_list)


def get_auto_augments(auto_augment_object):
    transform_list = []
    transform_list.append(auto_augment_object)
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def make_directory(path):
    dataset_path = Path(path)
    if dataset_path.exists():
        shutil.rmtree(path)
    os.makedirs(dataset_path)


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
    # label_ndarray = label_dataframe.iloc[:, 1:].as_matrix()
    return label_dataframe



#输入文件名称,返回该文件的完整路径
def get_file_path(filename):
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('filename')[0], filename)
    return file_path

#get wrongly classified image name
def get_image_name_by_number(csv_path,number_list):
    label_csv_path = get_csv_by_path_name(csv_path)
    label_dataframe = pd.read_csv(label_csv_path[0], header=0,  engine='python')
    error_classified_image_list = label_dataframe.iloc[number_list]['image'].values.tolist()
    return error_classified_image_list

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
def split_test_data(rowImagePath, testImagePath, testLabelPath):
    test_file_name = pd.read_csv(testLabelPath, usecols=['image'], header=0,  engine='python').values.squeeze(1)
    des_file_root = Path(testImagePath)
    src_file_root = Path(rowImagePath)
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


#input speficied rate return center-croped image,image type jepimagefile
def centercrop_image(input, rate):
    w, h = input.size[0]*rate, input.size[1]*rate
    centercrop = transforms.CenterCrop(size=(h, w))
    input_cropped = centercrop(input)
    return input_cropped
# def centercrop_image(image, target_width, target_height):
#     w, h = image.shape[0], image.shape[1]
#     if target_height > h or target_width > w:
#         raise ValueError('target width %d or target height %d is less then input size %d, %d' % (target_width, target_height, w, h))
#     x_start = round((w - target_width) / 2)
#     y_start = round((h - target_height) / 2)
#     return image[x_start:x_start + target_width, y_start: y_start + target_height, :]


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


def get_expand_coordinates(lmda, coordinates):
    if len(coordinates) != 4:
        raise ValueError("the second input must be tuple of 4 elements")
    x_start, y_start, w, h = coordinates
    w_hat = w*lmda
    h_hat = h*lmda
    x_hat = x_start - w * (lmda - 1)/ 2
    y_hat = y_start - h * (lmda - 1)/2
    if x_hat > 0 and y_hat > 0:
        return round(x_hat), round(y_hat), round(w_hat), round(h_hat)
    else:
        return x_start, y_start, round(w*lmda), round(h*lmda)


#input two points in numpy array,return the distance between
def get_distance(m, n):
    return np.sqrt(np.sum((m - n) ** 2))


#input image width and height,return order width
def get_expand_border(w, h, target_size):
    if target_size < w or target_size < h:
        raise ValueError('target size %d is less than input %d %d' % (target_size, w, h))
    return int((target_size - w)/2), int((target_size - h)/2)



#input a path,search for csv and return file name
def get_csv_by_path_name(path):
    csv_path = glob.glob(os.path.join(path, '*.csv'))
    return csv_path


#input /d/d/a.py return a.py
def get_file_name(path):
    return path.split(os.sep)[-1]

#input ['a/b/1.jpg', 'a/b/2.jpg], return ['1.jpg', '2.jpg']
def get_filename_list(file_path_list):
    file_name_list = []
    for file_path in file_path_list:
        file_name_list.append(file_path.split(os.sep)[-1])
    return file_name_list



#input evaluation metrics,output sensitivity
def calculate_mean_sensitivity(class_number, metrics_dict):
    sensitivity = 0
    for k, v in metrics_dict.items():
        # sensitivity is valid when only true positive sample of this class is not 0
        if 'tp' in k:
            class_no = k.split('_')[1]
            # get the the fn numbers of this tp sample,and caculate sensitivity
            fn_key = 'fn' + '_' + class_no
            if not fn_key in metrics_dict.keys():
                sensitivity += 1 / class_number
            else:
                sensitivity += v / ((v + metrics_dict[fn_key]) * class_number)
    return sensitivity


#input evaluation metrics,output accuracy
def calculate_test_metrics(truth_list, pred_list, class_number):
    assert len(truth_list) == len(pred_list)
    metric_dict = {}
    metric_dict['average marcro precision'] = round(precision_score(truth_list, pred_list, average='macro'), 3)
    metric_dict['average accuracy'] = round(accuracy_score(truth_list, pred_list), 3)
    metric_dict['average macro recall'] = round(recall_score(truth_list, pred_list, average='macro'), 3)
    metric_dict['average macro f1 score'] = round(f1_score(truth_list, pred_list, average='macro'), 3)
    target_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    metric_dict['overall report'] = classification_report(truth_list, pred_list, target_names=target_names)
    return metric_dict

#read dict from json string in a file
def get_dict_from_json(file_name):
    data_dict = {}
    with open(file_name, 'r') as log:
        log_record_list = log.readlines()
    log.close()
    for record in log_record_list:
        record_dict = json.loads(record)
        for k, v in record_dict.items():
            data_dict[k] = v
    return data_dict

#image_list is a tuple of (image,image_name)
def show_multiple_images(image_list, images_per_row):
    total_image_num = len(image_list)  # 总cam图片数量
    num_rows = total_image_num / images_per_row
    plt.figure(figsize=(20, 15))
    for idx, image in enumerate(image_list):
        plt.subplot(num_rows, images_per_row, idx + 1)
        plt.imshow(image)
    plt.show()


def tensor_transform(input, target_type):
    if isinstance(input, torch.Tensor):
        if target_type == 'image':
            return transforms.ToPILImage()(input).convert('RGB')
        if target_type == 'numpy':
            return input.numpy()
    if isinstance(input, np.ndarray):
        if target_type == 'tensor':
            tran = transforms.ToTensor()
            return tran(input)
    if isinstance(input, pd.DataFrame):
        if target_type == 'ndarray':
            return input.values


def accuracy_count(label_tensor, pred_tensor):
    nd_pred = tensor_transform(pred_tensor, 'numpy')
    y1_pred = tensor_transform(label_tensor, 'numpy')
    result = (nd_pred == y1_pred).astype(int)
    return result.sum()





if __name__ == '__main__':
    img2 = cv.imread("C:\\Users\\23270\\Desktop\\aa\\ISIC_0010605.jpg")
    a = centercrop_image(img2, 200, 200)
    cv.imshow('a', a)