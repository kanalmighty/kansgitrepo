# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import sys
sys.path.append('/content/cloned-repo/isic1')
import re
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models
import argparse
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from options.configer import Configer

from skimage import io
import cv2
from gradcam.interpretability.grad_cam import GradCAM, GradCamPlusPlus
from gradcam.interpretability.guided_back_propagation import GuidedBackPropagation
from efficientnet_pytorch import EfficientNet

def get_net(net_name, class_number, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    elif net_name in 'resnet18':
        net = models.resnet18(pretrained=pretrain)
    elif net_name in 'efficientnet-b0':
        net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=class_number)
    elif net_name in 'efficientnet-b1':
        net = EfficientNet.from_pretrained('efficientnet-b1', num_classes=class_number)
    elif net_name in 'efficientnet-b2':
        net = EfficientNet.from_pretrained('efficientnet-b2', num_classes=class_number)
    elif net_name in 'efficientnet-b3':
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=class_number)
    elif net_name in 'efficientnet-b4':
        net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=class_number)
    elif net_name in 'efficientnet-b5':
        net = EfficientNet.from_pretrained('efficientnet-b5', num_classes=class_number)
    elif net_name in 'efficientnet-b6':
        net = EfficientNet.from_pretrained('efficientnet-b6', num_classes=class_number)
    elif net_name in 'efficientnet-b7':
        net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=class_number)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
    elif weight_path is not None:
        if class_number is not None and 'efficientnet' not in net_name:
            fc_features = net.fc.in_features
            net.fc = nn.Linear(fc_features, class_number)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net.load_state_dict(torch.load(weight_path, map_location=device))

    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def get_cam_for_error(args, net, cam_image_path, original_image_path, check_point_path):
    # 输入
    image_dict = {}
    img = io.imread(original_image_path)
    # 保存原图
    image_dict['origin'] = img
    width, height = img.shape

    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)
    # 输出图像

    # 网络
    model_path = os.path.join(check_point_path, args.date, args.time + '.pth')
    if net == None:
        net = get_net(args.network, args.class_number, model_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    # grad_cam = GradCAM(net, layer_name)
    # mask = grad_cam(inputs, args.class_id)  # cam mask
    # image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    # grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    # image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    cam_plus_plus, heatmap = gen_cam(img, mask_plus_plus)
    image_dict['cam++'] = cv2.resize(cam_plus_plus, (width, height))
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    # gbp = GuidedBackPropagation(net)
    # inputs.grad.zero_()  # 梯度置零
    # grad = gbp(inputs)
    #
    # gb = gen_gb(grad)
    # image_dict['gb'] = gb
    # 生成Guided Grad-CAM
    # cam_gb = gb * mask[..., np.newaxis]
    # image_dict['cam_gb'] = norm_image(cam_gb)
    # # return image_dict

    image_save_root = os.path.join(cam_image_path, args.date)
    if not Path(image_save_root).exists():
        os.mkdir(image_save_root)
    image_save_directory = os.path.join(cam_image_path, args.date, args.time)


    save_image(image_dict, os.path.basename(original_image_path), args.network, image_save_directory)


def call_get_cam(args):
    configer = Configer().get_configer()
    cam_image_path = configer['camImagePath']
    utils.make_directory(cam_image_path)
    image_save_directory = os.path.join(cam_image_path, args.date, args.time)
    utils.make_directory(image_save_directory)
    check_point_path = configer['checkPointPath']
    test_log = os.path.join(configer['logpath'], args.date, args.time + '_test.log')
    data_dict = utils.get_dict_from_json(test_log)
    error_file_list = data_dict['ERROR LIST']
    right_file_list = data_dict['RIGHT LIST']

    model_path = os.path.join(check_point_path, args.date, args.time + '.pth')
    net = get_net(args.network, args.class_number, model_path)
    for error_image in tqdm(error_file_list):
        original_test_image = os.path.join(configer['testImagePath'], error_image + '.jpg')
        get_cam_for_error(args, net, cam_image_path, original_test_image, check_point_path)

    # error_file_list_length = len(error_file_list)
    # image_num_loop = 20
    # loops = int(error_file_list_length / image_num_loop)
    # for i in range(0, loops - 1):
    #     cam_images_list = []
    #     error_file_list_sliced = error_file_list[i * image_num_loop: i * image_num_loop + image_num_loop].copy()
    #     total_list_length = len(error_file_list)
    #     for error_image in error_file_list_sliced:
    #         original_test_image = os.path.join(configer['testImagePath'], error_image + '.jpg')
    #         cam_dict = get_cam_for_error(args, cam_image_path, original_test_image, check_point_path)
    #         cam_images_list.append(cam_dict)
    #     plt.figure(1)
    #     for cam_dict in cam_images_list:
    #         dict_length = len(cam_dict)
    #         idx = 1
    #         for cam_name, image in cam_dict.items():
    #             plt.subplot(total_list_length, dict_length, idx)
    #             plt.imshow(image)
    #             idx += 1
    #     plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50',
                        help='ImageNet classification network')

    parser.add_argument('--date', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--time', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='output directory to save results')
    parser.add_argument('--class-number', type=int, default='results',
                        help='class number')
    arguments = parser.parse_args()

    call_get_cam(arguments)
