import argparse
import datetime
import torchvision
import torch
import pdb
import os
import time
import matplotlib.pyplot as plt
import torch.nn as nn

from data.datarecorder import DataRecorder
from data.datasets import FaceSegDateset
from models.networks.qingnet import *
import cv2
import numpy as np
from options.train_options import TrainingOptions
from options.configer import Configer
from tqdm import tqdm
from torch.utils.data import DataLoader
from visualizer.visualizer import Visualizer
import utils
configer = Configer().get_configer()#获取环境配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options = TrainingOptions()
visualizer = Visualizer()#初始化视觉展示器
# resize = [224, 256, 320]
# cof = [16, 32, 64, 128]
# down_up_conv = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
# down_up_conv = [(5, 5), (6, 6)]



def train(args):

    if args.resize % pow(2, args.downLayerNumber) != 0:
        raise ValueError("输入尺寸必须是%d的整数倍" % pow(2, args.downLayerNumber))

    logger = DataRecorder()#初始化记录器
    logger.set_arguments(vars(args))
    #获取参数配置
    original_size = configer['original_size'].split(',')

    original_size = list(map(int, original_size))
    # original_size = tuple((int(str.split(',')[0]), int(str.split(',')[1]))) for str in original_size
    mode = configer['mode']
    batch_size = int(configer['batch_size'])
    num_class = int(configer['num_class'])
    label_root_path = configer['labelRootPath']
    train_label_file = configer['trainLabelFile']
    threshold = configer['threshold']
    test_label_file = configer['testLabelFile']
    mask_root = configer['maskImageRoot']
    model_save_path = configer['checkPointPath']

    dataset = FaceSegDateset(mode, label_root_path, train_label_file, args.resize, args.resize)

    trainingdata_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loss_f = torch.nn.BCELoss()
    stage_dict = {'GroupDownConvLayer': args.downLayerNumber, 'GroupUpConvLayer': args.upLayerNumber}  # 个数
    net = Assembler(stage_dict, 3, num_class, args.cof)
    net = net.to(device)
    accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    total_length = len(trainingdata_loader)
    opm = torch.optim.Adam(net.parameters(), lr=args.learningRate, betas=(0.9, 0.999), eps=1e-8)
    # opm = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.005)
    start = datetime.datetime.now()
    model_static_dict = {}
    train_statics_dict = {}#record overall training statics
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for EPOCH in tqdm(range(int(args.epoch))):
        net.train()
        accurate_count_epoch = 0
        train_loss_epoch = 0
        test_accuracy_count_epoch = 0
        for x, y in trainingdata_loader:
            x = x.to(device)
            y = y.to(device)

            pred = net(x)
            pred = torch.sigmoid(pred)
            pred_mask = pred.cpu().data.numpy().copy()
            # 以通道为维度进行softmax计算三个通道同一位置的像素的分类概率输出结果为b,c,h,w
            # 用argmax合并通道，像素位置取二通道最大值
            pred_mask = np.argmax(pred_mask, axis=1)
            y_mask = y.cpu().data.numpy().copy()
            y_mask = np.argmax(y_mask, axis=1)
            loss = loss_f(pred, y.squeeze().float())
            acc_per_step = (y_mask == pred_mask).mean()
            accurate_count_epoch += acc_per_step
            # accurate_count_epoch += accuracy_count(y_mask,pred_mask)/(resize[0]**2*batchSize)#一个batch里面正确分率的像素个数比率
            opm.zero_grad()
            train_loss_epoch += loss.item()
            loss.to(device).backward()
            opm.step()
        train_loss_list.append(train_loss_epoch / total_length)
        accuracy_list.append(accurate_count_epoch / (total_length))

        net.eval()
        transform_list_test = []
        # transform_list_test.append(transforms.Resize(resize))
        # transform_list_test.append(transforms.ToTensor())
        # trans_test = transforms.Compose(transform_list_test)
        test_dataset = FaceSegDateset('test', label_root_path,
                                      test_label_file, args.resize, args.resize)

        testdata_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        total_test_length = len(testdata_loader)

        with torch.no_grad():
            for idx, (x1, y1) in enumerate(testdata_loader):
                x1 = x1.to(device)
                y1 = y1.to(device)
                test_pred = torch.sigmoid(net(x1))

                test_pred = test_pred.cpu().data.numpy().copy()
                # 以通道为维度进行softmax计算三个通道同一位置的像素的分类概率输出结果为b,c,h,w
                # 用argmax合并通道，像素位置取二通道最大值
                test_pred_mask = np.argmax(test_pred, axis=1)
                test_y = y1.cpu().data.numpy().copy()
                test_y_mask = np.argmax(test_y, axis=1)
                test_acc_per_step = (test_y_mask == test_pred_mask).mean()
                test_accuracy_count_epoch += test_acc_per_step
                # test_accuracy_count_epoch += accuracy_count(y1.squeeze().cpu(),test_pred_argmax.cpu())/(resize[0]**2)#一个batch里面正确分率的像素个数

                # 最后一个epoch时验证效果开始
                if EPOCH == args.epoch - 1:
                    test_pred = np.argmax(test_pred, axis=1)
                    test_pred = test_pred.transpose(1, 2, 0)
                    test_pred = cv2.resize(test_pred, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST).astype(np.float)
                    test_pred = test_pred.astype(np.uint8)
                    # test_pred = np.expand_dims(test_pred, 2).repeat(3, axis=2)
                    # test_grey_image = cv2.cvtColor(test_pred, cv2.COLOR_BGR2GRAY)
                    # ret, mask_bin = cv2.threshold(test_grey_image, 127, 255, cv2.THRESH_TRUNC)
                    test_image = x1.squeeze().cpu().numpy().transpose(1, 2, 0)

                    contours, hierarchy = cv2.findContours(test_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    test_image_contour = cv2.drawContours(test_image, contours, -1, (0, 0, 255), 1)
                    test_image_contour = cv2.resize(test_image_contour, (original_size[0], original_size[1]))
                    mask_result_path = os.path.join(mask_root, start_time)
                    utils.make_directory(mask_result_path)
                    cv2.imwrite(os.path.join(mask_result_path, str(idx) + '.jpg'), test_image_contour)

                    #保存模型开始选择最优模型
                    key = str(args.resize) + '_' + str(args.cof) + '_' + str(args.downLayerNumber) + str(args.upLayerNumber)
                    # torch.save(net.state_dict(), os.path.join(model_save_path, key + '.pth'))
                    static_dict = vars(args).copy()
                    static_dict.pop('epoch')
                    static_dict.pop('learningRate')
                    # static_dict['test_acc'] = test_accuracy_list
                    model_static_dict[key] = static_dict


            test_accuracy_list.append(test_accuracy_count_epoch / (total_test_length))

        # 第二个epoch开始计算acc增长率

        if EPOCH >= 1:
            is_promising = utils.check_acc_rate(test_accuracy_list, threshold, args.epoch - EPOCH)
            if is_promising is not True:
                return False




    end = datetime.datetime.now()
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('down : %s, up :%s,size : %s, cof: %s, lr: %s, duration: %s' % (args.downLayerNumber, args.upLayerNumber, args.resize, args.cof, args.learningRate, (end-start).seconds))
    plt.xlabel = 'epoch'
    plt.ylabel = 'train_test_acc'
    plt.axis([0, len(accuracy_list), 0, 1])
    plt.yticks(np.arange(0, 1, 0.05))
    plt.plot(range(int(args.epoch)), test_accuracy_list, 'r-', label='test_acc')
    plt.plot(range(int(args.epoch)), accuracy_list, 'b-', label='train_acc')
    plt.legend(['test_acc', 'train_acc'])
    plt.subplot(122)
    plt.title('down : %s, up :%s,size : %s, cof: %s, lr: %s, duration: %s' % (args.downLayerNumber, args.upLayerNumber, args.resize, args.cof, args.learningRate, (end-start).seconds))
    plt.xlabel = 'epoch'
    plt.ylabel = 'train_loss'
    plt.axis([0, len(train_loss_list), 0, 3])
    plt.yticks(np.arange(0, 1, 0.1))
    plt.plot(range(int(args.epoch)), train_loss_list, 'g-', label='train_acc')
    plt.legend(['train_loss'])
    image_save_path = configer['staticImagePath']
    fig.savefig(os.path.join(image_save_path, start_time + '.png'), dpi=300, facecolor='gray')
    train_statics_dict['mean_train_accuracy'] = round(np.mean(accuracy_list), 3)
    train_statics_dict['mean_loss'] = round(np.min(train_loss_list), 3)
    train_statics_dict['mean_test_accuracy'] = round(np.mean(test_accuracy_list), 3)
    train_statics_dict['graph'] = start_time + '.png'
    train_statics_dict['duration_seconds'] = (end-start).seconds
    logger.set_training_data(train_statics_dict)
    logger.write_training_data()
    return model_static_dict


def set_args_final_train(args, primer_args):
    args_list = {'resize', 'cof', 'downLayerNumber', 'upLayerNumber'}
    least_relevent_arg_set = set(primer_args.keys()).difference(args_list)  # 看看最优参数列表中没有哪几个参数
    for least_relevent_arg in least_relevent_arg_set:
        if least_relevent_arg == 'cof':
            args.cof = sorted(list(map(int,configer['cof'].split(';'))))[0]
        elif least_relevent_arg == 'resize':
            args.resize = sorted(list(map(int, configer['resize'].split(';'))))[0]
        elif least_relevent_arg == 'downLayerNumber':
            d_u_lay_number = configer['down_up_conv'].split(';')
            args.downLayerNumber = sorted([tuple((int(str.split(',')[0]), int(str.split(',')[1]))) for str in d_u_lay_number])[0]
        elif least_relevent_arg == 'upLayerNumber':
            d_u_lay_number = configer['down_up_conv'].split(';')
            args.upLayerNumber = sorted([tuple((int(str.split(',')[0]), int(str.split(',')[1]))) for str in d_u_lay_number])[1]
    return args

# resize = 256; 320; 384
# cof = 16; 32; 64; 128
# down_up_conv = 2, 2; 3, 3; 4, 4; 5, 5; 6, 6
# original_size = 800,600

def auto_search():
    args = options.get_args()#获取参数
    resize_list = configer['resize'].split(';')
    epoch = int(configer['epoch'])
    resize_list = list(map(int, resize_list))
    cof_list = configer['cof'].split(';')
    cof_list = list(map(int, cof_list))
    d_u_lay_number = configer['down_up_conv'].split(';')
    d_u_lay_number = [tuple((int(str.split(',')[0]), int(str.split(',')[1]))) for str in d_u_lay_number]
    args.epoch = epoch / 2
    res_dict= {}
    for r in resize_list:
        for c in cof_list:
            for du in d_u_lay_number:
                args.resize = r
                args.cof = c
                args.downLayerNumber = du[0]
                args.upLayerNumber = du[1]
                try:
                    res = train(args)
                    if res is False:
                        print("参数组合%s达不到指定阈值,丢弃" % vars(args))
                    else:
                        print("参数组合%s完成预训练，训练结束后进行模型评估" % vars(args))
                        res_dict.update(res)
                except (RuntimeError) as e:
                    print("参数组合%s完成训练失败，原因:%s" % (vars(args), e))
    # 调用关联规则挖掘方法,获取最优参数组合
    primer_list = utils.get_feq_set(res_dict)
    dict_map = map(utils.sort_arg_dict, primer_list)
    #预设定的参数列表
    print("完成参数组合筛选",dict_map)
    for primer_args in dict_map:
        args = set_args_final_train(args, primer_args)

        try:
            res_final = train(args)
            if res_final is False:
                print("参数组合%s达不到指定阈值,丢弃" % vars(args))
            else:
                print("参数组合%s完成最好评估" % vars(args))
        except (RuntimeError, TypeError, ValueError) as e:
            print("参数组合%s完成训练失败，原因:%s" % (vars(args), e))


if __name__ == '__main__':
    auto_search()
