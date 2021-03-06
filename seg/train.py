import datetime
import torchvision
import torch
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
configer = Configer().get_configer()#获取环境配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options = TrainingOptions()
visualizer = Visualizer()#初始化视觉展示器
args = options.get_args()#获取参数
if args.resize[0] % pow(2, args.downLayerNumber) != 0:
    raise ValueError("输入尺寸必须是%d的整数倍" % pow(2, args.downLayerNumber))

logger = DataRecorder()#初始化记录器
logger.set_arguments(vars(args))
label_root_path = configer['labelRootPath']
train_label_file = configer['trainLabelFile']
test_label_file = configer['testLabelFile']
mask_root = configer['maskImageRoot']
dataset = FaceSegDateset(args.mode, label_root_path, train_label_file, args.resize[0], args.resize[1])

trainingdata_loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
loss_f = torch.nn.BCELoss()
stage_dict = {'GroupDownConvLayer': args.downLayerNumber, 'GroupUpConvLayer': args.upLayerNumber}  # 个数
net = Assembler(stage_dict, 3, args.numclass, args.cof)
net = net.to(device)
accuracy_list = []
test_accuracy_list = []
train_loss_list = []
total_length = len(trainingdata_loader)
opm = torch.optim.Adam(net.parameters(), lr=args.learningRate, betas=(0.9, 0.999), eps=1e-8)
# opm = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.005)
original_size = (args.originalSize[1], args.originalSize[0])
start = datetime.datetime.now()
train_statics_dict = {}#record overall training statics
start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
for EPOCH in tqdm(range(args.epoch)):
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
        loss.backward()
        opm.step()
    train_loss_list.append(train_loss_epoch / total_length)
    accuracy_list.append(accurate_count_epoch / (total_length))

    net.eval()
    transform_list_test = []
    # transform_list_test.append(transforms.Resize(resize))
    # transform_list_test.append(transforms.ToTensor())
    # trans_test = transforms.Compose(transform_list_test)
    test_dataset = FaceSegDateset('test', label_root_path,
                                  test_label_file, args.resize[0], args.resize[1])

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
                test_pred = cv2.resize(test_pred, (args.resize[0], args.resize[1]), interpolation=cv2.INTER_NEAREST).astype(np.float)
                test_pred = test_pred.astype(np.uint8)
                # test_pred = np.expand_dims(test_pred, 2).repeat(3, axis=2)
                # test_grey_image = cv2.cvtColor(test_pred, cv2.COLOR_BGR2GRAY)
                # ret, mask_bin = cv2.threshold(test_grey_image, 127, 255, cv2.THRESH_TRUNC)
                test_image = x1.squeeze().cpu().numpy().transpose(1, 2, 0)

                contours, hierarchy = cv2.findContours(test_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                test_image_contour = cv2.drawContours(test_image, contours, -1, (0, 0, 255), 1)
                test_image_contour = cv2.resize(test_image_contour, (original_size[0], original_size[1]))
                cv2.imwrite(mask_root + str(idx) + '.jpg', test_image_contour)

        test_accuracy_list.append(test_accuracy_count_epoch / (total_test_length))
end = datetime.datetime.now()
fig = plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title('down : %s, up :%s,size : %s, cof: %s, lr: %s, duration: %s' % (args.downLayerNumber, args.upLayerNumber, args.resize, args.cof, args.learningRate, (end-start).seconds))
plt.xlabel = 'epoch'
plt.ylabel = 'train_test_acc'
plt.axis([0, len(accuracy_list), 0, 1])
plt.yticks(np.arange(0, 1, 0.05))
plt.plot(range(args.epoch), test_accuracy_list, 'r-', label='test_acc')
plt.plot(range(args.epoch), accuracy_list, 'b-', label='train_acc')
plt.legend(['test_acc', 'train_acc'])
plt.subplot(122)
plt.title('down : %s, up :%s,size : %s, cof: %s, lr: %s, duration: %s' % (args.downLayerNumber, args.upLayerNumber, args.resize, args.cof, args.learningRate, (end-start).seconds))
plt.xlabel = 'epoch'
plt.ylabel = 'train_loss'
plt.axis([0, len(train_loss_list), 0, 3])
plt.yticks(np.arange(0, 1, 0.1))
plt.plot(range(args.epoch), train_loss_list, 'g-', label='train_acc')
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
