import torchvision
import torch
import pdb
import matplotlib
import torch.nn as nn
from data.datarecorder import DataRecorder
from data.dataprober import DataProber
import utils
from models.model import Model
from options.configer import Configer
from options.train_options import TrainingOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
from data.autoaugment import *
from visualizer.visualizer import Visualizer
# model = torchvision.models.resnet18(pretrained=True).cuda()
options = TrainingOptions()
logger = DataRecorder()#初始化记录器
visualizer = Visualizer()#初始化视觉展示器
args = options.get_args()#获取参数
auto_augment = AutoAugment()#初始化数据增强器
args.augment_policy = auto_augment.policy_detail#记录数据增强策略
model = Model(args)#根据参数获取模型
configer = Configer().get_configer()#获取环境配置
dataprober = DataProber(configer['trainingImagePath'], configer['traininglabelPath'])#初始化数据探查器
# dataprober.get_data_difference()
transforms = utils.get_auto_augments(auto_augment) if args.autoaugment else utils.get_transforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_path = configer['trainingImagePath']
label_path = configer['trainingLabelPath']
isic = ISICDataset(image_path, label_path, transforms)
isic.__assert_equality__()
trainingdata_loader = DataLoader(isic, batch_size=args.batchsize, shuffle=True, drop_last=True)

optimizer = model.optimizer
criteria = model.loss_function
logger.set_arguments(vars(args))
epoch_loss_list = []
epoch_statics_dict = {}#record epochly training statics
train_statics_dict = {}#record overall training statics
model.train()
for EPOCH in range(args.epoch):
    loss_all_samples_per_epoch = 0#记录每个epoch,所有batch的loss总和
    train_accuracy = 0#trainnig accuaracy per epoch
    for idx, (x, y) in enumerate(trainingdata_loader):
        batch_statics_dict = {}
        x = x.to(device)
        y = torch.argmax(y, dim=1)
        y_hat = model.network(x.float())
        train_accuracy += (y.to(device) == torch.argmax(y_hat, dim=1)).sum().item()

        loss = criteria(y_hat, y.long().to(device))
        loss_all_samples_per_epoch += loss.item()#loss.item()获取的是每个batchsize的平均loss
        # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
        batch_statics_dict['EPOCH'] = EPOCH
        batch_statics_dict[args.lossfunction] = loss.item()
        # loss_dict_print，每个epoch,都是损失函数名:值（值是list）
        visualizer.get_data_report(batch_statics_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg_per_epoch = loss_all_samples_per_epoch/(idx+1)#获取这个epoch中一个平input的均loss,idx从0开始，所以需要加1
    train_accuracy_epoch = train_accuracy / len(isic)#training accuracy/sample numbers
    epoch_statics_dict['AVG LOSS'] = loss_avg_per_epoch
    epoch_loss_list.append(loss_avg_per_epoch)#record epoch loss for drawing
    epoch_statics_dict['TRAINING ACCURACY'] = train_accuracy_epoch
    print('epoch %s finished '% EPOCH)
    visualizer.get_data_report(epoch_statics_dict)
    train_statics_dict[EPOCH] = epoch_statics_dict
logger.set_training_data(train_statics_dict)
logger.write_training_data()
visualizer.draw_picture_block(epoch_loss_list)
pkl_name = model.save_model(logger.date_string, logger.start_time_string)
