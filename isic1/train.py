import torchvision
import torch
import pdb
import matplotlib
import torch.nn as nn
from data.datarecorder import DataRecorder
from data.dataprober import DataProber
import utils
from models.model import Model
from options.base_options import BaseOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
from visualizer.visualizer import Visualizer
# model = torchvision.models.resnet18(pretrained=True).cuda()
options = BaseOptions()
logger = DataRecorder()
visualizer = Visualizer()
args = options.get_args()
model = Model(args)
dataprober = DataProber(args.datapath, args.labelpath)
dataprober.get_size_profile()
dataprober.get_type_profile()
transforms = utils.get_transforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

isic = ISICDataset(args, transforms)
# isic.__assert_equality__()
ld = DataLoader(isic, batch_size=args.batchsize, shuffle=True, drop_last=True)
optimizer = model.optimizer
criteria = model.loss_function
logger.start_record()
loss_list_draw = []
loss_dict_draw = {}
for EPOCH in range(args.epoch):
    loss_total_per_epoch = 0#记录每个epoch,所有batch的loss总和
    for idx, (x, y) in enumerate(ld):
        loss_dict_print = {}
        x = x.to(device)
        y = torch.argmax(y, dim=1)
        y_hat = model.network(x.view(args.batchsize, 3, 500, 500).float())
        loss = criteria(y_hat, y.long().to(device))
        loss_total_per_epoch += loss.item()#获取所有batch的loss总和
        # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
        loss_dict_print['EPOCH'] = EPOCH
        loss_dict_print['cross_loss'] = loss.item()
        # loss_dict_print，没有epoch,都是损失函数名:值（值是list）
        visualizer.get_data_report(loss_dict_print)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg_per_epoch = loss_total_per_epoch/(args.batchsize*idx)#获取这个epoch中一个平input的均loss
    loss_list_draw.append(loss_avg_per_epoch)

loss_dict_draw['cross_loss'] = loss_avg_per_epoch
visualizer.draw_picture_block(loss_dict_draw)
