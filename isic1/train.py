import torchvision
import torch
import pdb
import matplotlib
import torch.nn as nn
from data.datarecorder import DataRecorder
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

transforms = utils.get_transforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

isic = ISICDataset(args, transforms)
ld = DataLoader(isic, batch_size=2, shuffle=True, drop_last=True)
optimizer = model.optimizer
criteria = model.loss_function
logger.start_record()
loss_list_draw = []
loss_dict_draw = {}
for EPOCH in range(args.epoch):
    for x, y in ld:
        loss_dict_print = {}
        x = x.to(device)
        y = torch.argmax(y, dim=1)
        y_hat = model.network(x.view(2, 3, 767, 1022).float())
        loss = criteria(y_hat, y.long().to(device))
        # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
        loss_list_draw.append(loss.item())
        loss_dict_print['EPOCH'] = EPOCH
        loss_dict_print['cross_loss'] = loss
        # loss_dict_print，没有epoch,都是损失函数名:值（值是list）
        visualizer.get_data_report()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

loss_dict_draw['cross_loss'] = loss_list_draw
visualizer.draw_picture_block(loss_dict_draw)
