import torchvision
import torch
import pdb
import matplotlib
import torch.nn as nn
import utils
from models.model import Model
from options.base_options import BaseOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
# model = torchvision.models.resnet18(pretrained=True).cuda()
options = BaseOptions()

args = options.get_args()
model = Model(args)

transforms = utils.get_transforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

isic = ISICDataset(args, transforms)
ld = DataLoader(isic, batch_size=2, shuffle=True,drop_last=True)
optimizer = model.optimizer
criteria = model.loss_function
loss_array = []
for EPOCH in range(args.epoch):
    for x, y in ld:
        x = x.to(device)
        y = torch.argmax(y, dim = 1)
        y_hat = model.network(x.view(2, 3, 767, 1022).float())
        loss = criteria(y_hat, y.long().to(device))
        loss_array.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("loss_array = %s" % loss_array)
