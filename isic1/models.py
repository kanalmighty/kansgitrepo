import torchvision
import torch
import torch.nn as nn
import utils
from options.base_options import BaseOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
model = torchvision.models.resnet18(pretrained=True).cuda()
options = BaseOptions()

args = options.get_args()
transforms = utils.get_trainsforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

isic = ISICDataset(args, transforms)
ld = DataLoader(isic, batch_size=2, shuffle=True,)
optimizer = torch.optim.Adam(model.parameters(), 0.003)
criteria = nn.CrossEntropyLoss()
loss_array = []
for EPOCH in range(1):
    for x, y in ld:
        x = x.to(device)
        y_hat = model(x.view(2, 3, 767, 1022).float())
        loss = criteria(y_hat, torch.cuda.LongTensor(y))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
