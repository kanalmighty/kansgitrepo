import torchvision
import torch
import torch.nn as nn
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
model = torchvision.models.resnet18(pretrained=True)
isic = ISICDataset('D:\\pycharmspace\\datasets\\isic2019\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Training_GroundTruth.csv')
ld = DataLoader(isic, batch_size=2, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),0.003)
criteria = nn.CrossEntropyLoss()
loss_array = []
for EPOCH in range(1):
    for x, y in ld:
        y_hat = model(x.view(2, 3, 767, 1022).float())
        loss = criteria(y_hat, y.long())
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
