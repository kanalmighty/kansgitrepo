import argparse
import shutil

import cv2
from torch import optim
from torchsummary import summary
# from nets.CSPdarknet import  *
# from torchsummary import summary
# net = darknet53(pretrained=False)
import sys
# summary(net.cuda(), input_size=(3, 608, 608))
from torch.utils.data.dataset import Dataset

sys.path.append('/content/cloned-repo/yolov4-pytorch')
from cfg import *
import random

from nets.CSPdarknet import *
from model_prune.utils import *
from torch.utils.data import DataLoader



class VocDataset(Dataset):
    def __init__(self, label_path,transforms=None):

        label_list = get_image_annotation(label_path)
        random.shuffle(label_list)
        self.label_list = label_list
        self.transforms = transforms


    def __getitem__(self, index):
        label_dict = self.label_list[index]
        for k, v in label_dict.items():
            whole_image = cv2.imread(k)
            #bounding - box（包含左下角和右上角xy坐标）
            annotation_list = v.strip('\n').split(',')
            annotation_list = list(map(int, annotation_list))
            image = whole_image[annotation_list[1]: annotation_list[3], annotation_list[0]:annotation_list[2]]
            # cv2.imshow('a', image)
            class_label = annotation_list[4]
            image = cv2.resize(image, (int(608/4),int(608/4)), interpolation=cv2.INTER_NEAREST)
            return torch.cuda.FloatTensor(image).permute(2, 0, 1), class_label


    def __len__(self):
        return len(self.label_list)

class CSPdarknetGA(CSPDarkNet):
    def __init__(self):
        super(CSPdarknetGA, self).__init__([1, 2, 8, 8, 4])
        self.conv512 = nn.Conv2d(512,20,1)
        self.conv256 = nn.Conv2d(256, 20, 1)
        self.conv1024 = nn.Conv2d(1024, 20, 1)


    def forward(self, x):
        #out5 1024/5,out3 256/19,out4 512/10
        out3, out4 ,out5 = super(CSPdarknetGA, self).forward(x)
        out5 = torch.nn.functional.adaptive_avg_pool2d(self.conv1024(out5), (1,1))
        out4 = torch.nn.functional.adaptive_avg_pool2d(self.conv512(out4), (1,1))
        out3 = torch.nn.functional.adaptive_avg_pool2d(self.conv256(out3), (1,1))
        out = ((out3 + out4 + out5).squeeze())/3
        return out



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--epoch', type=int, default=120,
                        help='scale sparse rate (default: 0.5)')

    args = parser.parse_args()
    cfg = get_cfg()
    net = CSPdarknetGA()
    train_data_set = VocDataset(cfg.train_label_path)
    test_data_set = VocDataset(cfg.test_label_path)
    train_dl = DataLoader(train_data_set, batch_size=args.batchSize, drop_last=True)
    test_dl = DataLoader(test_data_set, batch_size=32, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=5e-4)
    if os.path.isfile(cfg.prune_model_path):
        print("=> loading checkpoint '{}'".format(cfg.prune_model_path))
        checkpoint = torch.load(cfg.prune_model_path)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) train_acc: {:f} test_acc: {:f}"
              .format(cfg.prune_model_path, checkpoint['epoch'], checkpoint['train_accuracy'], checkpoint['test_accuracy']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.prune_model_path))
        start_epoch = 0

    net.train()
    optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=5e-4)
    step = len(train_data_set)/args.batchSize
    test_step = len(test_data_set) / 32
    loss_f = nn.CrossEntropyLoss()
    for Epoch in range(start_epoch, args.epoch):
        print("start training at epoch %d" % Epoch)
        net.train()
        epoch_train_accuracy = 0
        epoch_test_accuracy = 0
        epoch_loss = 0
        for image, label in train_dl:
            pred = net(image)
            label = label.long().to(device)
            batch_mean_accurcy = (torch.argmax(pred, dim=1) == label).float().sum()/args.batchSize
            epoch_train_accuracy += batch_mean_accurcy.item()
            batch_mean_loss = loss_f(pred, label)
            epoch_loss += batch_mean_loss.item()
            batch_mean_loss.backward()
            #给bn层的gamma系统加上L1正则化先
            # for m in net.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.weight.grad.data.add_(args.percent * torch.sign(m.weight.data))  # L1
            #network smling只更新非resblock层里面的bn
            for name, layer in net.named_modules():
                if len(name.split('.')) < 7:
                    for i in layer.modules():
                        if isinstance(i, nn.BatchNorm2d):
                            i.weight.grad.data.add_(args.percent * torch.sign(i.weight.data))
            optimizer.step()
        print('epoch %d end,avarage train accuracy %f,avarage loss %f' % (Epoch, epoch_train_accuracy / step, epoch_loss/ step))
        print('start evaluate')
        with torch.no_grad():
            for test_image, test_label in test_dl:
                test_pred = net(test_image)
                test_label = test_label.long().to(device)
                test_mean_accurcy = (torch.argmax(test_pred, dim=1) == test_label).float().sum() / 32
                epoch_test_accuracy += test_mean_accurcy.item()

        print('avarage test accuracy %f' % (epoch_test_accuracy / test_step))




        print('saving model state')

        save_checkpoint({
            'epoch': Epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_accuracy': epoch_train_accuracy / step,
            'test_accuracy': epoch_test_accuracy / test_step,
        },cfg.prune_model_path)



