import argparse

import cv2
from torch import optim
from torchsummary import summary
# from nets.CSPdarknet import  *
# from torchsummary import summary
# net = darknet53(pretrained=False)
# summary(net.cuda(), input_size=(3, 608, 608))
from torch.utils.data.dataset import Dataset
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')

    args = parser.parse_args()
    cfg = get_cfg()
    net = CSPdarknetGA()
    data_set = VocDataset(cfg.train_label_path)
    dl = DataLoader(data_set, batch_size=args.batchSize, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=5e-4)
    step = len(dl)/args.batchSize
    loss_f = nn.CrossEntropyLoss()
    for Epoch in range(60):
        epoch_accuracy = 0
        epoch_loss = 0
        for image, label in dl:
            pred = net(image)
            label = label.long().to(device)
            batch_mean_accurcy = (torch.argmax(pred, dim=1) == label).sum()/args.batchSize
            epoch_accuracy += batch_mean_accurcy
            batch_mean_loss = loss_f(pred, label)
            epoch_loss += batch_mean_loss.item()
            batch_mean_loss.backward()
            optimizer.step()
        print('epoch %d end,avarage accuracy %d,avarage loss %d'%(Epoch,epoch_accuracy/step,epoch_loss/step))
        # a = cv2.imread('D:\\datasets\\voc\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\002448.jpg')
    # cv2.imshow('a',a)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # cv2.destroyWindow(wname)




