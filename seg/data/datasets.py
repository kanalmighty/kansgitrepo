from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils import *
import pandas as pd
import torch
import os
import cv2
import numpy as np

from utils import get_sample


class FaceSegDateset(Dataset):
    def __init__(self, mode, image_root_path, label_file, *resize):
        self.label_list = get_sample(label_file)
        self.imagesize = resize
        self.image_root_path = image_root_path
        self.mode = mode

    # def __getitem__(self, item):
    #     # data_image = cv2.imread(os.path.join(self.image_root_path, self.label_list[item][0]))
    #     # data_image = Image.fromarray(cv2.cvtColor(data_image, cv2.COLOR_RGBA2RGB))
    #     data_image = Image.open(os.path.join(self.image_root_path, self.label_list[item][0]))
    #     data_image = data_image.convert("RGB")
    #     label_image = Image.open(os.path.join(self.image_root_path, self.label_list[item][1]))
    #     # data_tensor = torch.from_numpy(data_ndarray)
    #     # label_tensor = torch.from_numpy(label_ndarray)

    #     return self.transform(data_image), self.transform(label_image)

    def __getitem__(self, item):
        if self.mode == 'train':
            img = Image.open(os.path.join(self.image_root_path, self.label_list[item][0]))
            img = img.convert('RGB')
            label = Image.open(os.path.join(self.image_root_path, self.label_list[item][1]))
            label = label.convert('RGB')
            auto_augment = AutoAugment()  # 初始化数据增强器
            img = auto_augment(img)

            label = auto_augment(label)

            img = np.array(img)
            # img = cv2.imread(os.path.join(self.image_root_path, self.label_list[item][0]))
            img = cv2.resize(img, (self.imagesize[1], self.imagesize[0]), interpolation=cv2.INTER_NEAREST).astype(
                np.float)
            # label = cv2.imread(os.path.join(self.image_root_path, self.label_list[item][1]), 0)
            label = np.array(label)[:, :, 0]
            label = cv2.resize(label, (self.imagesize[1], self.imagesize[0]), interpolation=cv2.INTER_NEAREST)

            _, label = cv2.threshold(label, 1, 1, cv2.THRESH_TRUNC)
            # randoffsetx = np.random.randint(self.imagesize - self.cropsize)
            # randoffsety = np.random.randint(self.imagesize - self.cropsize)
            # img = img[randoffsety:randoffsety+self.cropsize,randoffsetx:randoffsetx+self.cropsize]
            # label = label[randoffsety:randoffsety+self.cropsize,randoffsetx:randoffsetx+self.cropsize]
            label = np.eye(2)[label]

            label = label.transpose(2, 0, 1).astype(np.float)

            img = img.transpose(2, 0, 1).astype(np.float)
            img = torch.from_numpy(img)
            label = torch.from_numpy(label)
            return img.float(), label.float()
        else:
            img = cv2.imread(os.path.join(self.image_root_path, self.label_list[item][0]))
            label = cv2.imread(os.path.join(self.image_root_path, self.label_list[item][1]), 0)
            img = cv2.resize(img, (self.imagesize[1], self.imagesize[0]), interpolation=cv2.INTER_NEAREST).astype(
                np.float)
            label = cv2.resize(label, (self.imagesize[1], self.imagesize[0]), interpolation=cv2.INTER_NEAREST)
            _, label = cv2.threshold(label, 1, 1, cv2.THRESH_TRUNC)
            label = np.eye(2)[label]

            label = label.transpose(2, 0, 1).astype(np.float)

            img = img.transpose(2, 0, 1).astype(np.float)
            img = torch.from_numpy(img)
            label = torch.from_numpy(label)
            return img.float(), label.float()

    def __len__(self):
        return len(self.label_list)

