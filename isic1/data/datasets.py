from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import utils
import pandas as pd
import torch
import os
import glob
import numpy as np
from data.dataprober import DataProber
from options.configer import Configer
from torchvision.transforms import transforms


class ISICDataset(Dataset):
    def __init__(self, image_path, label_path, transforms=None):
        self.image_dir = image_path
        self.label_path = label_path
        self.configer = Configer().get_configer()#获取环境配置
        self.image_path_list = utils.get_image_set(self.image_dir)
        self.image_path_list.sort()
        self.transforms = transforms
        label_df = utils.read_csv(self.label_path)
        label_df_sorted = label_df.sort_values('image')
        dp = DataProber(self.image_dir, self.label_path)
        #check data and lable length
        dp.get_length_difference()
        image_name_list = utils.get_filename_list(self.image_dir)
        dp.check_order(np.array(image_name_list), label_df_sorted['image'].values)
        # label_name_ndarray = label_df_sorted['image'].values
        label_ndarray = label_df_sorted.iloc[:, 1:].as_matrix()
        self.label_tensor = torch.from_numpy(label_ndarray)
        # self.image_array = utils.get_images(self.image_path_array)
        # self.image_array_trainsformed = transform(self.image_array)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = utils.get_image(image_path)
        # resize_img = transforms.Resize(128)
        # image = resize_img(image)
        if self.transforms:
            image = self.transforms(image)
        return (image, self.label_tensor[index])

    def __len__(self):
        return len(self.image_path_list)

    def __assert_equality__(self):
        print('assert equality images : %d ?= labels: %d ' % (self.__len__(), self.label_tensor.size()[0]))
        assert self.__len__() == self.label_tensor.size()[0]


if __name__ == '__main__':
    isic = ISICDataset('D:\\pycharmspace\\datasets\\isic2019\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Training_GroundTruth.csv')
    ld = DataLoader(isic, batch_size=2, shuffle=True)
    for x, y in ld:
        print('data  = %s,label = %s' % (x, y))