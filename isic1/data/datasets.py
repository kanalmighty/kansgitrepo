from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import utils
import pandas as pd
import torch
from torchvision.transforms import transforms

class ISICDataset(Dataset):
    def __init__(self, args, transforms):
        self.image_dir = args.datapath
        self.label_dir = args.labelpath
        self.image_path_list = utils.get_image_set(self.image_dir)
        self.transforms = transforms
        self.label_tensor = torch.from_numpy(utils.read_csv(self.label_dir))
        # self.image_array = utils.get_images(self.image_path_array)
        # self.image_array_trainsformed = transform(self.image_array)

    def __getitem__(self, index):

        image_path = self.image_path_list[index]
        image = utils.get_image(image_path)
        image = image.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return (image, self.label_tensor[index])

    def __len__(self):
        return len(self.image_path_list)

    def __assert_equality__(self):
        print('assert equality images : %d ?= labels: %d ' % (self.__len__(), self.label_tensor.size()[0]))
        assert self.__len__() == self.label_tensor.size()[0]

    # def get_trainsforms(opt):
    #     transform_list = []
    #     if opt.Normalize:
    #         transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
    #                                                    (0.5, 0.5, 0.5)))
    #         transform_list.append(transforms.ToTensor)
    #     return transforms.Compose(transform_list)

if __name__ == '__main__':
    isic = ISICDataset('D:\\pycharmspace\\datasets\\isic2019\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Training_GroundTruth.csv')
    ld = DataLoader(isic, batch_size=2, shuffle=True)
    for x, y in ld:
        print('data  = %s,label = %s' % (x, y))