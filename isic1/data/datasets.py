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
        # self.image_array = utils.get_images(self.image_path_array)
        # self.image_array_trainsformed = transform(self.image_array)

    def __getitem__(self, index):
        label_dataframe = pd.read_csv(self.label_dir)
        #把dataframe转换为ndarray
        label_ndarray = label_dataframe.iloc[:, 1:].as_matrix()
        self.label_tensor = torch.from_numpy(label_ndarray)
        image_path = self.image_path_list[index]
        image = utils.get_image(image_path)
        image_transformed = self.transforms(image)
        return (image_transformed, self.label_tensor[index])

    def __len__(self):
        return len(self.image_path_list)

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