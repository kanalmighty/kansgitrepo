import pandas
from options.base_options import BaseOptions
import utils
import pdb


class DataProber:
    def __init__(self, image_root_path, label_path):
        self.image_root_path = image_root_path
        self.label_path = label_path

    def get_size_profile(self):
        image_path_list = utils.get_image_set(self.image_root_path)
        pdb.set_trace()
        image_size_dict = {}
        for image_path in image_path_list:
            image = utils.get_image(image_path)
            width, height = image.size
            dict_key = 'w'+str(width)+'h'+str(height)
            if not dict_key in image_size_dict.keys():
                image_size_dict['w' + str(width) + 'h' + str(height)] = 1
            else:
                image_size_dict['w' + str(width) + 'h' + str(height)] += 1
        print(image_size_dict)


if __name__ == '__main__':
    dp = DataProber('D:\\pycharmspace\\datasets\\isic2019\\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Training_GroundTruth.csv')
    dp.get_size_profile()




