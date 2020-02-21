from options.base_options import BaseOptions
import utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import random


class DataProber:
    def __init__(self, image_root_path, label_path):
        self.image_root_path = image_root_path
        self.label_path = label_path
        self.image_path_list = utils.get_image_set(self.image_root_path)

    def get_size_profile(self):
        image_size_dict = {}
        for image_path in self.image_path_list:
            image = utils.get_image(image_path)
            width, height = image.size
            dict_key = 'w'+str(width)+'h'+str(height)
            if not dict_key in image_size_dict.keys():
                image_size_dict['w' + str(width) + 'h' + str(height)] = 1
            else:
                image_size_dict['w' + str(width) + 'h' + str(height)] += 1
        print(image_size_dict)

    def get_type_profile(self):
        image_path_list = utils.get_image_set(self.image_root_path)
        image_type_dict = {}
        for image_path in image_path_list:
            filename, extension = os.path.splitext(image_path)
            if not extension in image_type_dict.keys():
                image_type_dict[extension] = 1
            else:
                image_type_dict[extension] += 1
        print(image_type_dict)

    def get_length_difference(self):
        #获取label里的文件名并转为set
        label_name_list = pd.read_csv(self.label_path, header=0, usecols=[0], skiprows=0, engine='python').values
        print('label number: ' + str(len(label_name_list)))
        print('image number: ' + str(len(self.image_path_list)))
        if len(label_name_list) != len(self.image_path_list):
            raise AssertionError('lables do not match data')

    def check_order(self, data_order_ndarray, label_order_ndarray):
        print(data_order_ndarray)
        print(label_order_ndarray)
        res = (data_order_ndarray == label_order_ndarray).all()
        if not res:
            raise ValueError('sample order doesnt match!')

    def get_label_histgram(self):
        df = pd.read_csv(self.label_path, header=0, index_col=0)
        Row_sum = df.apply(lambda x: x.sum())
        print(Row_sum)
        Row_sum.plot()
        plt.show()
        # plt.hist(list(Row_sum),bins=)

        # a = [random.randint(1, 10) for _ in range(10)]
        #
        # plt.hist(a, bins=10)
        # plt.show()







if __name__ == '__main__':
    # dp = DataProber('D:\\pycharmspace\\datasets\\isic2019\\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Test_GroundTruth.csv')
    # dp = DataProber('D:\\pycharmspace\\datasets\\isic2019\\image','D:\\pycharmspace\\datasets\\isic2019\\label\\ISIC_2019_Training_GroundTruth.csv')
    # dp = DataProber('D:\\pycharmspace\\datasets\\isic2019\\image','D:\\pycharmspace\\datasets\\isic2019\\csv\\ISIC_2019_Training_GroundTruth_All.csv')
    dp = DataProber('D:\\pycharmspace\\datasets\\isic2019\\image','D:\\pycharmspace\\datasets\\isic2019\\label_processed\\processed_label.csv')
    dp.get_label_histgram()








