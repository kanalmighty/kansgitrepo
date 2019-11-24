import pandas as pd
from options.configer import Configer
import glob
import utils
import pdb as pdb
import cv2
from tqdm import tqdm
from data.autoaugment import AutoAugment
from pathlib import Path
import os
from options.preprocess_options import PreprocessOptions

class DataPreProcesser():
    def __init__(self):
        self.configer = Configer().get_configer()
        self.auto_augment = AutoAugment()
        self.row_image_path = self.configer['rowImagePath']
        self.row_label_path = self.configer['rowLabelPath']
        row_label = glob.glob(os.path.join(self.configer['rowLabelPath'], '*.csv'))
        if len(row_label) != 1:
            raise ValueError('expect 1 csc file but got %s' + row_label.size)
        self.row_lable_dataframe = pd.read_csv(row_label[0], header=0)

    def __call__(self, expected_number):
        self.check_all_paths()
        self.data_pre_process(expected_number)

    def check_all_paths(self):
        utils.make_directory(self.configer['trainingImagePath'])
        utils.make_directory(self.configer['traininglabelPath'])
        if not Path(self.configer['rowImagePath']).exists():
            raise IOError(self.configer['rowImagePath'] + 'does not exist')
        if not Path(self.configer['rowLabelPath']).exists():
            raise IOError(self.configer['rowLabelPath'] + 'does not exist')

    def data_pre_process(self, image_number):
        lable_dataframe = pd.DataFrame(self.row_lable_dataframe)
        lable_dataframe = lable_dataframe.drop('UNK', axis=1)
        lable_dataframe_without_image = lable_dataframe.drop('image', axis=1)
        sum_dict = {}
        bias_dict = {}
        new_lable_dataframe = pd.DataFrame(columns=['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'])
        total_image_number = len(lable_dataframe.drop(0, axis=0))
        #get all class from row label csv
        class_list = lable_dataframe_without_image.columns.tolist()
        #count images that have been moved
        image_count_index = 0
        image_detail_dict = {}
        for column_name in lable_dataframe_without_image.columns:
            #字典{'列名':图片数量}
            sum_dict[column_name] = len(lable_dataframe[lable_dataframe[column_name].isin([1])]['image'].values.tolist())
            #get image list for a specfied class
            image_detail_dict[column_name] = lable_dataframe[lable_dataframe[column_name].isin([1])]['image'].values.tolist()
        #get the bais between expected number and actual number
        for k1, v1 in sum_dict.items():
            bias_dict[k1] = image_number - v1
        for k2, v2 in bias_dict.items():
            single_class_images = image_detail_dict[k2]
            if v2 > 0:
                pdb.set_trace()
                multitude = v2//sum_dict[k2]#取整
                left_over = v2 % sum_dict[k2]#取余
                #rename origin images
                print('copy original %s data' % k2)
                for image_name in tqdm(single_class_images):
                    image_path = os.path.join(self.configer['rowImagePath'], image_name + '.jpg')
                    image = utils.get_image(image_path)
                    image_count_index += 1
                    image_name_encoded = utils.encode_image_name(total_image_number, image_count_index)
                    encoded_file_path = os.path.join(self.configer['trainingImagePath'], k2 + image_name_encoded + '.jpg')
                    image.save(encoded_file_path)
                    # append label to label dataframe
                    onehot_dict = utils.get_onehot_by_class(class_list, k2)
                    onehot_dict['image'] = k2 + image_name_encoded
                    onehot_dict = pd.DataFrame(onehot_dict, index=[image_count_index])
                    new_lable_dataframe = new_lable_dataframe.append(onehot_dict)
                        #append augumented data
                print('multiply by augumented %s data' % k2)
                for _ in range(multitude):
                    for image_name in tqdm(single_class_images):
                        image_path = os.path.join(self.configer['rowImagePath'], image_name + '.jpg')
                        image = utils.get_image(image_path)
                        image_processed = self.auto_augment(image)
                        image_count_index += 1
                        image_name_encoded = utils.encode_image_name(total_image_number, image_count_index)
                        encoded_file_path = os.path.join(self.configer['trainingImagePath'], k2 + image_name_encoded + '.jpg')
                        image_processed.save(encoded_file_path)
                        # append label to label dataframe
                        onehot_dict = utils.get_onehot_by_class(class_list, k2)
                        onehot_dict['image'] = k2 + image_name_encoded
                        onehot_dict = pd.DataFrame(onehot_dict, index=[image_count_index])
                        new_lable_dataframe = new_lable_dataframe.append(onehot_dict)
                print('adding left augumented %s data' % k2)
                for _ in tqdm(single_class_images[:left_over]):
                    image_path = os.path.join(self.configer['rowImagePath'], image_name + '.jpg')
                    image = utils.get_image(image_path)
                    image_processed = self.auto_augment(image)
                    image_count_index += 1
                    image_name_encoded = utils.encode_image_name(total_image_number, image_count_index)
                    encoded_file_path = os.path.join(self.configer['trainingImagePath'], k2+image_name_encoded + '.jpg')
                    image_processed.save(encoded_file_path)
                    # append label to label dataframe
                    onehot_dict = utils.get_onehot_by_class(class_list, k2)
                    onehot_dict['image'] = k2 + image_name_encoded
                    onehot_dict = pd.DataFrame(onehot_dict, index=[image_count_index])
                    new_lable_dataframe = new_lable_dataframe.append(onehot_dict)
            else:
                print('remove redundant %s data' % k2)
                for image_name in tqdm(single_class_images[:image_number]):
                    image_path = os.path.join(self.configer['rowImagePath'], image_name + '.jpg')
                    image = utils.get_image(image_path)
                    image_count_index += 1
                    image_name_encoded = utils.encode_image_name(total_image_number, image_count_index)
                    encoded_file_path = os.path.join(self.configer['trainingImagePath'],
                                                     k2 + image_name_encoded + '.jpg')
                    image.save(encoded_file_path)
                    # append label to label dataframe
                    onehot_dict = utils.get_onehot_by_class(class_list, k2)
                    onehot_dict['image'] = k2 + image_name_encoded
                    onehot_dict = pd.DataFrame(onehot_dict, index=[image_count_index])
                    new_lable_dataframe = new_lable_dataframe.append(onehot_dict)
        new_lable_dataframe = new_lable_dataframe[['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC']]
        new_lable_dataframe.to_csv(os.path.join(self.configer['traininglabelPath'], 'processed_label.csv'),index=False)










if __name__ == '__main__':
    d = DataPreProcesser()
    d(10)



