import pandas as pd
from options.configer import Configer
import glob
import utils
import cv2 as cv
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
        self.temp_image_path = self.configer['tempImagePath']
        self.training_image_path = self.configer['trainingImagePath']
        row_label = glob.glob(os.path.join(self.configer['rowLabelPath'], '*.csv'))
        if len(row_label) != 1:
            raise ValueError('expect 1 csc file but got %s' + row_label.size)
        self.row_lable_dataframe = pd.read_csv(row_label[0], header=0)

    def __call__(self, expected_number):
        self.check_all_paths()
        self.save_bordercroped_images(0.55)
        self.save_centercropsed_images()

        self.extend_dataset(expected_number)
        self.pad_images(500)

    def check_all_paths(self):
        utils.make_directory(self.configer['trainingImagePath'])
        utils.make_directory(self.configer['traininglabelPath'])
        utils.make_directory(self.configer['tempImagePath'])
        if not Path(self.configer['rowImagePath']).exists():
            raise IOError(self.configer['rowImagePath'] + 'does not exist')
        if not Path(self.configer['rowLabelPath']).exists():
            raise IOError(self.configer['rowLabelPath'] + 'does not exist')

    def extend_dataset(self, image_number):
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
                multitude = v2//sum_dict[k2]#取整
                left_over = v2 % sum_dict[k2]#取余
                #rename origin images
                print('\n copy original %s data' % k2)
                for image_name in tqdm(single_class_images):
                    image_path = os.path.join(self.configer['tempImagePath'], image_name + '.jpg')
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
                print('\n multiply by augumented %s data' % k2)
                for _ in range(multitude):
                    for image_name in tqdm(single_class_images):
                        image_path = os.path.join(self.configer['tempImagePath'], image_name + '.jpg')
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
                print('\n adding left augumented %s data' % k2)
                for _ in tqdm(single_class_images[:left_over]):
                    image_path = os.path.join(self.configer['tempImagePath'], image_name + '.jpg')
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
                print('\n removing redundant %s data' % k2)
                for image_name in tqdm(single_class_images[:image_number]):
                    image_path = os.path.join(self.configer['tempImagePath'], image_name + '.jpg')
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
        new_lable_dataframe = new_lable_dataframe[['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']]
        new_lable_dataframe.to_csv(os.path.join(self.configer['traininglabelPath'], 'processed_label.csv'), index=False)

    def save_bordercroped_images(self, lmda):
        dataset = utils.get_image_set(self.row_image_path)
        print('crop image border by %s' % lmda)
        for image in tqdm(dataset):
            image_name = image.split(os.sep)[-1]
            image = cv.imread(image)
            w, h = image.shape[0], image.shape[1]
            w_hat = w * lmda
            h_hat = h * lmda
            x_start = round((w - w_hat) / 2)
            x_end = round((w + w_hat) / 2)
            y_start = round((h - h_hat) / 2)
            y_end = round((h + h_hat) / 2)
            croped = image[x_start: x_end, y_start: y_end, :]
            cv.imwrite((os.path.join(self.temp_image_path, image_name)), croped)

    def save_centercropsed_images(self):
        dataset = utils.get_image_set(self.temp_image_path)
        print('centercrop begins')
        for image in tqdm(dataset):
            image_name = image.split(os.sep)[-1]
            image = cv.imread(image)
            grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, binary = cv.threshold(grey, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
            # kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            # 先开操作，取除噪声
            opened_image = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel_open, iterations=5)
            # 填充内部空隙
            closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, kernel_close, iterations=5)
            dst1 = cv.dilate(closed_image, kernel_dilate)
            contours, herichy = cv.findContours(dst1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_cordinates = (0, 0, 0, 0)
            if len(contours) != 0:
                for _, contour in enumerate(contours):
                    x, y, w, h = cv.boundingRect(contour)
                    if w * h > max_cordinates[2] * max_cordinates[3]:
                        max_cordinates = x, y, w, h
                x_start, y_start, w, h = max_cordinates
                x_hat, y_hat, w_hat, h_hat = utils.get_expand_coordinates(1.2, max_cordinates)
                image_croped = image[y_hat: y_hat + h_hat, x_hat: x_hat+w_hat, :]
            else:
                image_croped = utils.centercrop_image(image, 200, 200)

            cv.imwrite((os.path.join(self.configer['tempImagePath'], image_name)), image_croped)

    def pad_images(self, target_size):
        print('padding images begins')
        dataset = utils.get_image_set(self.training_image_path)
        for image in tqdm(dataset):
            image_name = image.split(os.sep)[-1]
            image_cv = cv.imread(image)
            if image_cv.shape[0] < 500 and image_cv.shape[1] < 500:
                horizontal, vertical = utils.get_expand_border(image_cv.shape[0], image_cv.shape[1], 500)
                image_padded = cv.copyMakeBorder(image_cv, horizontal, horizontal, vertical, vertical, cv.BORDER_CONSTANT, value=0)
                cv.imwrite((os.path.join(self.configer['trainingImagePath'], image_name)), image_padded)
            else:
                cv.imwrite((os.path.join(self.configer['trainingImagePath'], image_name)), image_cv)
    #show pictures with a frame to mark the mass of the lesion
    def save_framed_images(self):
        dataset = utils.get_image_set(self.temp_image_path)
        print('centercrop begins')
        for image in tqdm(dataset):
            image_name = image.split(os.sep)[-1]
            image = cv.imread(image)
            grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, binary = cv.threshold(grey, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
            # kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            # 先开操作，取除噪声
            opened_image = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel_open, iterations=5)
            # 填充内部空隙
            closed_image = cv.morphologyEx(opened_image, cv.MORPH_CLOSE, kernel_close, iterations=5)
            dst1 = cv.dilate(closed_image, kernel_dilate)
            contours, herichy = cv.findContours(dst1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_cordinates = (0, 0, 0, 0)
            for i, contour in enumerate(contours):
                x, y, w, h = cv.boundingRect(contour)
                if w * h > max_cordinates[2] * max_cordinates[3]:
                    max_cordinates = x, y, w, h
            x_hat, y_hat, w_hat, h_hat = utils.get_expand_coordinates(1.2, max_cordinates)
            cv.rectangle(image, (x_hat,y_hat,x_hat+w_hat,y_hat+h_hat), (0, 255, 0), 2)
            cv.imwrite((os.path.join(self.configer['tempImagePath'], image_name)), image)

if __name__ == '__main__':
    d = DataPreProcesser()
    d(50)



