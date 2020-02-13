import cv2 as cv
import utils
from options.configer import Configer
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


class ImageProcessorBuilder():
    #strategy is a dict
    def __init__(self, args):
        self.configer = Configer().get_configer()
        self.args = args
        strategy = {'blur': 'gs', 'morphology': 'erode', 'threshold': 'inverse'}
        self.blur = BlurAdapter(strategy['blur'])
        self.morphology = MorphologyAdapter(strategy['morphology'])
        self.threshold = ThresholdAdapter(strategy['threshold'])
        self.cam_threshold = ThresholdAdapter('normal')

    def get_input_binary(self, input):
        if not isinstance(input, torch.Tensor):
            raise TypeError('input must be a Tensor!')
        image_list = []
        binary_vector_list = []

        #循环tensor,把每个tensor转为图片,原始tensor形式为（b,c,w,h)
        for batch_num in range(0, self.args.batchsize-1):
            single_tensor = input[batch_num, :, :, :]
            original_image = utils.tensor_transform(single_tensor, 'image')

            image_cropped_ndarray = np.array(original_image)#to ndarray
            image_list.append(image_cropped_ndarray)
            grey_image = cv.cvtColor(image_cropped_ndarray, cv.COLOR_BGR2GRAY)
            # 灰度直方图均衡化
            grey_image = cv.equalizeHist(grey_image)
            image_list.append(grey_image)

            # 模糊处理
            grey_image = self.blur.get_blur_image(grey_image)
            image_list.append(grey_image)

            #二值化
            ret, binary_image = self.threshold.get_binary_image(grey_image)
            image_list.append(binary_image)

            #形态学处理
            binary_image = self.morphology.get_processed_image(binary_image)
            image_list.append(binary_image)


            w = int(binary_image.shape[0])
            h = int(binary_image.shape[1])
            #把图片拉直为一个向量
            binary_vector = np.reshape(binary_image, (w*h))
            binary_vector_list.append(binary_vector)

            return binary_vector_list

    def get_cam_binary(self, heatmap_list):

        image_list = []
        binary_vector_list = []

        for grey_image in heatmap_list:
            # 灰度直方图均衡化
            # grey_image = cv.cvtColor(grey_image, cv.COLOR_RGB2GRAY)
            image_list.append(grey_image)
            # grey_image = cv.equalizeHist(grey_image)
            # image_list.append(grey_image)
            #
            # # 模糊处理
            # grey_image = self.blur.get_blur_image(grey_image)
            # image_list.append(grey_image)

            #二值化
            ret, binary_image = self.cam_threshold.get_binary_image(grey_image)
            image_list.append(binary_image)

            #形态学处理
            binary_image = self.morphology.get_processed_image(binary_image)
            image_list.append(binary_image)

            w = int(binary_image.shape[0])
            h = int(binary_image.shape[1])
            #把图片拉直为一个向量
            binary_vector = np.reshape(binary_image, (w*h))
            binary_vector_list.append(binary_vector)
        return binary_vector_list



class BlurAdapter():
    def __init__(self, blur_name='gs'):
        self.blur_name = blur_name

    def get_blur_image(self, original_image):
        if self.blur_name == 'gs':  #高斯模糊
            return cv.GaussianBlur(original_image, (0, 0), 15)
        if self.blur_name == 'md':  #中值模糊
            return cv.medianBlur(original_image, 5)
        if self.blur_name == 'mn':  #均值
            return cv.blur(original_image, (1, 15))
        if self.blur_name == 'bl':  #边缘保留
            return cv.bilateralFilter(original_image, 0, 100, 15)


class MorphologyAdapter():
    def __init__(self, morphology_name):
        self.morphology_name = morphology_name
        self.structuring_element = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

    def get_processed_image(self, original_image):
        if self.morphology_name == 'erode':
            return cv.erode(original_image, self.structuring_element)
        if self.morphology_name == 'dilate':
            return cv.dilate(original_image, self.structuring_element)
        if self.morphology_name == 'open':
            return cv.morphologyEx(original_image, cv.MORPH_OPEN, self.structuring_element, iterations=5)
        if self.morphology_name == 'close':
            return cv.morphologyEx(original_image, cv.MORPH_CLOSE, self.structuring_element, iterations=5)


class ThresholdAdapter():
    def __init__(self, method_name):
        self.method_name = method_name

    def get_binary_image(self, original_image):
        if self.method_name == 'normal':
            return cv.threshold(original_image, 0, 1, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        if self.method_name == 'inverse':
            return cv.threshold(original_image, 0, 1, cv.THRESH_BINARY_INV | cv.THRESH_TRIANGLE)
        if self.method_name == 'trunc':
            return cv.threshold(original_image, 0, 1, cv.THRESH_TRUNC | cv.THRESH_TRIANGLE)
        if self.method_name == 'zero':
            return cv.threshold(original_image, 0, 1, cv.THRESH_TOZERO | cv.THRESH_TRIANGLE)


if __name__ == '__main__':
    stratgy = {'blur': 'gs', 'morphology': 'erode'}
    ip = ImageProcessorBuilder(stratgy, 'a')
    image_path = utils.get_image_set('C:\\Users\\23270\\Desktop\\cvtest')
    ip.get_binaray_image(image_path)
    # image = cv.imread('C:\\Users\\23270\\Desktop\\heatmap\\heatmap.jpg')
    # image = image[:, :, 0]
    # ret, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # cv.imshow('channel', binary)

