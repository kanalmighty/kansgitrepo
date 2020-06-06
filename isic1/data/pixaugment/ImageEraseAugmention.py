import random
import cv2
from myimplemention.tools.utils import get_image_shape
import archieved.dcgan.settings as settings
import numpy as np
from archieved.myutils.ImageReader import ImageReader

class ImagePixlAugmention:

    # 传入参数为文件读取器
    def __init__(self, filereader):
        self.reader = filereader
        self.image_raw = []
    # 把读取图片对象进lit
    def get_images(self):
        for image in self.reader:
            self.image_raw.append(image)

    # 给图像添加高斯噪声
    def get_image_part_erased(self):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            h = get_image_shape(origin_image_matrix)[0]#767
            w = get_image_shape(origin_image_matrix)[1]#1022
            w_e = 0
            h_e = 0
            start_y_e = 0
            start_x_e = 0
            while True:
                s_e = random.uniform(0, 1) * h * w#擦除区域大小在此控制
                r_e = random.uniform(0, 1)
                h_e = int(np.sqrt(r_e * s_e))
                w_e = int(np.sqrt(s_e / r_e))
                start_x_e = random.randint(0, w - 1)#横坐标起点
                start_y_e = random.randint(0, h - 1)#纵坐标起点
                if (start_x_e + w_e > w) or (start_y_e + h_e > h):
                    continue
                else:
                    break
            for x_e in range(start_x_e, start_x_e + w_e - 1):
                for y_e in range(start_y_e, start_y_e + h_e - 1):
                    print('start_x_e = {},x_e = {}, start_y_e = {}, y_e = {},w_e = {}, h_e = {}'.format(start_x_e, x_e, start_y_e,y_e, w_e, h_e))
                    origin_image_matrix[y_e, x_e , :] = random.randint(0, 255)
            images_processd.append(origin_image_matrix)
        return  images_processd



if __name__ == '__main__':
    reader = ImageReader(settings.TEST_RAW_IMAGE_PATH)
    reader.get_files()
    imageaugmention = ImagePixlAugmention(reader)
    imageaugmention.get_images()
    images_rotated = imageaugmention.get_image_part_erased()
    for idx, image_rotated in enumerate(images_rotated):
        cv2.imwrite(settings.TEST_GEOMETRY_IMAGE_PATH + str(idx) + '.JPEG', image_rotated)
    exit(0)
    # plt.subplot(1, 2, 1)
    # plt.imshow(geoaugmentor.img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(geoaugmentor.get_image_rotate())
    # plt.show()
    #
