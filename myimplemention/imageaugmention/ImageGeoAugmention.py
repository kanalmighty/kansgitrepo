import random
import cv2
import dcgan.settings as settings
from myutils.Exception import Exception
from myutils.ImageReader import ImageReader


# 这个类用于
# 进行图片数据的几何增强
class ImageGeoAugmention:
    # 传入参数为文件读取器
    def __init__(self, filereader):

        self.reader = filereader
        self.image_raw = []

    def get_images(self):
        for image in self.reader:
            self.image_raw.append(image)

    # 垂直反转
    def get_image_vertical_fliped(self):
        images_processed = []
        for origin_image_matrix in self.image_raw:
            # origin_image_matrix = cv2.imread(origin_image)
            image_fliped = cv2.flip(origin_image_matrix, 0)
            images_processed.append(image_fliped)
        return images_processed

    # 水平翻转
    def get_image_horizontal_fliped(self):
        images_processed = []
        for origin_image_matrix in self.image_raw:
            image_fliped = cv2.flip(origin_image_matrix, 1)
            images_processed.append(image_fliped)
        return images_processed

    # 随机裁剪
    def get_image_random_croped(self, size=()):
        images_processed = []
        height_expected = size[0]
        width_expectd = size[1]
        for origin_image_matrix in self.image_raw:
            origin_height = origin_image_matrix.shape[0]
            origin_width = origin_image_matrix.shape[1]
            origin_channel = origin_image_matrix.shape[2]
            if size[0] > origin_height / 2 or size[1] > origin_width / 2:
                raise Exception('SizeTooLong', 'size expected is larger than the origin one')
            origin_height_half = int(origin_height / 2)
            origin_width_half = int(origin_width / 2)
            height_crop_begin = random.randint(1, origin_height_half)
            width_crop_begin = random.randint(1, origin_width_half)
            image_processed = origin_image_matrix[height_crop_begin:height_crop_begin + height_expected,
                              width_crop_begin:width_crop_begin + width_expectd, 0:origin_channel]
            images_processed.append(image_processed)
        return images_processed

    # 图像按角度旋转
    def get_image_rotated(self, angel, size=()):
        images_processed = []
        size_output = size
        for origin_image_matrix in self.image_raw:
            orgin_height = origin_image_matrix.shape[0]
            orgin_width = origin_image_matrix.shape[1]
            center = (orgin_height / 2, orgin_width / 2)
            if len(size_output) == 0:
                size_output = (orgin_height, orgin_width)
            rotate_matrix = cv2.getRotationMatrix2D(center, angel, 1)
            image_rotated = cv2.warpAffine(origin_image_matrix, rotate_matrix, size_output)
            images_processed.append(image_rotated)
        return images_processed

if __name__ == '__main__':
    reader = ImageReader(settings.TEST_RAW_IMAGE_PATH)
    reader.get_files()
    imageaugmention = ImageGeoAugmention(reader)
    imageaugmention.get_images()
    images_rotated = imageaugmention.get_image_rotated(45)
    for idx, image_rotated in enumerate(images_rotated):
        cv2.imwrite(settings.TEST_GEOMETRY_IMAGE_PATH + str(idx) + '.JPEG', image_rotated)
    # plt.subplot(1, 2, 1)
    # plt.imshow(geoaugmentor.img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(geoaugmentor.get_image_rotate())
    # plt.show()
    #
