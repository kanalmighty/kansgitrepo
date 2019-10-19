import random
import torch
import cv2
from myimplemention.tools.utils import get_image_shape
import dcgan.settings as settings
from myutils.ImageReader import ImageReader

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
    def get_image_gaussion_noised(self):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            image_shape = get_image_shape(origin_image_matrix)
            gaussion_noise = torch.randn(image_shape, dtype=torch.float64)
            image_processd = origin_image_matrix + gaussion_noise.numpy()*255
            images_processd.append(image_processd)
        return images_processd

    # （5）判定浮点数是否大于0
    # 给图像添加椒盐噪声
    # 椒盐噪声也叫脉冲噪声，即在一幅图像里随机将一个像素点变为椒噪声或盐噪声，其中椒噪声像素值为“0”，盐噪声像素值为“255”。
    # 生成（添加）椒盐噪声算法步骤如下：
    # （1）输入一幅图像并自定义信噪比SNR （其取值范围在[0, 1]之间）；
    # （2）计算图像像素点个数SP 进而得到椒盐噪声的像素点数目NP = SP * (1 - SNR)；
    # （3）随机获取要加噪的每个像素位置img[i, j]；
    # （4）随机生成[0, 1]
    # 之间的一个浮点数；
    # .5，并指定像素值为255或者0；
    # （6）重复3，4，5
    # 三个步骤完成所有像素的NP个像素加粗样式；
    # （7）输出加噪以后的图像。
    def get_image_pas_noised(self, snr):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            matrix_shape = get_image_shape(origin_image_matrix)
            total_pixel = matrix_shape[0]*matrix_shape[1]
            np = int(total_pixel*(1-snr))
            for i in range(0, np):
                row_position = random.randint(0, matrix_shape[0] - 1)
                column_position = random.randint(0, matrix_shape[1] - 1)
                salt_or_paper = 255 if torch.rand(1) > 1 else 0
                origin_image_matrix[row_position, column_position] = salt_or_paper
            images_processd.append(origin_image_matrix)
        return  images_processd

    # 图中矩阵分别为二维原图像素矩阵，二维的图像滤波矩阵（也叫做卷积核，下面讲到滤波器和卷积核都是同个概念），以及最后滤波后的新像素图。对于原图像的每一个像素点，计算它的领域像素和滤波器矩阵的对应元素的成绩，然后加起来，作为当前中心像素位置的值，这样就完成了滤波的过程了。
    # 可以看到，一个原图像通过一定的卷积核处理后就可以变换为另一个图像了。而对于滤波器来说，也是有一定的规则要求的。
    # ① 滤波器的大小应该是奇数，这样它才有一个中心，例如3x3，5
    # x5或者7x7。有中心了，也有了半径的称呼，例如5x5大小的核的半径就是2。
    # ② 滤波器矩阵所有的元素之和应该要等于1，这是为了保证滤波前后图像的亮度保持不变。当然了，这不是硬性要求了。
    # ③ 如果滤波器矩阵所有元素之和大于1，那么滤波后的图像就会比原图像更亮，反之，如果小于1，那么得到的图像就会变暗。如果和为0，图像不会变黑，但也会非常暗。
    # ④ 对于滤波后的结构，可能会出现负数或者大于255的数值。对这种情况，我们将他们直接截断到0和255之间即可。对于负数，也可以取绝对值。
    # 标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。概括地讲，高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大。
    def get_image_gaussion_blurred(self, k_size, sigma):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            image_processd = cv2.GaussianBlur(origin_image_matrix, k_size, sigma)
            images_processd.append(image_processd)
        return images_processd

    def get_image_meandian_blurred(self, k_size):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            image_processd = cv2.medianBlur(origin_image_matrix, 3)
            images_processd.append(image_processd)
        return images_processd

    def get_image_greyed(self):
        images_processd = []
        for origin_image_matrix in self.image_raw:
            image_processd = cv2.cvtColor(origin_image_matrix, cv2.COLOR_BGR2RGB)
            image_processd = cv2.cvtColor(origin_image_matrix, cv2.COLOR_RGB2GRAY)
            images_processd.append(image_processd)
        return images_processd
if __name__ == '__main__':
    reader = ImageReader(settings.TEST_RAW_IMAGE_PATH)
    reader.get_files()
    imageaugmention = ImagePixlAugmention(reader)
    imageaugmention.get_images()
    images_rotated = imageaugmention.get_image_greyed()
    for idx, image_rotated in enumerate(images_rotated):
        cv2.imwrite(settings.TEST_GEOMETRY_IMAGE_PATH + str(idx) + '.JPEG', image_rotated)
    exit(0)
    # plt.subplot(1, 2, 1)
    # plt.imshow(geoaugmentor.img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(geoaugmentor.get_image_rotate())
    # plt.show()
    #
