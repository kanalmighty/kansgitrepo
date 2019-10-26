import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

#check if it's a file with a valid suffix
def is_image_file(filename):
    return any(filename.endwith(extension) for extension in IMG_EXTENSIONS)


def get_image_set(dir):
    images = []
    for root,_,filenames in os.walk(dir):
        for filename in filenames:
            image_file = os.path.join(root, filename)
            images.append(image_file)
    return images

