import os
import cv2
import dcgan.settings as setting


class ImageReader:
    def __init__(self, file_path):
        self.filenames = os.listdir(file_path)#filename 不完整
        self.files = []
        self.index = 0

    def get_files(self):
        if len(self.filenames) == 0:
            raise IOError
        for filename in self.filenames:
            file = cv2.imread(os.path.join(setting.TEST_RAW_IMAGE_PATH,filename))
            self.files.append(file)

    def __getitem__(self, item):
        return self.files[item]

    def __next__(self):
        if self.index > len(self.files):
            raise StopIteration
        return self.files[self.index]
        index += 1


if __name__ == '__main__':
    reader = ImageReader(setting.TEST_RAW_IMAGE_PATH)
    reader.get_files()
    it = iter(reader)
    while True:
        print(next(it))




