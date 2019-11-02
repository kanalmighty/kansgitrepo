import configparser
import platform
import os
import utils

class Configer:
    def __init__(self, config_file_name='config.ini'):
        self.config = configparser.ConfigParser()
        config_path = utils.get_file_path(config_file_name)
        self.config.read(filenames=config_path, encoding='UTF-8')
        self.os = platform.platform().split('-')[0]

    def show_sections(self):
        print(self.config.sections())

    def get_configer(self):
        return self.config[self.os]


if __name__ == '__main__':
    # configer = Configer('config.ini')
    # config_dict = configer.get_configer()
    root = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('config.ini')[0],'config.ini')
    print('root=', root)
    # config = configparser.ConfigParser()
    # config.read(filenames='D:\pycharmspace\kansgitrepo\isic1\options\config.ini', encoding='UTF-8')
    # print(config['Windows']['logpath'])

