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
    c = Configer().get_configer()
