import numpy as np
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read(filenames='D:\pycharmspace\kansgitrepo\isic1\options\config.ini', encoding='UTF-8')
print(config['Windows']['logpath'])