import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
import collections
import argparse
import torchvision.models as model
from torchvision import transforms
import numpy as np
import utils as utils
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualizer.visualizer import Visualizer
from torch.utils.data import DataLoader
import os
class MobileNet_v1(nn.Module):
    def __init__(self, num_class):
        super(MobileNet_v1, self).__init__()
        self.num_class = num_class
        self.net = nn.Sequential(
            collections.OrderedDict([('conv1', nn.Conv2d(3, 64, 3, 1, 1)), ('bn1', nn.BatchNorm2d(64)),
                                     ('relu1', nn.ReLU(inplace=True)),
                                     ('conv2', nn.Conv2d(64, 64, 3, 2, 1)),
                                     ('bn2', nn.BatchNorm2d(64)),
                                     ('relu2', nn.ReLU(inplace=True)),
                                     ('conv3', nn.Conv2d(64, 128, 3, 2, 1)),
                                     ('bn3', nn.BatchNorm2d(128)),
                                     ('relu3', nn.ReLU(inplace=True)),
                                     ('conv4', nn.Conv2d(128, 128, 3, 2, 1)),
                                     ('bn4', nn.BatchNorm2d(128)),
                                     ('relu4', nn.ReLU(inplace=True)),
                                     ('conv5', nn.Conv2d(128, 256, 3, 2, 1)),
                                     ('bn5', nn.BatchNorm2d(256)),
                                     ('relu5', nn.ReLU(inplace=True)),
                                     ('conv6', nn.Conv2d(256, 512, 3, 2, 1)),
                                     ('bn6', nn.BatchNorm2d(512)),
                                     ('relu6', nn.ReLU(inplace=True)),
                                     ('conv7', nn.Conv2d(512, 1024, 3, 1, 1)),
                                     ('bn7', nn.BatchNorm2d(1024)),
                                     ('relu7', nn.ReLU(inplace=True)),
                                     ('maxpool', nn.MaxPool2d(7))
                                     ]))

        self.classifier = nn.Sequential(nn.Linear(1024, self.num_class))

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 1024)
        prediction = self.classifier(output)
        return prediction