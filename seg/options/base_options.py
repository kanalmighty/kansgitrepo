import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.argument_parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specificss
        # self.argument_parser.add_argument('--network', type=str, help='choices including vgg16,vgg19,alexnet,inception,resnet18,densenet161', choices=['mobileNetV1'])
        # self.argument_parser.add_argument('--testAccThreshold', type=float, default=0.8, help='test accuracy threshold')



        self.initialized = True

    def get_args(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.argument_parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt
