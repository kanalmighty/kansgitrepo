import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.argument_parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.argument_parser.add_argument('--numclass', type=int, help='the number of classes of the input')

        self.argument_parser.add_argument('--device', type=str, help='device that you want you model to be trained on')
        self.argument_parser.add_argument('--resize', action='append', type=int, help='the size(w,h) of images if you want a resize')
        self.argument_parser.add_argument('--network', type=str, help='choices including vgg16,vgg19,alexnet,inception,resnet18,densenet161', choices=['vgg16', 'vgg19', 'alexnet', 'inception', 'resnet18', 'googlenet'])
        self.argument_parser.add_argument('--epoch', type=int, default=10, help='number of epoch you want to iterate')
        self.argument_parser.add_argument('--batchsize', type=int, help='batch size')


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
