import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.argument_parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.argument_parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
        self.argument_parser.add_argument('--device', type=str, default='cpu', help='device that you want you model to be trained on')
        self.argument_parser.add_argument('--resize', type=str, default='cpu', help='device that you want you model to be trained on')
        self.argument_parser.add_argument('--datapath', type=str, default='./', help='where the images are stored')
        self.argument_parser.add_argument('--labelpath', type=str, default='./', help='where the labels are stored')
        self.argument_parser.add_argument('--Normalize', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
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
