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
