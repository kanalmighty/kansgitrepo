import argparse
from options.base_options import BaseOptions
import os
import torch

class TrainingOptions(BaseOptions):

    def initialize(self):
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--resize', type=int, help='the size(w,h) of images if you want a resize')
        self.argument_parser.add_argument('--mode', type=str,  default='train', help='model mode')
        self.argument_parser.add_argument('--autoaugment', type=bool, help='activate data auto augment,true of false')
        self.argument_parser.add_argument('--optimizer', type=str, help='choices including adam,sgd,momentum', choices=['adam', 'sgd'])
        self.argument_parser.add_argument('--lossfunction', type=str, help='choices including cross,softmax', choices=['cross', 'focalloss'])
        self.argument_parser.add_argument('--centerCropSize', type=int, action='append', help='center crop size')
        self.argument_parser.add_argument('--normalize', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.initialized = False
