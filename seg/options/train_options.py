import argparse
from options.base_options import BaseOptions
import os
import torch


class TrainingOptions(BaseOptions):

    def initialize(self):
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--mode', type=str,  default='train', help='model mode')
        self.argument_parser.add_argument('--cof', type=str,  help='coieffient')
        self.argument_parser.add_argument('--resize', action='append', help='activate data auto augment,true or false')
        self.argument_parser.add_argument('--originalSize', action='append', help='activate data auto augment,true or false')
        self.argument_parser.add_argument('--layerNumber', type=int, help='activate data auto augment,true or false')
        self.argument_parser.add_argument('--learningRate', type=float, help='learningRate')
        self.argument_parser.add_argument('--batchSize', type=int,  help='center crop size')
        self.initialized = False
