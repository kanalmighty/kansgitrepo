import argparse
from options.base_options import BaseOptions
import os
import torch

class TestOptions(BaseOptions):

    def initialize(self):    
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--date', type=str, help='the date of the trained model')
        self.argument_parser.add_argument('--time', type=str, help='the time of the trained model')

        self.initialized = False
