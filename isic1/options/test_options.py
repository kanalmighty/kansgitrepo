import argparse
from options.base_options import BaseOptions
import os
import torch

class TestOptions(BaseOptions):

    def initialize(self):    
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--model_path', type=str, help='where you can load the model pretrained')
        self.initialized = False
