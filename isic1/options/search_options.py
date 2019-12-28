import argparse
from options.base_options import BaseOptions
import os
import torch


class SearchOptions(BaseOptions):

    def initialize(self):
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--lossDescendThreshold', type=float, help='tell the search when to stop')
        self.initialized = False
