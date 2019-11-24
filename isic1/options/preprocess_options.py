import torch
from options.base_options import BaseOptions
class PreprocessOptions(BaseOptions):

    def initialize(self):
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--databalance', type=int, help='expand all data from different classed to the same scale with augument')

        self.initialized = False
