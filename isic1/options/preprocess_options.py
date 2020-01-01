import torch
from options.base_options import BaseOptions


class PreprocessOptions(BaseOptions):

    def initialize(self):
        # experiment specifics
        BaseOptions.initialize(self)
        self.argument_parser.add_argument('--dataBalance', type=int, help='expand all data from different classed to the same scale with augument')
        self.argument_parser.add_argument('--borderCropRate', type=float, default=0.5, help='trim images border at the given rate')
        self.argument_parser.add_argument('--padBorderSize', type=int, default=500, help='rescale images to the given resolution')
        self.argument_parser.add_argument('--massCrop', type=bool, help='use opencv to crop up the lesion without the background')
        self.argument_parser.add_argument('--off', type=bool, help='do nothing but move images and label from row path to processed path')
        self.argument_parser.add_argument('--testSamples', type=int, help='test sample set aside from the training dataset')
        self.initialized = False
