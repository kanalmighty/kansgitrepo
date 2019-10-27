import torchvision
import torch
import torch.nn as nn
import utils
from options.base_options import BaseOptions


class Model:
    def __init__(self, opt):
        self.args = opt
        self.network = self.get_network()
        self.loss_function = self.get_loss_function()
        self.optimizer = self.get_optimizer()

    def get_network(self):
        if self.args.network not in ['vgg16','vgg19','alexnet','inception_v3','resnet18']:
            raise LookupError("no such network")
        if self.args.network == 'vgg16':
            nk = torchvision.models.vgg16(pretrained=True)
        if self.args.network == 'vgg19':
            nk = torchvision.models.vgg19(pretrained=True)
        if self.args.network == 'alexnet':
            nk = torchvision.models.alexnet(pretrained=True)
        if self.args.network == 'inception':
            nk = torchvision.models.inception_v3(pretrained=True)
        if self.args.network == 'resnet18':
            nk = torchvision.models.resnet18(pretrained=True)
        if torch.cuda.is_available():
            nk = self.network.cuda()
        return nk

    def get_loss_function(self):
        if self.args.lossfunction not in ['cross','softmax']:
            raise LookupError("no such loss function")
        if self.args.lossfunction == 'cross':
            lf = torch.nn.CrossEntropyLoss()
        if self.args.lossfunction == 'softmax':
            lf = torch.nn.Softmax()
        return lf

    def get_optimizer(self):
        if self.args.optimizer not in ['adam', 'sgd']:
            raise LookupError("no such optimizer")
        if self.args.optimizer == 'adam':
            opm = torch.optim.Adam(self.network.parameters(), lr=0.003, betas=(0.9,0.999),eps=1e-8)
        if self.args.optimizer == 'sgd':
            opm = torch.optim.SGD(self.network.parameters(), lr=0.003)
        return opm





