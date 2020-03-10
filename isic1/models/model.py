import torchvision
import torch
import torch.nn as nn
from options.configer import Configer
from pathlib import Path
import os

from models.networks.mobilenets import *
import utils


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.args = opt
        self.network = self.get_network()
        #only training mode needs loss function
        if self.args.mode == 'train':
            self.loss_function = self.get_loss_function()
            if self.args.learningRate:
                self.learning_rate = self.args.learningRate
            else:
                self.learning_rate = 0.0003

        # only training mode optimizer
        if self.args.mode == 'train':
            self.optimizer = self.get_optimizer()
        configer = Configer()
        self.configer = configer.get_configer()
        self.scheduler = self.get_scheduler()



    def get_network(self):
        if self.args.network not in ['mobileNetV1']:
            raise LookupError("no such network")
        network = None
        if self.args.network == 'mobileNetV1':
            network = MobileNet_v1(self.args.numclass)
        network = network.cuda() if torch.cuda.is_available() else network.cpu()
        return network


    def get_loss_function(self):
        if self.args.lossfunction not in ['cross', 'focalloss']:
            raise LookupError("no such loss function")
        if self.args.lossfunction == 'cross':
            lf = torch.nn.CrossEntropyLoss()
        if self.args.lossfunction == 'focalloss':
            lf = FocalLoss(self.args.numclass, alpha=0.25, gamma=2, size_average=True)
        return lf

    def get_optimizer(self):
        if self.args.optimizer not in ['adam', 'sgd']:
            raise LookupError("no such optimizer")
        if self.args.optimizer == 'adam':
            opm = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        if self.args.optimizer == 'sgd':
            opm = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
        return opm

    def get_scheduler(self):
        scheduler = torch.optim.lr_scheduler.MultiStepLR
        return scheduler

    def save_model(self, date, time):
        checkpoint_path = os.path.join(self.configer['checkPointPath'], date)
        if not Path(checkpoint_path).exists():
            os.mkdir(checkpoint_path)
        pth_name = os.path.join(checkpoint_path, time + '.pth')
        torch.save(self.network.state_dict(), pth_name)
        return pth_name

#data and time represents the report of a trained model as well as the path where it saved
    def load_model(self, date_string, time_string):
        model_path = os.path.join(self.configer['checkPointPath'], date_string, time_string + '.pth')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        try:
            saved_model_parameter = torch.load(model_path, map_location=device)
            self.network.load_state_dict(saved_model_parameter)
        except IOError:
            print('there is not such model %s' % model_path)


if __name__ == '__main__':
    model = torchvision.models.resnet18()
    file = 'D:\\pycharmspace\\kansgitrepo\\isic1\\checkpoints\\20191102\\205441.pkl'
    a = torch.load(file)
    model.load_state_dict(a)
    print(next(model.parameters()))






