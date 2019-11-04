import torchvision
import torch
import torch.nn as nn
from options.configer import Configer
from pathlib import Path
import os
import utils
from options.base_options import BaseOptions


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.args = opt
        self.network = self.get_network()
        #only training mode needs loss function
        if self.args.lossfunction:
            self.loss_function = self.get_loss_function()
        # only training mode optimizer
        if self.args.optimizer:
            self.optimizer = self.get_optimizer()
        configer = Configer()
        self.configer = configer.get_configer()


    def get_network(self):
        if self.args.network not in ['vgg16', 'vgg19', 'alexnet', 'inception_v3', 'resnet18']:
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
            #if you want to customize the number of classes of the output
            if self.args.numclass:
                fc_features = nk.fc.in_features
                nk.fc = nn.Linear(fc_features, self.args.numclass)
                if torch.cuda.is_available():
                    nk = nk.cuda()
                    return nk
        if torch.cuda.is_available():
            nk = nk.cuda()
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
            opm = torch.optim.Adam(self.network.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8)
        if self.args.optimizer == 'sgd':
            opm = torch.optim.SGD(self.network.parameters(), lr=0.003)
        return opm

    def save_model(self, date, time):
        checkpoint_path = os.path.join(self.configer['checkPointPath'], date)
        if not Path(checkpoint_path).exists():
            os.mkdir(checkpoint_path)
        pkl_name = os.path.join(checkpoint_path, time + '.pkl')
        torch.save(self.network.state_dict(), pkl_name)
        return pkl_name

#time and date is the day and time when training starts and also the name of the saved model
    def load_model(self, date, time):
        checkpoint_path = os.path.join(self.configer['checkPointPath'], date)
        pkl_name = os.path.join(checkpoint_path, time + '.pkl')
        try:
            saved_model_parameter = torch.load(pkl_name)
            print(saved_model_parameter)
            self.network.load_state_dict(saved_model_parameter)
        except IOError:
            print('there is not such model %s' % pkl_name)


if __name__ == '__main__':
    model = torchvision.models.resnet18()
    file = 'D:\\pycharmspace\\kansgitrepo\\isic1\\checkpoints\\20191102\\205441.pkl'
    a = torch.load(file)
    model.load_state_dict(a)
    print(next(model.parameters()))






