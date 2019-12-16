import torchvision
import torch
import pdb
import matplotlib
from options.configer import Configer
import torch.nn as nn
from data.datarecorder import DataRecorder
from data.dataprober import DataProber
import utils
from models.model import Model
from options.test_options import TestOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
options = TestOptions()
logger = DataRecorder()
configer = Configer().get_configer()
args = options.get_args()
model = Model(args)
#load model being trained previously
model.load_model(args.date, args.time)

image_path = configer['testImagePath']
label_path = configer['testLabelPath']
test_csv = utils.get_csv_by_path_name(label_path)
dataprober = DataProber(image_path, test_csv[0])
dataprober.get_size_profile()
dataprober.get_type_profile()
dataprober.get_data_difference()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transforms = utils.get_transforms(args)

isic = ISICDataset(image_path, test_csv[0], transforms)
isic.__assert_equality__()
testdata_loader = DataLoader(isic, batch_size=args.batchsize)


model.eval()#模型为测试，不使用dropput等
metrics = {}
for idx, (x, y) in enumerate(testdata_loader):

    x = x.to(device)
    y_scalar = torch.argmax(y, dim=1)
    y_hat = model.network(x)
    y_hat_scalar = torch.argmax(y_hat, dim=1)
    if y_scalar.item() == y_hat_scalar.item():
        if not 'tp' + '_' + str(y_scalar.item()) in metrics.keys():
            metrics['tp' + '_' + str(y_scalar.item())] = 0
        metrics['tp' + '_' + str(y_scalar.item())] += 1
    else:
        if not 'fn' + '_' + str(y_scalar.item()) in metrics.keys():
            metrics['fn' + '_' + str(y_scalar.item())] = 0
        metrics['fn' + '_' + str(y_scalar.item())] += 1

class_number = y.size(1)
sensitivity = 0
for k, v in metrics.items():
    #sensitivity is valid when only true positive sample of this class is not 0
    if 'tp' in k:
        class_no = k.split('_')[1]
        #get the the fn numbers of this tp sample,and caculate sensitivity
        fn_key = 'fn' + '_' + class_no
        if not fn_key in metrics.keys():
            sensitivity += 1/class_number
        else:
            sensitivity += v/((v + metrics[fn_key])*class_number)

print(sensitivity)
logger.append_test_data(args.date, args.time, sensitivity)
