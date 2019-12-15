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
testdata_loader = DataLoader(isic, batch_size=args.batchsize, shuffle=True, drop_last=True)


model.eval()#模型为测试，不使用dropput等
acurracy = 0
total = 0
true_positive = 0#检查出患病，真患病，true代表预测和实际都不对得上
true_negative = 0#检查出没病，真没病
false_positive = 0#检擦出没病，其实有病
false_negative = 0#检查有病，其实没病
for idx, (x, y) in enumerate(testdata_loader):
    x = x.to(device)
    y_scalar = torch.argmax(y, dim=1)
    y_hat = model.network(x)
    y_hat_scalar = torch.argmax(y_hat, dim=1)
    print(y_hat_scalar)
    #if groundtruth is positive
    if y_scalar == 0:
        if y_hat_scalar == 0:#prediction is positive
            true_positive += 1# then it's true positive
        else:
            false_positive += 1#it's positive,not predition is negative
    else:
        # if groundtruth is negative
        if y_hat_scalar == 0:#prediction is positive
            false_negative += 1# false_positive
        else:
            true_negative += 1

metrics_dict = utils.get_evaluation_metrics(true_positive, true_negative, false_positive, false_negative)
print(metrics_dict)
logger.append_test_data(args.date, args.time, metrics_dict)
