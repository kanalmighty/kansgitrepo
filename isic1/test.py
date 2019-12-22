import torchvision
import torch
from visualizer.visualizer import Visualizer
from sklearn.preprocessing import label_binarize
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
visualizer = Visualizer()
isic = ISICDataset(image_path, test_csv[0], transforms)
isic.__assert_equality__()
testdata_loader = DataLoader(isic, batch_size=args.batchsize)


model.eval()#模型为测试，不使用dropput等
y_list = []
y_hat_list = []
for idx, (x, y) in enumerate(testdata_loader):

    x = x.to(device)
    y_scalar = torch.argmax(y, dim=1)
    y_hat = model.network(x)
    y_hat_scalar = torch.argmax(y_hat, dim=1)
    # if y_scalar.item() == y_hat_scalar.item():
    #     if not 'tp' + '_' + str(y_scalar.item()) in metrics.keys():
    #         metrics['tp' + '_' + str(y_scalar.item())] = 0
    #     metrics['tp' + '_' + str(y_scalar.item())] += 1
    # else:
    #     if not 'fn' + '_' + str(y_scalar.item()) in metrics.keys():
    #         metrics['fn' + '_' + str(y_scalar.item())] = 0
    #     metrics['fn' + '_' + str(y_scalar.item())] += 1
    y_list.append(y_scalar)
    y_hat_list.append(y_hat_scalar)
print(y_list)
print(y_hat_list)
class_number = y.size(1)
metrics_dict = utils.calculate_test_metrics(y_list, y_hat_list, class_number)
visualizer.get_data_report(metrics_dict)
logger.append_test_data(args.date, args.time, metrics_dict)
