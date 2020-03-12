import torchvision
import torch
import pdb
import matplotlib
import torch.nn as nn
from torch.autograd import Variable


from tqdm import tqdm
from data.datarecorder import DataRecorder
from data.dataprober import DataProber
import utils
from models.model import Model
from options.configer import Configer
from options.train_options import TrainingOptions
from data.datasets import ISICDataset
from torch.utils.data import DataLoader
from data.autoaugment import *
from visualizer.visualizer import Visualizer
# model = torchvision.models.resnet18(pretrained=True).cuda()
options = TrainingOptions()
logger = DataRecorder()#初始化记录器
visualizer = Visualizer()#初始化视觉展示器
args = options.get_args()#获取参数
auto_augment = AutoAugment()#初始化数据增强器
args.augment_policy = auto_augment.policy_detail#记录数据增强策略
model = Model(args)#根据参数获取模型
#continue training if date and time are specified
if args.date and args.time:
    model.load_model(args.date, args.time)
configer = Configer().get_configer()#获取环境配置

# dataprober.get_data_difference()
transforms = utils.get_transforms(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_path = configer['trainingImagePath']
label_path = configer['trainingLabelPath']
training_csv = utils.get_csv_by_path_name(label_path)
dataprober = DataProber(image_path, training_csv[0])#初始化数据探查器
isic = ISICDataset(image_path, training_csv[0], transforms)
isic.__assert_equality__()
trainingdata_loader = DataLoader(isic, batch_size=args.batchsize, shuffle=True, drop_last=True)

optimizer = model.optimizer
criteria = model.loss_function
logger.set_arguments(vars(args))
#define a loss dict to plot different losses
train_loss_dict = {}
epoch_statics_list = []#store epoch loss and training accuracy
train_statics_dict = {}#record overall training statics
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []
#image process stratgy
# stratgy = {'blur': 'gs', 'morphology': 'erode', 'threshold': 'normal'}
# image_processor = ImageProcessorBuilder(stratgy, args)
for EPOCH in range(args.epoch):
    #training start
    model.train()
    print('current lr is ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
    epoch_statics_dict = {}#record epochly training statics
    loss_per_epoch = 0#记录每个epoch,所有batch的loss总和
    accuracy_count_epoch = 0#trainnig accuaracy per epoch
    test_accuracy_count_epoch = 0
    for idx, (x, y) in tqdm(enumerate(trainingdata_loader)):
        batch_statics_dict = {}
        x = x.to(device)
        y_hat = model.network(x.float())
        y_arg = torch.argmax(y, dim=1).cpu()
        pred_arg = torch.argmax(y_hat, dim=1).cpu()
        accuracy_count_epoch += utils.accuracy_count(pred_arg, y_arg)

        loss = criteria(y_hat.to(device), y_arg.long().to(device))
        #计算attention loss
        # att_loss = attention_loss.get_attention_loss(model.network, x)
        # att_loss = torch.from_numpy(np.array(att_loss))
        # att_loss = att_loss.type_as(loss)
        loss_per_epoch += loss.item()#loss.item()获取的是每个batchsize的平均loss
        # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
        batch_statics_dict['EPOCH'] = EPOCH
        batch_statics_dict[args.lossfunction] = loss.item()
        # loss_dict_print，每个epoch,都是损失函数名:值（值是list）
        # visualizer.get_data_report(batch_statics_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg_per_epoch = loss_per_epoch/(len(trainingdata_loader))#计算本epoch每个样本的平均loss
    train_accuracy_epoch = accuracy_count_epoch / (args.batchsize*len(trainingdata_loader))#计算本epoch每个样本的平均accuracy
    train_loss_list.append(loss_avg_per_epoch)
    train_accuracy_list.append(train_accuracy_epoch)

    #参数信息计入日志
    epoch_statics_dict['AVG LOSS'] = loss_avg_per_epoch
    epoch_statics_dict['TRAINING ACCURACY'] = train_accuracy_epoch

    #保存模型
    pkl_name = model.save_model(logger.date_string, logger.start_time_string)#save the nn every epoch
    #test start
    test_image_path = configer['testImagePath']
    test_label_path = configer['testLabelPath']
    test_csv = utils.get_csv_by_path_name(test_label_path)
    test_dataset = ISICDataset(test_image_path, test_csv[0], transforms)
    testdata_loader = DataLoader(test_dataset, batch_size=1)
    model.eval()  # 模型为测试，不使用dropput等
    with torch.no_grad():
        for idx, (x, y) in enumerate(testdata_loader):
            x = x.to(device)
            y_test_arg = torch.argmax(y, dim=1)
            y_test_hat = model.network(x)
            y_hat_test_arg = torch.argmax(y_test_hat, dim=1)
            test_accuracy_count_epoch += utils.accuracy_count(y_hat_test_arg.cpu(), y_test_arg.cpu())#到这里为止
    test_accuracy_list.append(test_accuracy_count_epoch / len(testdata_loader))

    epoch_statics_dict['saved_model'] = pkl_name
    epoch_statics_list.append(epoch_statics_dict)  # record epoch loss for drawing
visualizer.draw_curve(train_accuracy_list, test_accuracy_list, train_loss_list)
train_statics_dict['training_statics'] = epoch_statics_list

logger.set_training_data(train_statics_dict)
logger.write_training_data()