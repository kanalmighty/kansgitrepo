import torch
import pdb
import matplotlib
import torch.nn as nn
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
import argparse
from options.search_options import SearchOptions


class Searcher:
    def __init__(self, setting):
        self.logger = DataRecorder()  # 初始化记录器
        self.visualizer = Visualizer()  # 初始化视觉展示器
        self.setting = setting
        # continue training if date and time are specified
        self.configer = Configer().get_configer()  # 获取环境配置
        # self.learning_rate_space = [0.1, 0.01, 0.0001, 0.00001, 0.000001]
        # self.resolution_space = [224, 280, 320, 400, 500, 600]
        # self.nn_space = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        #                 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        #                 'efficientnet-b6', 'efficientnet-b7']
        self.learning_rate_space = [0.1, 0.01]
        self.resolution_space = [224, 280]
        self.nn_space = ['resnet18']
        self.optimizer_space = ['adam']
        self.loss_space = ['cross']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.one_search_data = {}
        self.super_param_combination_dict = {}
        for network in self.nn_space:
            for lr in self.learning_rate_space:
                for pix in self.resolution_space:
                    for optimizer in self.optimizer_space:
                        for loss_f in self.loss_space:
                            spd = network + '_' + str(lr) + '_' + str(pix) + '_' + optimizer + '_' + loss_f
                            self.super_param_combination_dict[spd] = (network, lr, pix, optimizer, loss_f)


    def train(self, parameters):
        self.one_search_data.clear()
        self.one_search_data['parameters'] = vars(parameters)
        image_path = self.configer['trainingImagePath']
        label_path = self.configer['trainingLabelPath']
        training_csv = utils.get_csv_by_path_name(label_path)
        transforms = utils.get_transforms(parameters)
        isic_dataset = ISICDataset(image_path, training_csv[0], transforms)
        isic_dataset.__assert_equality__()
        trainingdata_loader = DataLoader(isic_dataset, batch_size=32, shuffle=True, drop_last=True)
        self.model = Model(parameters)  # 根据参数获取模型
        optimizer = self.model.optimizer
        criteria = self.model.loss_function
        epoch_statics_list = []  # store epoch loss and training accuracy
        self.model.train()
        for EPOCH in range(self.setting.epoch):
            if EPOCH > 3:
                loss_descend_rate = epoch_statics_list[-1]['AVG LOSS']/epoch_statics_list[0]['AVG LOSS'] >= self.setting.lossDescendThreshold
                print('current loss descend rate is %d ,less than threshold %d, abandon this SPD' % (loss_descend_rate, self.setting.lossDescendThreshold))
                break
            epoch_statics_dict = {}  # record epochly training statics
            loss_all_samples_per_epoch = 0  # 记录每个epoch,所有batch的loss总和
            train_accuracy = 0  # trainnig accuaracy per epoch
            for idx, (x, y) in tqdm(enumerate(trainingdata_loader)):
                batch_statics_dict = {}
                x = x.to(self.device)
                y = torch.argmax(y, dim=1)
                y_hat = self.model.network(x.float())
                train_accuracy += (y.to(self.device) == torch.argmax(y_hat, dim=1)).sum().item()

                loss = criteria(y_hat, y.long().to(self.device))
                loss_all_samples_per_epoch += loss.item()  # loss.item()获取的是每个batchsize的平均loss
                # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
                batch_statics_dict['EPOCH'] = EPOCH
                batch_statics_dict[parameters.lossfunction] = loss.item()
                # loss_dict_print，每个epoch,都是损失函数名:值（值是list）
                # visualizer.get_data_report(batch_statics_dict)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_avg_per_epoch = loss_all_samples_per_epoch / (idx + 1)  # 获取这个epoch中一个平input的均loss,idx从0开始，所以需要加1
            train_accuracy_epoch = train_accuracy / len(isic_dataset)  # training accuracy/sample numbers
            epoch_statics_dict['EPOCH'] = EPOCH
            epoch_statics_dict['AVG LOSS'] = loss_avg_per_epoch

            epoch_statics_dict['TRAINING ACCURACY'] = train_accuracy_epoch

            pkl_name = self.model.save_model(self.logger.date_string, self.logger.start_time_string)  # save the nn every epoch
            epoch_statics_dict['saved_model'] = pkl_name
            epoch_statics_list.append(epoch_statics_dict)  # record epoch loss for drawing
            print('epoch %s finished ' % EPOCH)
            self.visualizer.get_data_report(epoch_statics_dict)
        self.one_search_data['training_statics'] = epoch_statics_list
        self.logger.set_training_data(self.one_search_data)

    def test(self, args):
        image_path = self.configer['testImagePath']
        label_path = self.configer['testLabelPath']
        test_csv = utils.get_csv_by_path_name(label_path)
        transforms = utils.get_transforms(args)
        isictest = ISICDataset(image_path, test_csv[0], transforms)
        isictest.__assert_equality__()
        testdata_loader = DataLoader(isictest, batch_size=1)

        self.model.eval()  # 模型为测试，不使用dropput等
        y_list = []
        y_hat_list = []
        for idx, (x, y) in enumerate(testdata_loader):
            x = x.to(self.device)
            y_scalar = torch.argmax(y, dim=1)
            y_hat = self.model.network(x)
            y_hat_scalar = torch.argmax(y_hat, dim=1)
            # if y_scalar.item() == y_hat_scalar.item():
            #     if not 'tp' + '_' + str(y_scalar.item()) in metrics.keys():
            #         metrics['tp' + '_' + str(y_scalar.item())] = 0
            #     metrics['tp' + '_' + str(y_scalar.item())] += 1
            # else:
            #     if not 'fn' + '_' + str(y_scalar.item()) in metrics.keys():
            #         metrics['fn' + '_' + str(y_scalar.item())] = 0
            #     metrics['fn' + '_' + str(y_scalar.item())] += 1
            y_list.append(y_scalar.item())
            y_hat_list.append(y_hat_scalar.item())
        class_number = y.size(1)
        metrics_dict = utils.calculate_test_metrics(y_list, y_hat_list, class_number)
        self.visualizer.get_data_report(metrics_dict)
        self.one_search_data['test_data'] = metrics_dict

    def search(self):
        args = argparse.Namespace()
        args.centerCropSize = False
        args.normalize = True
        args.numclass = self.setting.numclass
        args.lossDescendThreshold = self.setting.lossDescendThreshold
        for k, v in self.super_param_combination_dict.items():
            search_record_string = self.logger.get_search_data()
            #pass the super parameter combination that has been searched
            if k in search_record_string:
                print('%s has been done,continue..' % k)
                continue
            one_search_dict = {}
            args.network = v[0]
            args.learningRate = v[1]
            args.resize = (v[2], v[2])
            args.optimizer = v[3]
            args.lossfunction = v[4]
            args.mode = 'train'
            self.train(args)
            self.test(args)
            one_search_dict[k] = self.one_search_data
            one_search_dict[k]['flag'] = 1
            print('write log' + k)
            self.logger.append_search_data(one_search_dict)
        print('search end..')


if __name__ == '__main__':
    options = SearchOptions()
    args = options.get_args()  # 获取参数
    s = Searcher(args)
    s.search()




