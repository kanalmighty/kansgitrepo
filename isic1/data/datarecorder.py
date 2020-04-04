import urllib.request
import datetime
import time
from sys import exit
from pathlib import Path
from options.configer import Configer
import os
import json
import utils
class DataRecorder:
    def __init__(self):
        configer = Configer()
        ##get the file name that currently being executed
        # execute_file = sys.argv[0].split('/')[-1].split('.')[0]
        self.data_dict = {}
        self.config_dict = configer.get_configer()
        self.date_string = time.strftime("%Y%m%d", time.localtime())
        self.start_time = time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())
        self.start_time_string = time.strftime("%H_%M_%S", time.localtime())
        # self.root_path = Path('/content/drive/My Drive/daily_report' + date_string)
        self.root_path = Path(os.path.join(self.config_dict['logpath'], self.date_string))
        self.log_path = os.path.join(self.root_path, self.start_time_string + '.log')
        self.record_start_time = time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())

    def start_record(self):
        with open(self.log_path, 'w') as log:
            log.writelines('--------------------------------------------------------------------------'+'\n')
            log.writelines('\n')
            log.writelines('process started at ' + self.record_start_time+'\n')
            log.writelines('\n')
        log.close()


    #参数是字典
    def set_training_data(self, data_dict):
        self.data_dict["training_data"] = data_dict

    def set_test_data(self, data_dict):
        self.data_dict["test_data"] = data_dict

    def set_arguments(self, data_dict):
        self.data_dict["arguments"] = data_dict

    def write_training_data(self):

        json_dict = json.dumps(self.data_dict, indent=1)
        with open(self.log_path, 'a') as log:
            log.write(json_dict)
        log.close()

    def record_checkpoint(self, checkpoint_path):
        with open(self.log_path, 'a') as log:
            log.writelines('\n')
            log.writelines('model saved in ' + checkpoint_path +'\n')
            log.writelines('\n')
        log.close()

    def append_test_data(self, date_string, time_string, test_data_dict):
        test_log = os.path.join(self.config_dict['logpath'], date_string, time_string + '_test.log')
        #get training data dictionary from training log
        # try:
        #     file = open(training_log)
        # except IOError:
        #     print("%s doesn't exist" % training_log)
        #     file = open(training_log, 'w')
        # data_json = file.read()
        # self.data_dict = json.loads(data_json)
        # # append test data
        # self.data_dict["test_data"] = test_data_dict
        json_dict = json.dumps(test_data_dict)
        with open(test_log, 'w') as log:
            log.write(json_dict)
        log.close()

    def append_search_data(self, data_dict):
        log_file = os.path.join(self.config_dict['searchLogPath'], 'search.log')
        #dumps将字典转换为字符串,在写入文件
        json_str = json.dumps(data_dict)
        with open(log_file, 'a+') as log:
            log.writelines(json_str + '\n')
        log.close()

    def get_search_data(self):
        search_log_dict = {}
        if not Path(self.config_dict['searchLogPath']).exists():
            os.mkdir(self.config_dict['searchLogPath'])
        log_file = os.path.join(self.config_dict['searchLogPath'], 'search.log')
        if not Path(log_file).exists():
            return search_log_dict
        with open(log_file, 'r') as log:
            log_record_list = log.readlines()
        log.close()
        for record in log_record_list:
            record_dict = json.loads(record)
            for k, v in record_dict.items():
                search_log_dict[k] = v
        return search_log_dict


if __name__ == '__main__':
    dr = DataRecorder()
