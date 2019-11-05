import urllib.request
import datetime
import time
from sys import exit
from pathlib import Path
from options.configer import Configer
import os
import json
import sys
class DataRecorder:
    def __init__(self):
        configer = Configer()
        ##get the file name that currently being executed
        # execute_file = sys.argv[0].split('/')[-1].split('.')[0]
        self.data_dict = {}
        self.config_dict = configer.get_configer()
        self.date_string = time.strftime("%Y%m%d", time.localtime())
        self.start_time = time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())
        self.start_time_string = time.strftime("%H%M%S", time.localtime())
        # self.root_path = Path('/content/drive/My Drive/daily_report' + date_string)
        self.root_path = Path(os.path.join(self.config_dict['logpath'], self.date_string))
        self.log_path = os.path.join(self.root_path, self.start_time_string + '.log')
        if not self.root_path.exists():
            try:
                os.mkdir(self.root_path)
            except IOError:
                print("please lunch google drive first")
                exit(0)
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
        time_dict = {}
        time_dict["start_time"] = self.start_time
        time_dict["end_time"] = time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())
        self.data_dict["time_interval"] = time_dict
        json_dict = json.dumps(self.data_dict)
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
        training_log = os.path.join(self.config_dict['logpath'], date_string, time_string + '.log')
        #get training data dictionary from training log
        try:
            file = open(training_log)
        except IOError:
            print("%s doesn't exist" % training_log)
            exit(0)
        data_json = file.read()
        self.data_dict = json.loads(data_json)
        # append test data
        self.data_dict["test_data"] = test_data_dict
        json_dict = json.dumps(self.data_dict)
        with open(training_log, 'w') as log:
            log.write(json_dict)
        log.close()


if __name__ == '__main__':
    dr = DataRecorder()
