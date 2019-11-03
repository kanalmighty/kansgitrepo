import urllib.request
import datetime
import time
from sys import exit
from pathlib import Path
from options.configer import Configer
import os
import sys
class DataRecorder:
    def __init__(self):
        pwd = os.getcwd()
        configer = Configer()
        #get the file name being executed
        execute_file = sys.argv[0].split('/')[-1].split('.')[0]
        config_dict = configer.get_configer()
        self.date_string = time.strftime("%Y%m%d", time.localtime())
        self.time_string = time.strftime("%H%M%S", time.localtime())
        # self.root_path = Path('/content/drive/My Drive/daily_report' + date_string)
        self.root_path = Path(config_dict['logpath'] + self.date_string)
        self.log_path = os.path.join(self.root_path, execute_file + self.time_string + '.log')
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
    def log_data(self, data_dict):
        with open(self.log_path, 'a') as log:
            log.writelines('\n')
            log.writelines('*********** data ***********'+'\n')
            log.writelines('\n')
            for k, v in sorted(data_dict.items()):
                log.writelines('%s: %s' % (str(k), str(v))+'\n')
            log.writelines('\n')
            log.writelines('*********** End ***********'+'\n')
            log.writelines('\n')
        log.close()

    def finish_record(self):
        with open(self.log_path, 'a') as log:
            log.writelines('\n')
            log.writelines('process ended at ' + time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())+'\n')
            log.writelines('\n')
            log.writelines('-----------------------------------------------------------------------------'+'\n')
        log.close()

    def record_checkpoint(self, checkpoint_path):
        with open(self.log_path, 'a') as log:
            log.writelines('\n')
            log.writelines('model saved in ' + checkpoint_path +'\n')
            log.writelines('\n')
        log.close()
if __name__ == '__main__':
    dr = DataRecorder()
