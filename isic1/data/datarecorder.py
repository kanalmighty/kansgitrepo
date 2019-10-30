import urllib.request
import datetime
import time
from sys import exit
from pathlib import Path
import os

class DataRecorder:
    def __init__(self):
        pwd = os.getcwd()
        date_string = time.strftime("%Y%m%d", time.localtime())
        time_string = time.strftime("%H%M%S", time.localtime())
        # self.root_path = Path('/content/drive/My Drive/daily_report' + date_string)
        self.root_path = Path('D:\\资料\\研究课题\\' + date_string)
        self.log_path = os.path.join(self.root_path, time_string + '.log')
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
            log.writelines('process started at ' + self.record_start_time+'\n')
            log.writelines('--------------------------------------------------------------------------'+'\n')
        log.close()


    #参数是字典
    def log_data(self, data_dict):
        with open(self.log_path, 'a') as log:
            log.writelines(str(data_dict)+'\n')
        log.close()

    def finish_record(self):
        with open(self.log_path, 'a') as log:
            log.writelines('-----------------------------------------------------------------------------'+'\n')
            log.writelines('process ended at ' + time.strftime("%Y%m%d %H:%M:%S  ", time.localtime())+'\n')
            log.writelines('-----------------------------------------------------------------------------'+'\n')
        log.close()
#
# if __name__ == '__main__':
#     dr = DataRecorder()
#     dr.start_record()
#     dict = {'aa':1, 'bb':3}
#     dict1 = {'aa': 2, 'bb': 3}
#     dr.log_data(dict)
#     dr.log_data(dict1)
#     time.sleep(5)
#     dr.finish_record()