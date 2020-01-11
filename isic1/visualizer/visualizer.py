import os
import random

import matplotlib.pyplot as plt

import utils
from data.datarecorder import DataRecorder
from options.configer import Configer
class Visualizer:

    def __init__(self):
        self.config_dict = Configer().get_configer()
    # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
    def get_data_report(self, data_data):
        string_report = ''
        for key in data_data.keys():
            string_report += key + ' : \n' + str(data_data[key]) + '\n'
        print(string_report)



    # loss_dict，没有epoch,都是损失函数名:值（值是list）
    # input is a dict {'loss':{'model1':list},'model2':list},'accruacy':{'model1':list},'model2':list}}
    def draw_search_report(self):
        dr = DataRecorder()
        log_visualize_dict = {}
        log_visualize_dict['AVG LOSS'] = {}
        log_visualize_dict['TRAINING ACCURACY'] = {}
        search_log_dict = dr.get_search_data()
        print('VALID SPD:' + '\n')
        for k, v in search_log_dict.items():
            if v['flag'] == 1:
                avg_loss_list = [epoch_data['AVG LOSS'] for epoch_data in v['training_statics']]

                log_visualize_dict['AVG LOSS'][k] = avg_loss_list

                accuracy = [epoch_data['TRAINING ACCURACY'] for epoch_data in v['training_statics']]

                log_visualize_dict['TRAINING ACCURACY'][k] = accuracy
        graph_number = len(log_visualize_dict)
        plt.figure(1)
        idx = 1
        icon_list = []
        color_list = ['r', 'y', 'k', 'g', 'm', 'b']
        marker_list = ['-', '.', 'o', '*', 's', 'p', '+', '<', '8', 'h', 'd', 'x']
        for color in color_list:
            for marker in marker_list:
                icon_list.append(color + marker)
        plt.title("test")
        plt.ylabel('losses')
        plt.xlabel('epoch')
        plt.axis([0, 20, 0, 3])
        for item_name, model_data in log_visualize_dict.items():
            plt.title(item_name)
            plt.ylabel(item_name)
            plt.xlabel('epoch')
            plt.subplot(1, graph_number, idx)
            for model_name, data in model_data.items():
                plt.plot(range(0, len(data)), model_data[model_name], random.choice(icon_list), label=model_name)
                plt.legend(loc='upper right')
            idx += 1
        plt.show()

    def show_cam_images(self, date, time, images_per_raw, raw_num):
        cam_image_path = os.path.join(self.configer['camImagePath'], date, time)
        cam_image_list = utils.get_image_set(cam_image_path)
        cam_image_list = cam_image_list.sort()#排序
        total_image_num = len(cam_image_list)#总cam图片数量
        images_per_loop = images_per_raw * raw_num#希望每次展示的图片数量，这个数字就代表行数了
        loops = int(total_image_num / images_per_loop)#求出需要循环的次数
        for i in range(0, loops - 1):
            cam_images_list = []
            cam_list_sliced = cam_image_list[i * images_per_loop: i * images_per_loop + images_per_loop].copy()
            cam_list_sliced_length = len(cam_list_sliced)
            plt.figure(1)
            for cam_dict in cam_images_list:
                dict_length = len(cam_dict)
                idx = 1
                for cam_name, image in cam_dict.items():
                    plt.subplot(raw_num, images_per_raw, idx)
                    plt.imshow(image)
                    idx += 1
        plt.show()




if __name__ == '__main__':

    v = Visualizer()

    v.draw_search_report()




