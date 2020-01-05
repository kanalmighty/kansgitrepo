import random

import matplotlib.pyplot as plt
from data.datarecorder import DataRecorder

class Visualizer:

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




if __name__ == '__main__':

    v = Visualizer()

    v.draw_search_report()




