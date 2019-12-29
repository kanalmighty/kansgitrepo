import matplotlib.pyplot as plt


class Visualizer:

    # 传入的data是一给字典，第个位置是epoch,后面是损失函数名:值
    def get_data_report(self, loss_data):
        string_report = ''
        for key in loss_data.keys():
            string_report += key + ' = ' + str(loss_data[key]) + '\n'
        print(string_report)

    # loss_dict，没有epoch,都是损失函数名:值（值是list）
    # input is a loss dict
    def draw_picture_block(self, loss_dict):
        icons = ['r--', 'g^', 'cs', 'k*', 'bs', 'yv', 'r+', 'b<', 'm8']
        plt.title("test")
        plt.ylabel('losses')
        plt.xlabel('epoch')
        for idx, key in enumerate(loss_dict.keys()):
            if not isinstance(loss_dict[key], list):
                raise TypeError("the loss: %s in the dict is not a list" % loss_dict[key] )
            plt.plot(range(0, len(loss_dict[key])), loss_dict[key], icons[idx], label=key)
        plt.legend(loc='upper right')
        plt.show()




if __name__ == '__main__':
    data = {'loss_g': [10,9,8,7,6,5,4,3,2,1],'loss_d': [1,2,3,4,5,6,4,3,2,1],'loss_y': [3,4,6,9,5,10,4,13,1,5]}
    visualizer = Visualizer()
    visualizer.draw_picture_block(data)


