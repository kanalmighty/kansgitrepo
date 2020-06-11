# import torch
# from torchsummary import summary
# from nets.CSPdarknet import darknet53
# from nets.yolo4 import YoloBody
#
# if __name__ == "__main__":
#     # 需要使用device来指定网络在GPU还是CPU运行
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = YoloBody(3,20).to(device)
#     summary(model, input_size=(3, 416, 416))

import torch
from nets.yolo4 import YoloBody
from tensorboardX import SummaryWriter
dummy_input = torch.randn(4,3,416,416) #假设输入13张1*28*28的图片
model = YoloBody(9,20)
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input, ))