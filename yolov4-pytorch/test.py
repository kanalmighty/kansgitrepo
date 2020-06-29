import torch
from torch import nn
from torchsummary import summary
from nets.CSPdarknet import darknet53, CSPDarkNet
from nets.yolo4 import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSPDarkNet([1, 2, 8, 8, 4])
    # print(model)
    for name, layer in model.named_modules():
        if len(name.split('.')) == 7:
            for i in layer.modules():
                if isinstance(i, nn.BatchNorm2d):
                    print(i)
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print(m)
    # summary(model, input_size=(3, 416, 416))
    print(model.children())
import torch
from nets.yolo4 import YoloBody
from tensorboardX import SummaryWriter
# dummy_input = torch.randn(4,3,416,416) #假设输入13张1*28*28的图片
# model = YoloBody(9,20)
# with SummaryWriter(comment='LeNet') as w:
#     w.add_graph(model, (dummy_input, ))