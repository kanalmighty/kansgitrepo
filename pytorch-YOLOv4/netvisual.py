import torch
from tensorboardX import SummaryWriter
from models import *
dummy_input = torch.rand(8, 3, 320, 320) #假设输入13张1*28*28的图片
model = Yolov4()
with SummaryWriter(comment='aa') as w:
    w.add_graph(model, (dummy_input, ))