import torch
import collections
from torch import nn
import math


class DownConvLayer(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=2, padding=1):
        super(DownConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.inc = inc
        self.outc = outc
        self.layer = nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride, padding, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        self.down_sample_times = 100 / (math.ceil((100 + 2 * self.padding - self.kernel_size) / self.stride) + 1)

    def forward(self, x):
        x_h, x_w = x.shape[2:]
        self.output_h = str(math.ceil((x_h + 2 * self.padding - self.kernel_size) / self.stride))
        self.output_w = str(math.ceil((x_w + 2 * self.padding - self.kernel_size) / self.stride))
        self.output_size = '(' + str(self.outc) + ',' + str(self.output_h) + ',' + str(self.output_w) + ')'
        return self.layer(x)


class GroupUpConvLayer(nn.Module):
    def __init__(self, inc, outc, kernel_size=4, stride=2, padding=1):
        super(GroupUpConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.inc = inc
        self.outc = outc
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(self.inc, self.inc, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_h, x_w = x.shape[2:]
        self.output_h = (x_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        self.output_w = (x_w - 1) * self.stride - 2 * self.padding + self.kernel_size
        self.output_size = '(' + str(self.outc) + ',' + str(self.output_h) + ',' + str(self.output_w) + ')'
        return self.layer(x)


class GroupDownConvLayer(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=2, padding=1):
        super(GroupDownConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.inc = inc
        self.outc = outc
        self.layer = nn.Sequential(
            nn.Conv2d(inc, inc, 3, stride, 1, groups=inc, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_h, x_w = x.shape[2:]
        self.output_h = math.ceil((x_h + 2 * self.padding - self.kernel_size) / self.stride)
        self.output_w = math.ceil((x_w + 2 * self.padding - self.kernel_size) / self.stride)
        self.output_size = '(' + str(self.outc) + ',' + str(self.output_h) + ',' + str(self.output_w) + ')'
        return self.layer(x)

#
# class ClassifyLayer(nn.Module):
#     def __init__(self, class_number, mode):
#         super(ClassifyLayer, self).__init__()
#         if mode == 'segment':
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.inc = inc
#         self.outc = outc
#         self.layer = nn.Sequential(
#             nn.Conv2d(inc, inc, 3, stride, 1, groups=inc, bias=False),
#             nn.BatchNorm2d(inc),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inc, outc, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(outc),
#             nn.ReLU(inplace=True))

    def forward(self, x):
        x_h, x_w = x.shape[2:]
        self.output_h = math.ceil((x_h + 2 * self.padding - self.kernel_size) / self.stride)
        self.output_w = math.ceil((x_w + 2 * self.padding - self.kernel_size) / self.stride)
        self.output_size = '(' + str(self.outc) + ',' + str(self.output_h) + ',' + str(self.output_w) + ')'
        return self.layer(x)


class Assembler(nn.Module):
    def __init__(self, stages_dict, input_c, class_number, cof=64):
        super(Assembler, self).__init__()
        layers_dict = collections.OrderedDict()
        self.class_number = class_number
        self.stage_dict = stage_dict
        down_layer_index = -1
        up_layer_index = -1
        for layers_name, layer_number in self.stage_dict.items():
            if layers_name == 'DownConvLayer':
                for i in range(layer_number):
                    down_layer_index += 1
                    layers_dict['DownConvLayer_' + str(down_layer_index)] = DownConvLayer(input_c if down_layer_index == 0 else pow(2, down_layer_index-1)*cof, pow(2,(down_layer_index)) * cof)

            if layers_name == 'GroupDownConvLayer':
                for i in range(layer_number):
                    down_layer_index += 1
                    layers_dict['GroupDownConvLayer' + str(down_layer_index)] = GroupDownConvLayer(input_c if down_layer_index == 0 else pow(2, down_layer_index-1)*cof, pow(2, (down_layer_index)) * cof)

            if layers_name == 'GroupUpConvLayer':
                for i in range(layer_number):
                    #判断是是否为上采样最后一层，如果是则为分类层
                    if up_layer_index == layer_number - 2:
                        up_layer_index += 1
                        layers_dict['GroupUpConvLayer' + str(up_layer_index)] = GroupUpConvLayer(pow(2, down_layer_index - up_layer_index) * cof, self.class_number)
                    else:
                        up_layer_index += 1
                        layers_dict['GroupUpConvLayer' + str(up_layer_index)] = GroupUpConvLayer(pow(2, down_layer_index - up_layer_index) * cof, pow(2, down_layer_index - up_layer_index - 1) * cof)
        self.layers = nn.ModuleDict(layers_dict)


    def get_structure(self):
        for layer_name, layer in self.layers.items():
            print('%s input channel: %s, output channel  %s' % (layer_name, layer.inc, layer.outc))

    def check_stage_dict(self):
        #下采样几倍，则上采样几倍，对图片尺寸进行还原
        assert self.stage_dict['GroupDownConvLayer'] == self.stage_dict['GroupUpConvLayer']


    def forward(self, x):
        output_dict = {}
        x_c, x_h, x_w = 0, 0, 0
        for layer_name, layer in self.layers.items():

            #分割网络调层连接
            if str(x_c + x_h + x_w) in output_dict.keys():
                x = torch.add(x, output_dict[str(x_c + x_h + x_w)])

            x = layer(x)
            #保存之前层的输出
            x_c, x_h, x_w = x.shape[1:]
            x_key = x_c + x_h + x_w
            output_dict[str(x_key)] = x

            #如果下采样输出的尺寸太小了就报错
            if int(layer.output_h) * int(layer.output_w) < 30:
                raise ValueError("输出尺寸太小了，请调整参数")
        return x




if __name__ == '__main__':
    stage_dict = {'GroupDownConvLayer': 5, 'GroupUpConvLayer': 5}#个数
    a = Assembler(stage_dict, 3, 2,16)
    a.get_structure()
    input = torch.randn(64, 3, 224, 224)
    x = a.forward(input)
