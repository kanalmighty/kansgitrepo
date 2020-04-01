import torch
import collections
from torch import nn
import math


def get_upconv_setting(s):
    for i in range(3, 8):
        for j in range(0, 5):
            if i - 2*j - s == 0:
                return i, j, int(s)
    print('could find shit.')
    return False


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
        print('GroupUpConvLayer' + self.output_size)
        return self.layer(x)

#降维卷积层，用于给encoder的feature map在跳层连接之前降维
class DimReduConvLayer(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1):
        super(DimReduConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.inc = int(inc)
        self.outc = int(outc)
        self.layer = nn.Sequential(
            nn.Conv2d(self.inc, self.outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.outc),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_h, x_w = x.shape[2:]
        self.output_h = (x_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        self.output_w = (x_w - 1) * self.stride - 2 * self.padding + self.kernel_size
        self.output_size = '(' + str(self.outc) + ',' + str(self.output_h) + ',' + str(self.output_w) + ')'
        print('DimReduConvLayer ' + self.output_size)
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
        print('GroupDownConvLayer '+ self.output_size)
        return self.layer(x)

# n次下采样的总倍数2^n
# 下采样第d_i层的size计算：原始size/(2^(i+1))
# 下采样第d_i层的输出通道数：2^(i)*cof
# 上采样m层，每层分辨率上采样倍数:2^(n/m)
# 上采样u_i层输出通道数cof*2^(N-i-2),0<=i<m-1
# 降维卷积r_i的输出通道数与上采样层u_i的输入通道数一致等于u_i-1层的输出通道数
# 降维卷积层r_i的输入尺寸为原始size*2^(n(i-m+1)/m)
# 降维卷积层r_i的输入通道数量为(n*m-n*i-m)/m

class Assembler(nn.Module):
    def __init__(self, stages_dict, input_c, class_number, cof=64):
        super(Assembler, self).__init__()
        down_layers_dict = collections.OrderedDict()
        up_layers_dict = collections.OrderedDict()
        skip_layers_dict = collections.OrderedDict()
        self.class_number = class_number
        self.stage_dict = stages_dict
        n = stages_dict['GroupDownConvLayer']
        m = stages_dict['GroupUpConvLayer']
        k, p, s = get_upconv_setting(pow(2,n/m))
        print(k, p, s)
        down_layer_index = -1
        up_layer_index = -1
        for layers_name, layer_number in self.stage_dict.items():
            if layers_name == 'DownConvLayer':
                for i in range(layer_number):
                    down_layer_index += 1
                    down_layers_dict['DownConvLayer_' + str(down_layer_index)] = DownConvLayer(input_c if down_layer_index == 0 else pow(2, down_layer_index-1)*cof, pow(2, down_layer_index) * cof)

            if layers_name == 'GroupDownConvLayer':
                for i in range(layer_number):
                    down_layer_index += 1
                    down_layers_dict['GroupDownConvLayer_' + str(down_layer_index)] = GroupDownConvLayer(input_c if down_layer_index == 0 else pow(2, down_layer_index-1)*cof, pow(2, (down_layer_index)) * cof)

            if layers_name == 'GroupUpConvLayer':
                for i in range(layer_number):
                    #判断是是否为上采样最后一层，如果是则为分类层
                    if up_layer_index == layer_number - 2:
                        up_layer_index += 1
                        skip_layers_dict['DimReduConvLayer_' + str(up_layer_index)] = DimReduConvLayer(cof * pow(2, (n*m-n*up_layer_index - m)/m), pow(2, down_layer_index - up_layer_index) * cof, kernel_size=k, stride=s, padding=p)
                        up_layers_dict['GroupUpConvLayer_' + str(up_layer_index)] = GroupUpConvLayer(pow(2, down_layer_index - up_layer_index) * cof, self.class_number, kernel_size=k, stride=s, padding=p)
                        #降维卷积层
                    else:
                        up_layer_index += 1
                        up_layers_dict['GroupUpConvLayer_' + str(up_layer_index)] = GroupUpConvLayer(pow(2, down_layer_index - up_layer_index) * cof, pow(2, down_layer_index - up_layer_index - 1) * cof, kernel_size=k, stride=s, padding=p)
                        # 降维卷积层
                        skip_layers_dict['DimReduConvLayer_' + str(up_layer_index)] = DimReduConvLayer(cof * pow(2, (n*m-n*up_layer_index - m)/m), pow(2, down_layer_index - up_layer_index) * cof, kernel_size=k, stride=s, padding=p)

        self.down_layers = nn.ModuleDict(down_layers_dict)
        self.up_layers = nn.ModuleDict(up_layers_dict)
        self.skip_layers = nn.ModuleDict(skip_layers_dict)


    def get_structure(self):
        for layer_name, layer in self.down_layers.items():
            print('%s input channel: %s, output channel  %s' % (layer_name, layer.inc, layer.outc))
        for layer_name, layer in self.skip_layers.items():
            print('%s input channel: %s, output channel  %s' % (layer_name, layer.inc, layer.outc))
        for layer_name, layer in self.up_layers.items():
            print('%s input channel: %s, output channel  %s' % (layer_name, layer.inc, layer.outc))

    def select_skip_fm(self, image_h):
        downsample_layer_number = self.stage_dict['GroupDownConvLayer']
        upsample_layer_number = self.stage_dict['GroupUpConvLayer']
        encoder_output_size = image_h/pow(2, downsample_layer_number)
        #根据上采样过程中的feature map大小选择下采样的feature map
        #计算每次上采样倍数
        if not downsample_layer_number % upsample_layer_number == 0:
            raise TypeError('下采样层数必须是上采样层数的整数倍')
        upsample_factor = pow(2, downsample_layer_number/upsample_layer_number)
        #计算历次上采样过程中产生的feature map大小
        upsample_size_list = [int(encoder_output_size)*pow(upsample_factor, i) for i in range(1, int(upsample_factor))]
        return upsample_size_list


    def check_stage_dict(self):
        #下采样几倍，则上采样几倍，对图片尺寸进行还原
        assert self.stage_dict['GroupDownConvLayer'] == self.stage_dict['GroupUpConvLayer']


    def forward(self, x):
        original_image_h = x.shape[2]
        output_dict = {}
        upsample_size_list = self.select_skip_fm(original_image_h)
        print('upsample_size_list', upsample_size_list)
        if original_image_h in upsample_size_list:
            output_dict[str(original_image_h)] = x


        #先进行下采样，保存跳层连接用的feature map
        for layer_name, layer in self.down_layers.items():
            x = layer(x)
            #保存之前层的输出
            x_h = x.shape[2]
            if x_h in upsample_size_list:
                output_dict[str(x_h)] = x

            #如果下采样输出的尺寸太小了就报错
            if int(layer.output_h) * int(layer.output_w) < 30:
                raise ValueError("输出尺寸太小了，请调整参数")
        #进行上采样及跳层连接
        for layer_name, layer in self.up_layers.items():
            idx = layer_name.split('_')[-1]
            x_h = x.shape[2]
            #跳层连接
            if str(x_h) in output_dict.keys():

                r_out = self.skip_layers['DimReduConvLayer_' + str(idx)](output_dict[str(x_h)])
                print('DimReduConvLayer变换',  output_dict[str(x_h)].shape, r_out.shape)
                print('add',r_out.shape, x.shape)
                x = torch.add(r_out, x)

                x = layer(x)
            else:
                x = layer(x)
        return x




if __name__ == '__main__':
    stage_dict = {'GroupDownConvLayer': 6, 'GroupUpConvLayer': 3}#个数
    a = Assembler(stage_dict, 3, 2, 16)
    # a.get_structure()
    input = torch.randn(4, 3, 512, 512)
    x = a.forward(input)
