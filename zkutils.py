import numpy as np
import mxnet as nd
import gzip
import os


# 双线性插值函数，生成权重用与更新反卷积网络的参数
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]#ogrid函数作为产生numpy数组与numpy的arange函数功能有点类似，不同的是：
    filt = (1 - abs(og[0] -  center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)
