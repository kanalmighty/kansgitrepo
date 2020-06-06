# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import logging
import os, sys
from tqdm import tqdm
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
import argparse
from easydict import EasyDict as edict
from torch.nn import functional as F

import numpy as np


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    #bboxes_a是当前图片当前尺度下的所有有效真值框的0，0，w，h，形状(图片有效真值框个数,4)
    #bboxes_b，初始9个anchor被缩放到当前尺度后的w,h，其中前两位为0，后两位anchor的h,w
    # 例子:tensor([[ 0.0000,  0.0000,  1.2500,  1.6250],
        # [ 0.0000,  0.0000,  2.0000,  3.7500],
        # [ 0.0000,  0.0000,  4.1250,  2.8750],
        # [ 0.0000,  0.0000,  3.7500,  7.6250],
        # [ 0.0000,  0.0000,  7.7500,  5.6250],
        # [ 0.0000,  0.0000,  7.3750, 14.8750],
        # [ 0.0000,  0.0000, 14.5000, 11.2500],
        # [ 0.0000,  0.0000, 19.5000, 24.7500],
        # [ 0.0000,  0.0000, 46.6250, 40.7500]])
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl_bboxes_a = bboxes_a[:, None, :2]#形状为(有效真值框个数,1,2)，2取出的全是0
        tl_bboxes_b = bboxes_b[:, :2]#(9,2)9个当前迟度下的先验anchor中取出0的数
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])#(有效真值框个数,9,2)全是0
        # bottom right

        br_bboxes_a = bboxes_a[:, None, 2:]  # 形状为(有效真值框个数,1,2)，2取出的全是w,h
        br_bboxes_b = bboxes_b[:, 2:]  # (9,2)9个当前迟度下的先验anchor中取出w,h

        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])#形状(有效真值框个数,9,2)，后面两位取当前真值框w,h和当前尺度
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)#这个操作就是把当前尺度下有效真值框的w*h算出来
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)#这个操作就是把当前尺度下先验9个anchor的的w*h算出来
    else:
        #bboxes_a pred_box(4800,4),4为预测框中心x坐标,预测框中心y,w,h,bboxes_b truth box(有效真值框个数，4))
        #中心x坐标 - w/2得到预测框左上角坐标和真值框左上角坐标
        #torch.max取大的左上角坐标
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # 右下角坐标取小的
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    #(tl < br).type(tl.type())把bool转换位0，1
    en = (tl < br).type(tl.type()).prod(dim=2)#en是br<tl的掩码
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())#交集面积
    return area_i / (area_a[:, None] + area_b - area_i)


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []
        for i in range(3):

            # all_anchors_grid放的原始的9个anchor缩小到对应的尺度feature上的大小以后的大小，尺度由i控制，shape 9*2
            #例子[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875), (3.75, 7.625), (7.75, 5.625), (7.375, 14.875), (14.5, 11.25), (19.5, 24.75), (46.625, 40.75)]
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]

            # i=0,j=1,2,3, i=1,j=3,4,5,每次i迭代masked_anchors从all_anchors_grid取出三个anchor,masked_anchors的shape 3*2，代表3个anchor的w,h
            # masked_anchors的一个例子
            # [[1.25  1.625]
            #  [2.    3.75]
            # [4.125 2.875]]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)

            # 9*4矩阵 全是0
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)

            # 每行后两位放入all_anchors_grid的长宽数据，一共9个
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)


            ref_anchors = torch.from_numpy(ref_anchors)

            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]#下采样以后的feature map尺寸

            # b,3,h,w的tensor,每行都是0-39,网格的x坐标
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)

            # b,3,h,w的tensor,第0行全是是0，最后一行是39，网格y坐标
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)

            # 每次取出all_anchors_grid中三个anchor的w并扩展为(b,anchor的个数,h,w)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            # 每次all_anchors_grid中三个anchor的h并扩展为(batchsize,3,当前feature map的h,当前feature map的w)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            #masked_anchors9个初始anchor分成三组，每组缩放到3个尺度
            self.masked_anchors.append(masked_anchors)

            # ref_anchors为9*4矩阵 全是0,每行后两位放入原始的9个anchor缩小到对应的尺度feature上的大小以后的的长宽数据，一共9个
            self.ref_anchors.append(ref_anchors)

            # self.grid_x存放三种不同feature map尺度下的网格x坐标,主干网络输出的feature map有多少个像素，就对应原图多少个网格
            self.grid_x.append(grid_x)

            # self.grid_x存放三种不同feature map尺度下的网格y坐标,主干网络输出的feature map有多少个像素，就对应原图多少个网格
            self.grid_y.append(grid_y)


            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)


#就是将目标先进行三种下采样，分别和目标落在的网格产生的 9个anchor分别计算iou，大于阈值0.3的记为正样本。
# 如果9个iou全部小于0.3，那么和目标iou最大的记为正样本。对于正样本，我们在label上 相对应的anchor位置上，赋上真实目标的值。
    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        #labels = labels.cpu().data
        aa1 = labels.sum(dim=2)#label的最后一维也就是5个数字加起来
        aa2 = (labels.sum(dim=2) > 0)#生成真值框的掩码，有检测对象的是true,没有的是false
        aa3 = aa2.sum(dim=1)#第一维相加，等到当前这个batch里面每张图片的真值框的个数
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # 这个 batch中每张图片中的对象个数number of objects
        #label里面是整个batch的真值框的的左上角坐标，右上角坐标(x_l,y_l,x_r,y_r)

        #把真值框下的数据分别缩放到当前尺度，并进行适当变化
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)#60个真值框在当前尺度下的x轴方向中心点坐标,shape(batch size,60)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)#60个真值框在当前尺度下的y轴方向中心点坐标,shape(batch size,60)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]#60个真值框的w缩放到当前尺度，(batch size,60)代表一个batch中所有真值框的w
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]#60个真值框的h缩放到当前尺度，,shape(batch size,60)
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()#整个batch中每张图片的真值框坐标转为整数，获取单元格的左上角x坐标
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()#整个batch中每张图片的真值框坐标转为整数，获取单元格的左上角y坐标

        #进入batch中的一张图片
        for b in range(batchsize):
            n = int(nlabel[b])#第b张图片里的真值框个数
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)#定义一个向量存放单张图片所有真值框的w,h



            truth_box[:n, 2] = truth_w_all[b, :n]#当前从label中取出当前尺度下有效真值框的w放入形状为(真值框个数,4)的tensor
            truth_box[:n, 3] = truth_h_all[b, :n]#当前从label中取出取出当前尺度下有效真值框的h放入形状为(真值框个数,4)的tensor
            truth_i = truth_i_all[b, :n]#当前图片当前尺度下的有效真值框的x方向中心点，这里存放的都是int
            truth_j = truth_j_all[b, :n]#当前图片当前尺度下的有效真值框的y轴方向中心点，这里存放的都是int

            # calculate iou between truth and reference anchors
            #truth_box里面放的单张图片中有效真值框的w,h数据
            #self.ref_anchors长度为3的列表，3组数据分别存放初始9个anchor被缩放到三个不同尺度后的w,h，其中前两位为0，后两位anchor的h,w
            #output_id代表第i个尺寸的feature map
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id])#shape(有效真值框个数,9)应该是有效每个有效真值框与当前尺度下的9个anchor的iou
            best_n_all = anchor_ious_all.argmax(dim=1)#取每个真值框最大的iou的anchor在所有当前尺度下anchor中的索引，shape(有效真值框个数,1)
            best_n = best_n_all % 3#每个数除以3 取余

            best_n_mask_0 = best_n_all == self.anch_masks[output_id][0]#形状9*1，best_n_all中的索引 = self.anch_masks[output_id][0]值的位置为true
            best_n_mask_1 = best_n_all == self.anch_masks[output_id][1]
            best_n_mask_2 = best_n_all == self.anch_masks[output_id][2]
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))
            #上面三个向量取或，得到n个真值框的IOU最大anchor是否在self.anch_masks第i组中，是不是可以理解为为当前尺度下，与真值框iou最大的anchor
            # 正好在为当前尺度配置准备的三个anchor中

            if sum(best_n_mask) == 0:#如果当前与当前图片真值框IOU最大的anchor不在与分辨率对应的3个anchor中则跳过放弃这张图片及其label
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]#当前尺度下，当前图片中的有效真值框的x轴方向中心点坐标放入形状为(真值框个数,4)的tensor
            truth_box[:n, 1] = truth_y_all[b, :n]#当前从label中取出当前尺度下有效真值框的y轴方向中心点坐标放入形状为(真值框个数,4)的tensor
            #truth_box形状为(有效真值框个数,4)的tensor，4代表真值框的中心x,y，w,h
            #pred[b]shape 3,40,40,4
            #pred_ious shape(4800,真值框个数)，预测出的4800的框与15个真值框的iou
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)#(pred[b].view(-1, 4)=(4800,4))
            pred_best_iou, _ = pred_ious.max(dim=1)#测出的4800的框与n个真值框的iou,取出最大iou
            pred_best_iou = (pred_best_iou > self.ignore_thre)#大于阈值的位1生成掩码

            # pred[b].shape[:3]是(3,feature map的w,feature map的h)
            #代表每个像素点上的3个预测框中与真值框iou最大的预测框
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truthanchor_ious_all
            obj_mask[b] = ~ pred_best_iou#~代表逐位取反，false变true

            for ti in range(best_n.shape[0]):#best_n.shape[0]真值框个数
                if best_n_mask[ti] == 1:#如果与第ti个真值框iou最大的anchor在self.anch_masks前i个中
                    i, j = truth_i[ti], truth_j[ti]#当前图片当前尺度下的有效真值框的x,y方向中心点，这里都是int
                    # 每个真值框最大的iou的anchor索引，shape(有效真值框个数,1)，除以3取余，
                    # best_n_mask[ti] == 1情况下可以认为a就是索引值
                    a = best_n[ti]
                    #b, a, j, i用来标注当feature map的哪个像素点的哪个anchor
                    obj_mask[b, a, j, i] = 1#(batch size, self.n_anchors=3,fsize,fisize)
                    tgt_mask[b, a, j, i, :] = 1#(batch size, self.n_anchors=3,fsize,fisize,84),少一个是否存在物体的mask码

                    # target 形状(batch size, self.n_anchors=3,fsize,fisize,85)的，最后一维第的0,1位放入第ti个真值框中心点的小数部分
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)

                    # target 形状(batch size, self.n_anchors=3,fsize,fisize,85)的，最后一维第的2,3位放入第ti个真值框中心点的小数部分

                    # masked_anchors9个初始anchor分成三组，每组缩放到3个尺度下的9个anchor
                    #当前尺度下的3个anchors，取anchor索引值除以3的余数 位的w

                    #self.masked_anchors[output_id]当前尺度下的anchor
                    #反解出理想状态下的的t_w,t_h
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)

                    #标著哪个框有物体
                    target[b, a, j, i, 4] = 1

                    # 标准类别
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    #2-真值框面积/特征图面积后开根号
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):#output_id给第i个feature map
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize) #self.n_anchors =3
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous() （batchsize, self.n_anchors, fsize, fsize，n_ch)

            # logistic activation for xy, obj, cls

            #从预测结果中拿出x,y,obe,cls激活以后再放回预测结果向量
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            #取出预测框的4个坐标
            pred = output[..., :4].clone()#（batchsize, self.n_anchors, fsize, fsize，4)
            # self.grid_x是(3,b, 3, h, w)的tensor, 每行都是0 - 39, 网格的x坐标
            pred[..., 0] += self.grid_x[output_id]#预测出的激活后的偏差t_x+对应的当前尺度下的网格x坐标，得到预测框的中心x坐标
            # self.grid_y是(3,b, 3, h, w)的tensor,第0行全是是0，最后一行是39，网格y坐标
            pred[..., 1] += self.grid_y[output_id]#预测出的激活后的偏差t_y+对应的当前尺度下的网格x坐标，得到预测框的中心x坐标
            #self.anchor_w长度为3是个list,第0个位置存放了self.anchors中的前三个anchor的w/当前尺度以后扩张为(batchsize, 3, 当前feature map的h, 当前feature map的w)
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]#当前尺度下的anchor框的w，加上预测出的偏移取对数获取预测框的w
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]#当前尺度下的anchor框的h，加上预测出的偏移取对数获取预测框的h

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # obj_mask (batch size, self.n_anchors=3,fsize,fisize),有物体的那个网格值为1
            # tgt_mask (batch size, self.n_anchors=3,fsize,fisize,84),存在该分辨率下的有效anchor时(匹配anchor和gt的iou大于阈值)则图片像素位置对应的84个值都时1
            #target 形状(batch size, self.n_anchors=3,fsize,fisize,85) 85的0，1位是缩放到当前尺度下的x,y相对于网格左上角的偏移，其实就是最优的tx,ty，2，3位反解出理想状态下的的t_w,t_h，后85为对应的分类位置为1，其余是0
            #tgt_scale 为(2-真值框面积/特征图面积后开根号)


            # loss calculation
            #output是当前分辨率的feature map(batch size,3,fsize,fsize,85)
            #feature map最后一维的第4位标注位为1，代表这里有物体
            output[..., 4] *= obj_mask#当前分辨率的feature map中的某个像素(对应原始图片n*n的区域)是否有物体的预测，1为有物体，0为没有
            u = np.r_[0:4, 5:n_ch] #代表0，3到83，没有4

            # feature map最后一维的第4位标注位为1，代表这里有物体中心坐标，w,h，分类数据都保留
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            # feature map最后一维的第4位标注位为1，代表这里有物体中心坐标，w,h*权重
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config)
    val_dataset = Yolo_dataset(config.val_label, config)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                               pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                            pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    criterion = Yolo_loss(device=device, batch=config.batch//config.subdivisions)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if i % config.subdivisions == 0:
                    optimizer.step()
                    model.zero_grad()

                if epoch_step % log_step == 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('loss_xy/train', loss_xy.item(), global_step)
                    writer.add_scalar('loss_wh/train', loss_wh.item(), global_step)
                    writer.add_scalar('loss_obj/train', loss_obj.item(), global_step)
                    writer.add_scalar('loss_cls/train', loss_cls.item(), global_step)
                    writer.add_scalar('loss_l2/train', loss_l2.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item()
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item()))

                pbar.update(images.shape[0])

            if save_cp:
                try:
                    os.mkdir(config.checkpoints)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(model.state_dict(), os.path.join(config.checkpoints, f'Yolov4_epoch{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # model = model.cuda()
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
