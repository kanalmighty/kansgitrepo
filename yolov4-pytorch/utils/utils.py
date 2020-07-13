from __future__ import division
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from model_prune.prune import CSPdarknetGA
from nets.CSPdarknet import CSPDarkNet


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # input为bs,3*(1+4+num_classes),13,13

        # 一共多少张图片
        batch_size = input.size(0)
        # 13，13
        input_height = input.size(2)
        input_width = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # 416/13 = 32
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角 batch_size,3,13,13
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())

        # anchor_left = grid_x - anchor_w/2 
        # anchor_top = grid_y - anchor_h/2 

        # rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
        # rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
        # rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # ax = fig.add_subplot(122)
        # if input_height==13:
        #     plt.ylim(0,13)
        #     plt.xlim(0,13)
        # elif input_height==26:
        #     plt.ylim(0,26)
        #     plt.xlim(0,26)
        # elif input_height==52:
        #     plt.ylim(0,52)
        #     plt.xlim(0,52)
        # plt.scatter(grid_x.cpu(),grid_y.cpu())
        # plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

        # pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2 
        # pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2 

        # rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
        # rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
        # rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # plt.show()
        # 用于将输出调整为相对于416x416的大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
        
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    print(np.shape(boxes))
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

#prediction一张图片的所有预测框数据
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # 求左上角和右下角
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 利用置信度进行第一轮筛选
        #过滤掉置信度低于阈值的框
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # 获得种类及其置信度
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # 获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获得种类
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            # 获取预测为同一类的预测框
            detections_class = detections[detections[:, -1] == c]
            # 按照存在物体的置信度排序
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # 进行非极大抑制
            #根据置信度排序以后的预测框
            max_detections = []
            while detections_class.size(0):
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                # 取一个置信度最大的预测框放入集合
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                # 取一个集合中最新的预测框和剩余的进行iou比较
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                #去掉同类别detections_class中比较结果大于阈值的预测框
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠.list转为tensor
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
#box_datas里面是组成马赛克的多张图片的经过缩放的真值框
def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:#选出一个box
            tmp_box = []
            #取出box的左上，右下角坐标
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
            #cutx,cuty是分割点的x,y坐标

            #目标是把第一个框的限制在第一象限
            if i == 0:#如果是第一个框
                #第一个框的左上角y坐标超出分割点y坐标或者左上角x坐标超出分割点x坐标，就换一个框
                if y1 > cuty or x1 > cutx:
                    continue
                #第一个框的右下角y坐标超出分割点y坐标且左上角y坐标小于分割点y坐标
                if y2 >= cuty and y1 <= cuty:
                    #一个框的右下角y坐标变为分割点y坐标
                    y2 = cuty
                    #变化以后的y1,和y2很近就换一个框
                    if y2-y1 < 5:
                        continue
                # 第一个框的右下角x坐标超出分割点x坐标或者左上角x坐标小于分割点x坐标，就换一个框
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    # 变化以后的x1,和x2很近就换一个框
                    if x2-x1 < 5:
                        continue
            # 把第二个框限制在左下角象限
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
            #把第三个框限制在右下象限
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue
            # 把第三个框限制在右上象限
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


 # gt_cls (bs, int(self.num_anchors / 3), in_h, in_w) 根据真值数据标注中那个网格中存在物体,第二维的索引是与真值框iou最大的anchor的序号
 # gt_box （bs, int(self.num_anchors/3), in_h, in_w, 4) 标注当前尺度下真值框的中心坐标，h,w,第二维的索引是与真值框iou最大的anchor的序号
 # gt_cls （bs, int(self.num_anchors/3), in_h, in_w, num_classes)根据真值数据注存在物体的网格中的物体分类,第二维的索引是与真值框iou最大的anchor的序号
 # prediction[...4]是置信度,0,3是中心,x,y,w,h
def get_batch_positive(prediction, pred_cls,gt_mask, gt_box, gt_cls, cf_thre = 0.5):
    #缩减数据方便调试
    # prediction = prediction[:,:,0:2, 0:2,:]
    # gt_cls = gt_cls[:,:,0:2, 0:2,:]
    # gt_box = gt_box[:,:,0:2, 0:2,:]
    # gt_mask = gt_mask[:,:,0:2, 0:2]
    # pred_cls = pred_cls[:,:,0:2, 0:2,:]
    batch_metric_dict = {'0': [0,0],'1': [0,0],'2': [0,0],'3': [0,0],'4': [0,0],'5': [0,0],'6': [0,0],'7': [0,0],'8': [0,0],
                         '9': [0,0],'10': [0,0],'11': [0,0],'12': [0,0],'12': [0,0],'13': [0,0],'14': [0,0],'15': [0,0]
                         ,'16': [0,0],'17': [0,0],'18': [0,0],'19': [0,0]}
    if torch.cuda.is_available():
        prediction, pred_cls, gt_mask, gt_box, gt_cls = prediction.cuda(), pred_cls.cuda(), gt_mask.cuda(), gt_box.cuda(), gt_cls.cuda()
#因为预测置信度多少的数据哪里比较？get_map492行
    prediction[..., 4] = nn.functional.softmax(prediction[..., 4])
    #prediction置信度大于阈值的mask,用mask过滤掉置信度不满足条件的真值/预测数据
    cf_mask = prediction[..., 4] > cf_thre
    prediction_cf_thre= prediction[cf_mask]
    gt_box_cf_thre = gt_box[cf_mask]
    gt_cls_cf_thre = gt_cls[cf_mask]
    #gt_cls_cf_thre获取不全位0的真值框分类向量
    valid_gt_cls_mask = torch.sum(gt_cls_cf_thre, 1) > 0


    if valid_gt_cls_mask.sum() > 0:
        gt_cls_positive = gt_cls_cf_thre[valid_gt_cls_mask]
        cls_positive = torch.argmax(gt_cls_positive, 1)
        gt_box_positive = gt_box_cf_thre[valid_gt_cls_mask]
        prediction_box_positive = prediction_cf_thre[valid_gt_cls_mask][...,0:4]
        p_g_iou = bbox_iou(gt_box_positive,prediction_box_positive)
        p_g_iou[p_g_iou >= 0.5] = 1
        p_g_iou[p_g_iou < 0.5] = -1
        for idx, cls in enumerate(cls_positive):
            if p_g_iou[idx] > 0:
                batch_metric_dict[str(cls.item())][0] = batch_metric_dict[str(cls.item())][0] + 1
            else:
                batch_metric_dict[str(cls.item())][1] = batch_metric_dict[str(cls.item())][1] + 1

    return batch_metric_dict

def tail_model_backbone(source_model_name, target_model_name,state_dict_path,device):
    target_model = eval(target_model_name)()
    # source_model = eval(source_model)()
    target_model_pretrained_dict = torch.load('D:\\datasets\\saved_model\\prune_baseline.pth', map_location=device)['state_dict']
    source_model_pretrained_dict = torch.load(state_dict_path, map_location=device)['state_dict'] \
                        if 'state_dict' in torch.load(state_dict_path, map_location=device).keys() \
                        else torch.load(state_dict_path, map_location=device)
    if target_model_name == 'CSPdarknetGA':
        source_model_pretrained_dict = {'.'.join(k.split('.') [1:]) : v for k, v in source_model_pretrained_dict.items() if 'backbone' in k}
    shared_layer_ac_dict = {k: v for k, v in source_model_pretrained_dict.items() if k in target_model.state_dict().keys()}
    target_model_pretrained_dict.update(shared_layer_ac_dict)

    target_model.load_state_dict(target_model_pretrained_dict)


    # print(target_model)
    return target_model



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tail_model_backbone('CSPdarknetGA','CSPDarkNet', 'D:\\datasets\\saved_model\\prune.pth', device)
