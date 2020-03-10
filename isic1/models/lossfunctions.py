import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from data.imageprocessor import ImageProcessorBuilder
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = torch.ones(class_num, 1)*alpha
            else:
                self.alpha = torch.tensor(torch.ones(class_num, 1)*alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = torch.tensor(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# class AttentionLoss(nn.Module):
#
#     def __init__(self, args):
#         self.args = args
#         self.ip = ImageProcessorBuilder(self.args)
#
#     #输入tensor输出这个batch的图像二值化以后的N维向量
#     def get_input_binary_tensor(self, input_batch):
#         #get input_binary
#         batch_binary_vector = self.ip.get_input_binary(input_batch)
#         batch_binary_ndarray = np.array(batch_binary_vector)
#         return batch_binary_ndarray
#
#     def get_cam_binary_tensor(self, net, input_batch):
#         #get cam
#         heatmap_list = get_cam_for_training(self.args, net, input_batch)
#         cam_binary_vector = self.ip.get_cam_binary(heatmap_list)
#         cam_binary_vector = np.array(cam_binary_vector)
#         return cam_binary_vector
#
#
#     def get_attention_loss(self, net, input_batch):
#         a = self.get_input_binary_tensor(input_batch)
#         b = self.get_cam_binary_tensor(net, input_batch)
#         d1 = np.sqrt(np.sum(np.square(a-b)))/self.args.batchsize
#
#         return d1


