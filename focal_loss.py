import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:  # alpha 是平衡因子
            self.alpha = Variable(0.25*torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def forward(self, inputs, targets):
        # targets: 真值，大小为[batch_size]的一维tensor
        # inputs: 预测值，大小为[batch_size, num_classes]的二维tensor
        B, C = inputs.size()
        targets = targets.reshape(B, -1)  # 变为[batch_size,1]的形式
        P = F.softmax(inputs, dim=1)  # 先求p_t，将预测值之和归一化，转换为概率形式，大小为[batch_size,num_classes]
        class_mask = inputs.data.new(B, C).fill_(0)    # 全为0
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # [N,C]的tensor，其中每一列为真值的类值为1，其余为0

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = self.alpha[ids.data.view(-1)]
        # y*p_t  如果这里不用*， 还可以用gather提取出正确分到的类别概率。
        # 之所以能用sum，是因为class_mask已经把预测错误的概率清零了。
        probs = (P * class_mask).sum(1).view(-1, 1)    # [batch_size,1]
        # y*log(p_t)
        log_p = probs.log()
        # -a * (1-p_t)^2 * log(p_t)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

