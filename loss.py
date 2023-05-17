import torch.nn.functional as F


# 使用交叉熵损失函数
def bce_loss(pred, gt):
    loss = F.cross_entropy(pred, gt)    # loss为一个tensor
    return loss
