import torch
import torch.nn as nn
from .SENet import SENet
from .SGENet import SpatialGroupEnhance


class Dpatention(nn.Module):
    def __init__(self, in_channel=128, groups=4):
        super(Dpatention, self).__init__()
        self.SENet = SENet(in_channel=in_channel//2//groups)
        self.SGENet = SpatialGroupEnhance()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b*self.groups, -1, h, w)    # 将通道划分为g组
        x1, x2 = x.chunk(2, 1)    # 将每个子通道进一步分为两组，每组大小为b*g,c//2g,h,w
        x1 = self.SENet(x1)
        x2 = self.SGENet(x2)
        x = torch.cat([x1, x2], 1)    # 将子通道拼接
        x = x.view(b, c, h, w)    # 复原通道尺寸
        return x
