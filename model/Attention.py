import torch
import torch.nn as nn
from .SENet import SENet
from .SGENet import SpatialGroupEnhance


class Dpattention(nn.Module):
    def __init__(self, in_channel=128, groups=4):
        super(Dpattention, self).__init__()
        self.SENet = SENet(in_channel=in_channel // 2 // groups)
        self.SGENet = SpatialGroupEnhance()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)  # 将通道划分为g组
        x1, x2 = x.chunk(2, 1)  # 将每个子通道进一步分为两组，每组大小为b*g,c//2g,h,w
        x1 = self.SENet(x1)
        x2 = self.SGENet(x2)
        x = torch.cat([x1, x2], 1)  # 将子通道拼接
        x = x.view(b, c, h, w)  # 复原通道尺寸
        return x


class CA_BLOCK(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_BLOCK, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
