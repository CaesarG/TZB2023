import torch.nn as nn
import torch


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channels):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=1, stride=1,
                              padding=0)

    def forward(self, x):  # x(b,c,w,h) -> y(b,2c,w,h/2)
        return self.conv(torch.cat([x[..., :256], x[..., 256:]], 1))
