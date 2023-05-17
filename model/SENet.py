import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, in_channel=128, r=2):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Linear(in_features=in_channel, out_features=in_channel//r)
        self.relu = nn.ReLU()
        self.fc_layer2 = nn.Linear(in_features=in_channel//r, out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        x = self.sigmoid(x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = x*x0
        return x
