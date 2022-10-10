# Model
import numpy as np
import torch 
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Misc
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.3)
        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Layer 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        # relu
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        # relu, pool

        # Layer 2
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #bn2
        # relu
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        # relu, dropout
        # pool

        # Layer 3
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # bn3, relu
        self.conv6 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        # pool

        # Layer 4
        self.conv7 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        # bn4, relu
        self.conv8 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(512)

        # Upscale
        # Layer 3
        self.decon1 = nn.Conv3d(512+256, 256, kernel_size=3, stride=1, padding=1)
        #bn4, relu
        self.decon2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        #bn4, relu

        # Layer2
        self.decon3 = nn.Conv3d(128+256, 128, kernel_size=3, stride=1, padding=1)
        #bn3, relu
        self.decon4 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        #bn3, relu

        # Layer1
        self.decon5 = nn.Conv3d(128+64, 64, kernel_size=3, stride=1, padding=1)
        #bn2, relu
        self.decon6 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        #bn2, relu
        self.decon7 = nn.Conv3d(64, 1, 3, 1, 1)

    def forward(self, input):

        # L1
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.pool(out)
        carry1 = out

        # L2
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.pool(out)
        carry2 = out

        # L3
        out = self.conv5(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.bn4(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.pool(out)
        carry3 = out

        # L4
        out = self.conv7(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv8(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.pool(out)

        # L3
        out = self.upsample(out)
        out = torch.cat([out, carry3], dim=1)
        out = self.decon1(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.decon2(out)
        out = self.bn4(out)
        out = self.relu(out)

        # L2
        out = self.upsample(out)
        out = torch.cat([out, carry2], dim=1)
        out = self.decon3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.decon4(out)
        out = self.bn3(out)
        out = self.relu(out)

        # L1
        out = self.upsample(out)
        out = torch.cat([out, carry1], dim=1)
        out = self.decon5(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.decon6(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.upsample(out)
        out = self.decon7(out)
        out = self.relu(out)

        return out
