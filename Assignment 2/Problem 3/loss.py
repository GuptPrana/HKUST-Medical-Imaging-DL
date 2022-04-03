# Focal Loss
import torch
import numpy
from torch import nn 
import torch.nn.functional as F
import torchvision
from torchvision.ops import focal_loss

class Focal_Loss(nn.Module):
    def __init__(self, weight=None):
        super(Focal_Loss, self).__init__()

    def forward(self, output, label):
        # assert same size
        bce = nn.BCEWithLogitsLoss(output, label)
        image = F.softmax(image)
        image = nn.Flatten(image).sum(-1)
        label = nn.Flatten(label).sum(-1)
        # Dice loss
        num = (image*label).sum(-1) + 1
        den = image.sum(-1) + label.sum(-1) + 1
        dice = 2*num/den
        # Can add dice to summaryWriter? 
        # BCE Loss
        BCE = F.binary_cross_entropy(image, label, reduction="mean")
        return ((BCE + 1-dice), dice)