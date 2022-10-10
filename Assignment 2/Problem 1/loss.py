import torch
import numpy
from torch import nn 
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()

    def forward(self, image, label, batch_size, chwd):
        image = torch.sigmoid(image)
        image = torch.reshape(image, (batch_size, chwd))
        label = torch.reshape(label, (batch_size, chwd))
        label.float()
        # image = nn.Flatten(image) # [N, 1xHWD]
        # label = nn.Flatten(label) # [N, 1xHWD]

        # BCE Loss
        loss = nn.BCELoss(reduction="mean")
        BCE = loss(image, label)
        # Dice loss
        label_T = torch.transpose(label, 0, 1)
        num = torch.matmul(image, label_T)
        num = torch.sum(num, dim=[0,1]) + 1
        den = torch.add(image*image, label*label)
        den = torch.sum(den, dim=[0,1]) + 1
        den = torch.sum(image, -1) + torch.sum(label, -1) + 1 
        dice = 2*num/den
        return (0.5*(BCE + 1-dice), dice)

# Jaccard IOU Loss
class Jaccard(nn.Module):
    def __init__(self, weight=None):
        super(Jaccard, self).__init__()
    
    def forward(self, image, label):
        image = F.sigmoid(image)
        image = nn.Flatten(image).sum(-1)
        label = nn.Flatten(label).sum(-1)
        # Union = A + B - AB
        mul = (image*label).sum(-1)
        add = (image+label).sum(-1)
        union = add - mul
        return mul/(union+1e-7)

# Average Surface Distance
# 95 Hausdorff Distance
